#!/usr/bin/env python
# coding: utf-8

# # EPIC-TRACE Model

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, GlobalMaxPooling1D, concatenate, Dense, BatchNormalization, Dropout
from tensorflow.keras.layers import Bidirectional, TimeDistributed, LSTM, Conv1D, Add, Activation, Subtract, Lambda
from keras.callbacks import LambdaCallback,TensorBoard,ReduceLROnPlateau, EarlyStopping
from tensorflow.math import subtract
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import AUC, Precision, Recall
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import neptune
import neptune.integrations.optuna as optuna_utils
from neptune.integrations.optuna import NeptuneCallback
import optuna
from optuna.samplers import TPESampler
import os
from dotenv import load_dotenv, dotenv_values
import logging
import h5py
import sys


path = '/scratch/'

def load_hdf5_in_chunks(h5_path, chunk_size=5000):
    with h5py.File(h5_path, 'r') as f:
        total_samples = f['TCR'].shape[0]
        meta_keys = list(f['meta'].keys())

        for i in range(0, total_samples, chunk_size):
            end = min(i + chunk_size, total_samples)

            # Load chunk of embeddings
            TCR_chunk = f['TCR'][i:end]
            epitope_chunk = f['epitope'][i:end]

            # Load metadata chunk
            meta_chunk = {
                key: f[f'meta/{key}'][i:end] for key in meta_keys
            }

            # Convert to pandas DataFrame
            df_chunk = pd.DataFrame(meta_chunk)
            df_chunk['TCR'] = list(TCR_chunk)
            df_chunk['epitope'] = list(epitope_chunk)

            yield df_chunk

def load_data(filename):
    df = pd.DataFrame()
    
    for df_chunk in load_hdf5_in_chunks(f'{path}{filename}', chunk_size=5000):
        df = pd.concat([df, df_chunk], sort=False)  
        
    return df


EMBEDDING = int(sys.argv[1]) # 64
EMBEDDING_TYPE = sys.argv[2] # PCA / AE
EXPERIMENT = True if sys.argv[3] == "True" else False
BALANCED = True if sys.argv[4] == "True" else False


df_train = load_data(f'train_ProtBERT_{EMBEDDING}_{EMBEDDING_TYPE}.h5')
df_validation = load_data(f'validation_ProtBERT_{EMBEDDING}_{EMBEDDING_TYPE}.h5')


# Separate binding and non-binding samples
df_negative = df_train[df_train['binding'] == 0]
df_positive = df_train[df_train['binding'] == 1]


EPOCHS = 80
N_TRIALS = 1
TAG = 'EPIC-TRACE'
GROUP_TAG = '1:1' if BALANCED else '1:5'
EMBEDDING_TAG = f'{EMBEDDING}-{EMBEDDING_TYPE}'
PROJECT = "dega/tcr"

NUM_POSITIVE_SAMPLES = len(df_positive)


if not EXPERIMENT:
    EMBED_DIM = EMBEDDING
    CONV_FILTERS = 100
    CONV_KERNELS = 7
    NUM_HEADS = 5
    DROPOUT = 0.2
    DROPOUT2 = 0.45
    ATTN_DROPOUT = 0.2
    BATCH_SIZE = 128
    NUM_TF_BLOCK = 0
    USE_FF = False
    USE_POS_ENCODING = False


# ## Preprocess Data

def preprocess_data(df):
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    y_labels = df['binding'].values

    # Convert all to TensorFlow tensors
    X_tcr = np.stack(df['TCR'].values)
    X_epitope = np.stack(df['epitope'].values)
    y_labels = tf.convert_to_tensor(y_labels, dtype=tf.float32)
    
    return X_tcr, X_epitope, y_labels

def preprocess_tpp(df):
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()
       
    y_labels = df['binding'].values

    X_tcr = np.stack(df['TCR'].values)
    X_epitope = np.stack(df['epitope'].values)
    y_labels = tf.convert_to_tensor(y_labels, dtype=tf.float32)
    
    return X_tcr, X_epitope, y_labels


X_positive_tcr, X_positive_epitope, y_positive_labels = preprocess_data(df_positive)
X_negative_tcr, X_negative_epitope, y_negative_labels = preprocess_data(df_negative)

training_data = {
        'positive': {
            'tcr': X_positive_tcr,
            'epitope': X_positive_epitope,
            'labels': np.ones(len(df_positive), dtype='float32')
        },
        'negative': {
            'tcr': X_negative_tcr,
            'epitope': X_negative_epitope,
            'labels':np.zeros(len(df_negative), dtype='float32')
        }
    }


X_validation_tcr, X_validation_epitope, y_validation_labels = preprocess_data(df_validation)

if not EXPERIMENT:
    df_test = load_data(f'test_ProtBERT_{EMBEDDING}_{EMBEDDING_TYPE}.h5')
    X_test_tcr, X_test_epitope, y_test_labels = preprocess_data(df_test)


# ## Random Sample Generator

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# Sampling function
def generate_random_samples(data, balanced):
    num_negative_samples = NUM_POSITIVE_SAMPLES if balanced else NUM_POSITIVE_SAMPLES * 5
    
    # Sample random negative samples and use all positive samples
    indices_negative = np.random.choice(len(data['negative']['labels']), size=NUM_POSITIVE_SAMPLES, replace=False)
    indices_positive = np.arange(len(data['positive']['labels']))

    indices_negative = tf.convert_to_tensor(indices_negative, dtype=tf.int32)
    indices_positive = tf.convert_to_tensor(indices_positive, dtype=tf.int32)

    X_tcr = tf.concat([
        tf.gather(data['positive']['tcr'], indices_positive),
        tf.gather(data['negative']['tcr'], indices_negative)
    ], axis=0)
    X_epitope = tf.concat([
        tf.gather(data['positive']['epitope'], indices_positive),
        tf.gather(data['negative']['epitope'], indices_negative)
    ], axis=0)
    y_labels = tf.concat([
        tf.gather(data['positive']['labels'], indices_positive),
        tf.gather(data['negative']['labels'], indices_negative)
    ], axis=0)

    indices = tf.random.shuffle(tf.range(tf.shape(y_labels)[0]), seed=SEED)
    X_tcr = tf.gather(X_tcr, indices)
    X_epitope = tf.gather(X_epitope, indices)
    y_labels = tf.gather(y_labels, indices)

    return X_tcr, X_epitope, y_labels

def train_generator(data, batch_size, balanced):
    while True:
        X_tcr, X_epitope, y_labels = generate_random_samples(data, balanced)
        for i in range(0, len(y_labels), batch_size):
            yield (
                {
                    "TCR_Input": X_tcr[i:i + batch_size],
                    "Epitope_Input": X_epitope[i:i + batch_size]
                },
                y_labels[i:i + batch_size]
            )


# ## Model Architecture

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, sequence_len, embed_dim):
        super().__init__()
        self.pos_encoding = self.get_positional_encoding(sequence_len, embed_dim)

    def get_positional_encoding(self, seq_len, d_model):
        angle_rads = self.get_angles(
            np.arange(seq_len)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model
        )
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        return tf.constant(angle_rads[np.newaxis, ...], dtype=tf.float32)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

def transformer_block(x, embed_dim, num_heads, ff_dim, dropout=0.1):
    attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x, x)
    attn_output = layers.Dropout(dropout)(attn_output)
    x1 = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)

    ffn = layers.Dense(ff_dim, activation='relu')(x1)
    ffn = layers.Dropout(dropout)(ffn)
    ffn = layers.Dense(embed_dim)(ffn)
    x2 = layers.LayerNormalization(epsilon=1e-6)(x1 + ffn)
    return x2

def create_model(embed_dim=64, tcr_len=26, epitope_len=24, conv_filters=[100], conv_kernels=[7], num_heads=5,
                num_tf_blocks=0, use_ff=False, use_pos_encoding=False, dropout=0.2, dropout2= 0.45, attn_dropout=0.2):
    
    # Inputs
    tcr_input = tf.keras.Input(shape=(tcr_len, embed_dim), name='TCR_Input')
    epitope_input = tf.keras.Input(shape=(epitope_len, embed_dim), name='Epitope_Input')

    # Conv1D over TCR
    tcr_conv = layers.Concatenate(axis=-1)([
        layers.Conv1D(filters=f, kernel_size=k, padding='same', activation='relu')(tcr_input)
        for f, k in zip(conv_filters, conv_kernels)
    ])
    tcr_conv = layers.Dropout(dropout)(tcr_conv)

    # Conv1D over Epitope
    epitope_conv = layers.Concatenate(axis=-1)([
        layers.Conv1D(filters=f, kernel_size=k, padding='same', activation='relu')(epitope_input)
        for f, k in zip(conv_filters, conv_kernels)
    ])
    epitope_conv = layers.Dropout(dropout)(epitope_conv)

    if use_ff:
        ff_layer = layers.Dense(50, activation='relu')
        tcr_ff = ff_layer(tcr_input)
        epi_ff = ff_layer(epitope_input)
        tcr_comb = layers.Concatenate(axis=-1)([tcr_conv, tcr_ff])
        epitope_comb = layers.Concatenate(axis=-1)([epitope_conv, epi_ff])
    else:
        tcr_comb = tcr_conv
        epitope_comb = epitope_conv

    # Merge TCR + Epitope along time dimension
    x = layers.Concatenate(axis=1)([tcr_comb, epitope_comb])

    if use_pos_encoding:
        x = PositionalEncoding(sequence_len=tcr_len + epitope_len, embed_dim=x.shape[-1])(x)

    if num_tf_blocks == 0:
        x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=x.shape[-1])(x, x)
    else:
        for _ in range(num_tf_blocks):
            x = transformer_block(x, embed_dim=x.shape[-1], num_heads=num_heads, ff_dim=int(1.5 * x.shape[-1]), dropout=attn_dropout)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout2)(x)
    output = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[tcr_input, epitope_input], outputs=output)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", 'AUC'])
    return model


def f1(precision, recall):
    return round((2*(precision * recall)) / (precision + recall), 6)

def safe_log_metric(run, key, value):
    if not np.isnan(value) and not np.isinf(value):
        run[key].append(value)
    else:
        print(f"Skipping logging {key}: {value} is not valid.")


# ## Train best Model

from neptune.integrations.tensorflow_keras import NeptuneCallback

if not EXPERIMENT:    
    load_dotenv()
    run = neptune.init_run(
        project=PROJECT,
        name=f"Best {TAG} {GROUP_TAG}",
        api_token=os.getenv("NEPTUNE_API_KEY"),
        tags=[TAG, GROUP_TAG, EMBEDDING_TAG, "best"],
    )

    run["sys/group_tags"].add([GROUP_TAG])
    
    # Disable logging to output
    neptune_logger = logging.getLogger("neptune")
    neptune_logger.setLevel(logging.WARNING)
    
    model = create_model(embed_dim=EMBED_DIM, tcr_len=26, epitope_len=24, conv_filters=[CONV_FILTERS], conv_kernels=[CONV_KERNELS], num_heads=NUM_HEADS,
                num_tf_blocks=NUM_TF_BLOCK, use_ff=USE_FF, use_pos_encoding=USE_POS_ENCODING, dropout=DROPOUT, dropout2=DROPOUT2, attn_dropout=ATTN_DROPOUT)
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 
                 tf.keras.metrics.AUC(name='pr_auc', curve='PR'),
                 tf.keras.metrics.AUC(name='roc_auc', curve='ROC'),
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')
                ]
    )
    
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=30,
        restore_best_weights=True,
        mode='min',
    )
    
    lrate_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_delta=0.0001, min_lr=1e-6, verbose=0)
    neptune_callback = NeptuneCallback(run=run, base_namespace="metrics")
    
    # Train the model
    history = model.fit(
        train_generator(training_data, BATCH_SIZE, BALANCED),
        validation_data=(
            {
                "TCR_Input": tf.convert_to_tensor(X_validation_tcr),
                "Epitope_Input": tf.convert_to_tensor(X_validation_epitope)
            },
            tf.convert_to_tensor(y_validation_labels)
        ),
        epochs=EPOCHS,
        steps_per_epoch=int(NUM_POSITIVE_SAMPLES/BATCH_SIZE),
        batch_size=BATCH_SIZE,
        verbose=0,
        callbacks=[early_stopping, lrate_scheduler, neptune_callback]
    )
    
    for epoch in range(len(history.history['loss'])):
        safe_log_metric(run, "train/accuracy", history.history['accuracy'][epoch])
        safe_log_metric(run, "train/loss", history.history['loss'][epoch])
        safe_log_metric(run, "val/accuracy", history.history['val_accuracy'][epoch])
        safe_log_metric(run, "val/loss", history.history['val_loss'][epoch])
        safe_log_metric(run, "val/pr_auc", history.history['val_pr_auc'][epoch])
        safe_log_metric(run, "val/roc_auc", history.history['val_roc_auc'][epoch])
        safe_log_metric(run, "val/precision", history.history['val_precision'][epoch])
        safe_log_metric(run, "val/recall", history.history['val_recall'][epoch])

    test_metrics = model.evaluate(
        {
            "TCR_Input": tf.convert_to_tensor(X_test_tcr),
            "Epitope_Input": tf.convert_to_tensor(X_test_epitope)
        },
        tf.convert_to_tensor(y_test_labels),
        verbose=0,
        return_dict=True
    )
    
    for name in test_metrics:
        safe_log_metric(run, f"test/{name}", test_metrics[name])

    safe_log_metric(run, "test/f1_score", f1(test_metrics['precision'], test_metrics['recall']))


# ## Evaluate TPP

if not EXPERIMENT:
    def decode_if_bytes(val):
        if isinstance(val, bytes):
            return val.decode('utf-8')
        elif isinstance(val, str) and val.startswith("b'") and val.endswith("'"):
            return val[2:-1]  # Remove b'' notation
        return str(val)

    train_tcr_raw = set(df_train['TCR_raw'].apply(decode_if_bytes))
    train_epitope_raw = set(df_train['epitope_raw'].apply(decode_if_bytes))

    tpp1, tpp2, tpp3, tpp4 = [], [], [], []

    for i, (_, row) in enumerate(df_test.iterrows()):
        tcr_raw = decode_if_bytes(row['TCR_raw'])
        epitope_raw = decode_if_bytes(row['epitope_raw'])
    
        tcr_seen = tcr_raw in train_tcr_raw
        epitope_seen = epitope_raw in train_epitope_raw
    
        if tcr_seen and epitope_seen:
            tpp1.append(i)
        elif not tcr_seen and epitope_seen:
            tpp2.append(i)
        elif not tcr_seen and not epitope_seen:
            tpp3.append(i)
        elif tcr_seen and not epitope_seen:
            tpp4.append(i)
    
    df_tpp1 = df_test.iloc[tpp1]
    df_tpp2 = df_test.iloc[tpp2]
    df_tpp3 = df_test.iloc[tpp3]
    df_tpp4 = df_test.iloc[tpp4]

    print(f"TPP1: {len(df_tpp1)}, TPP2: {len(df_tpp2)}, TPP3: {len(df_tpp3)}, TPP4: {len(df_tpp4)}")

    def predict_tpp(num_tpp, df_tpp):
        X_test_tcr, X_test_epitope, y_test_labels = preprocess_tpp(df_tpp)
        y_pred_probs  = model.predict(
            {
                "TCR_Input": tf.convert_to_tensor(X_test_tcr),
                "Epitope_Input": tf.convert_to_tensor(X_test_epitope),
            }
        )
        
        y_pred = (y_pred_probs > 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test_labels, y_pred).ravel()

        run["tpp"].log(f"{num_tpp}: True Positives (TP): {tp}, True Negatives (TN): {tn}, False Positives (FP): {fp}, False Negatives (FN): {fn}")
    
    predict_tpp("TPP1", df_tpp1)
    predict_tpp("TPP2", df_tpp2)
    predict_tpp("TPP3", df_tpp3)
    predict_tpp("TPP4", df_tpp4)

    run.stop()

    # Save model
    sample = "1-1" if BALANCED else "1-5"
    model.save(f'{TAG}_model_{sample}_{EMBEDDING}_{EMBEDDING_TYPE}.keras')

