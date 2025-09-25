#!/usr/bin/env python
# coding: utf-8

# # PiTE Transformer 

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
from tensorflow.keras import layers, Sequential
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


EMBEDDING = int(sys.argv[1])
EMBEDDING_TYPE = sys.argv[2]
EXPERIMENT = True if sys.argv[3] == "True" else False
BALANCED = True if sys.argv[4] == "True" else False


df_train = load_data(f'train_ProtBERT_{EMBEDDING}_{EMBEDDING_TYPE}.h5')
df_validation = load_data(f'validation_ProtBERT_{EMBEDDING}_{EMBEDDING_TYPE}.h5')


# Separate binding and non-binding samples
df_negative = df_train[df_train['binding'] == 0]
df_positive = df_train[df_train['binding'] == 1]


EPOCHS = 80
N_TRIALS = 1
TAG = 'PiTE'
GROUP_TAG = '1:1' if BALANCED else '1:5'
EMBEDDING_TAG = f'{EMBEDDING}-{EMBEDDING_TYPE}'
PROJECT = "dega/tcr"

NUM_POSITIVE_SAMPLES = len(df_positive)


if not EXPERIMENT:
    NUM_HEADS = 2
    FF_DIM = 32
    BATCH_SIZE = 32


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

    # Convert all to TensorFlow tensors
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


def generate_random_samples(data, balanced):
    num_negative_samples = NUM_POSITIVE_SAMPLES if balanced else NUM_POSITIVE_SAMPLES * 5
    
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

## Transformer Block
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
        })
        return cfg

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    

## The transformer-based model
def create_model(embed_dim, num_heads, ff_dim):
    X_tcr = Input(shape=(26, embed_dim), name="TCR_Input")
    X_epi = Input(shape=(24, embed_dim), name="Epitope_Input")
    
    transformer_block_tcr = TransformerBlock(embed_dim, num_heads, ff_dim)
    transformer_block_epi = TransformerBlock(embed_dim, num_heads, ff_dim)
    
    sembed_tcr = transformer_block_tcr(X_tcr)
    sembed_tcr = Activation('swish')(sembed_tcr)
    sembed_tcr = GlobalMaxPooling1D()(sembed_tcr)
    
    sembed_epi = transformer_block_epi(X_epi)
    sembed_epi = Activation('swish')(sembed_epi)
    sembed_epi = GlobalMaxPooling1D()(sembed_epi)
    
    diff = Subtract()([sembed_tcr, sembed_epi])  # u - v
    abs_diff = Lambda(lambda x: tf.abs(x), output_shape=lambda input_shape: input_shape)(diff)  # |u - v|
    concate = concatenate([sembed_tcr, sembed_epi, abs_diff])
    concate = Dense(1024)(concate)
    concate = BatchNormalization()(concate)
    concate = Dropout(0.3)(concate)
    concate = Activation('swish')(concate)
    concate = Dense(1, activation='sigmoid')(concate)
    
    model = Model(inputs = [X_tcr, X_epi], outputs=concate, name='Transformer_based_model')
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam')
    print(model.summary())
    return model


# ## Hyperparameter Tuning

# Custom F1 score metric
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
    
    run["parameters"] = {
        "ff_dim": FF_DIM,
        "num_heads": NUM_HEADS
    }
    
    model = create_model(
        embed_dim=EMBEDDING,
        ff_dim=FF_DIM,
        num_heads=NUM_HEADS
    )
    
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

