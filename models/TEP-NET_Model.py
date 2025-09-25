#!/usr/bin/env python
# coding: utf-8

# # V3 Attention - Feature Embedding Model

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, MultiHeadAttention, Dense, GlobalAveragePooling1D, Dropout, Concatenate, Layer, Flatten
from tensorflow.keras.callbacks import EarlyStopping
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
import gc


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

            yield df_chunk  # returns one DataFrame at a time

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


# Experiment Constants
EPOCHS = 75
N_TRIALS = 50
MODEL_TAG = 'V3'
GROUP_TAG = '1:1' if BALANCED else '1:5'
EMBEDDING_TAG = f'{EMBEDDING}-{EMBEDDING_TYPE}'
PROJECT = "dega/tcr"

# Program Constants
NUM_POSITIVE_SAMPLES = len(df_positive)
FEATURE_COLUMNS = ['TCR_KF7', 'TCR_KF1', 'TCR_hydrophobicity', 'TCR_aromaticity', 
                   'TCR_isoelectric_point', 'TCR_instability_index', 
                   'epitope_KF7', 'epitope_KF1','epitope_hydrophobicity', 'epitope_aromaticity',
                   'epitope_isoelectric_point', 'epitope_instability_index']


if not EXPERIMENT:
    ACTIVATION = "tanh"
    BATCH_SIZE = 64
    DROPOUT = 0.12210356074038448
    FF_DIM = 150
    L2_REG = 0.008226676611352674
    LEARNING_RATE = 0.008465284218810416
    NUM_LAYERS = 3
    OPTIMIZER = tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE, clipnorm=1.0)
    NUM_HEADS = 26
    EMBED_NUMERICAL = "PLE"


# ## Preprocess Data

def preprocess_data(df):
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Extract feature columns and labels
    X_features = df[FEATURE_COLUMNS].values
    y_labels = df['binding'].values

    # Convert all to TensorFlow tensors
    X_tcr = np.stack(df['TCR'].values)
    X_epitope = np.stack(df['epitope'].values)
    X_features = tf.convert_to_tensor(X_features, dtype=tf.float32)
    y_labels = tf.convert_to_tensor(y_labels, dtype=tf.float32)
    
    return X_tcr, X_epitope, X_features, y_labels

def preprocess_tpp(df):
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()
       
    # Extract feature columns and labels
    X_features = df[FEATURE_COLUMNS].values
    y_labels = df['binding'].values

    # Convert all to TensorFlow tensors
    X_tcr = np.stack(df['TCR'].values)
    X_epitope = np.stack(df['epitope'].values)
    X_features = tf.convert_to_tensor(X_features, dtype=tf.float32)
    y_labels = tf.convert_to_tensor(y_labels, dtype=tf.float32)
    
    return X_tcr, X_epitope, X_features, y_labels


X_positive_tcr, X_positive_epitope, X_positive_features, y_positive_labels = preprocess_data(df_positive)
X_negative_tcr, X_negative_epitope, X_negative_features, y_negative_labels = preprocess_data(df_negative)

training_data = {
        'positive': {
            'tcr': X_positive_tcr,
            'epitope': X_positive_epitope,
            'features': X_positive_features,
            'labels': np.ones(len(df_positive), dtype='float32')
        },
        'negative': {
            'tcr': X_negative_tcr,
            'epitope': X_negative_epitope,
            'features': X_negative_features,
            'labels':np.zeros(len(df_negative), dtype='float32')
        }
    }


X_validation_tcr, X_validation_epitope, X_validation_features, y_validation_labels = preprocess_data(df_validation)

if not EXPERIMENT:
    df_test = load_data(f'test_ProtBERT_{EMBEDDING}_{EMBEDDING_TYPE}.h5')
    X_test_tcr, X_test_epitope, X_test_features, y_test_labels = preprocess_data(df_test)


# ## Random Sample Generator

# Set a global random seed for reproducibility
SEED = int(sys.argv[5])
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
    X_features = tf.concat([
        tf.gather(data['positive']['features'], indices_positive),
        tf.gather(data['negative']['features'], indices_negative)
    ], axis=0)
    y_labels = tf.concat([
        tf.gather(data['positive']['labels'], indices_positive),
        tf.gather(data['negative']['labels'], indices_negative)
    ], axis=0)

    indices = tf.random.shuffle(tf.range(tf.shape(y_labels)[0]), seed=SEED)
    X_tcr = tf.gather(X_tcr, indices)
    X_epitope = tf.gather(X_epitope, indices)
    X_features = tf.gather(X_features, indices)
    y_labels = tf.gather(y_labels, indices)

    return X_tcr, X_epitope, X_features, y_labels

# Generator function
def train_generator(data, batch_size, balanced):
    while True:
        X_tcr, X_epitope, X_features, y_labels = generate_random_samples(data, balanced)
        for i in range(0, len(y_labels), batch_size):
            yield (
                {
                    "TCR_Input": X_tcr[i:i + batch_size],
                    "Epitope_Input": X_epitope[i:i + batch_size],
                    "Physicochemical_Features": X_features[i:i + batch_size]
                },
                y_labels[i:i + batch_size]
            )


# ## Embedding Physicochemical Features

class PiecewiseLinearEncoding(Layer):
    def __init__(self, bins, **kwargs):
        super(PiecewiseLinearEncoding, self).__init__(**kwargs)
        self.bins = tf.convert_to_tensor(bins, dtype=tf.float32)
        self.num_bins = len(bins) - 1

    def call(self, inputs):
        # Expand input to shape [batch_size, num_features, 1]
        inputs_expanded = tf.expand_dims(inputs, axis=-1)
        
        # Compute the widths of bins
        bin_widths = self.bins[1:] - self.bins[:-1]

        # Compute piecewise linear encoding
        bin_edges = (inputs_expanded - self.bins[:-1]) / bin_widths
        bin_edges = tf.clip_by_value(bin_edges, 0.0, 1.0)

        return bin_edges

    # For Keras serialization
    def get_config(self):
        config = super(PiecewiseLinearEncoding, self).get_config()
        config.update({
            "bins": self.bins.numpy().tolist()  # Convert tensor to list for serialization
        })
        return config

class PeriodicEmbeddings(Layer):
    def __init__(self, num_frequencies=16, **kwargs):
        super(PeriodicEmbeddings, self).__init__(**kwargs)
        self.num_frequencies = num_frequencies
        self.freqs = tf.Variable(
            initial_value=tf.random.uniform(
                shape=(num_frequencies,), minval=0.1, maxval=1.0
            ),
            trainable=True,
        )

    def call(self, inputs):
        # Shape of inputs: [batch_size, num_features]
        inputs_expanded = tf.expand_dims(inputs, axis=-1)  # [batch_size, num_features, 1]
        periodic_features = tf.concat(
            [
                tf.sin(2 * np.pi * inputs_expanded * self.freqs),
                tf.cos(2 * np.pi * inputs_expanded * self.freqs),
            ],
            axis=-1,
        )  # [batch_size, num_features, 2*num_frequencies]

        return periodic_features

    # For Keras serialization
    def get_config(self):
        config = super(PeriodicEmbeddings, self).get_config()
        config.update({
            "num_frequencies": self.num_frequencies
        })
        return config


# Define bins for piecewise linear encoding
bins = np.linspace(0.0, 1.0, num=11)

# Create layers
PLE = PiecewiseLinearEncoding(bins)
Periodic = PeriodicEmbeddings(num_frequencies=16)


# ## Model Architecture

class ExpandDimsLayer(Layer):
    def __init__(self, axis=1, **kwargs):
        super(ExpandDimsLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.expand_dims(inputs, axis=self.axis)

    def get_config(self):
        config = super(ExpandDimsLayer, self).get_config()
        config.update({
            "axis": self.axis
        })
        return config


import tensorflow as tf
from tensorflow.keras.layers import Input, MultiHeadAttention, Dense, GlobalAveragePooling1D, Dropout, Concatenate, Reshape, Lambda, Flatten
from tensorflow.keras.models import Model

def create_model(embed_dim, ff_dim, feature_dim, dropout_rate, activation, l2_reg, num_layers, num_heads, embed_numerical):
    # Input layers for precomputed embeddings
    tcr_input = Input(shape=(26, embed_dim), name="TCR_Input")
    epitope_input = Input(shape=(24, embed_dim), name="Epitope_Input")
    feature_input = Input(shape=(feature_dim,), name="Physicochemical_Features")
    
    # Attention input
    attention_input = tcr_input
    
    # Multi-head attention for interaction between TCR and Epitope embeddings
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(
        query=attention_input, 
        value=epitope_input
    )
    
    # Pooling layer to aggregate the attention output
    x = GlobalAveragePooling1D()(attention_output)

    # Embed numerical features (using PLE or periodic activations)
    feature_embeddings = PLE(feature_input) if embed_numerical == "PLE" else Periodic(feature_input)

    feature_embeddings_flatten = Flatten()(feature_embeddings)

    # Concatenate the pooled attention output with the embedded physicochemical features
    x = Concatenate(name="Concatenate_Protein_Physicochemical")([x, feature_embeddings_flatten])

    # Feed-forward network
    for _ in range(num_layers):
        x = Dense(ff_dim, activation=activation, kernel_regularizer=l2(l2_reg))(x)
        x = Dropout(dropout_rate)(x)

    # Output layer (binary classification for binding prediction)
    output = Dense(1, activation="sigmoid", name="Binding_Output")(x)

    # Model definition
    model = Model(inputs=[tcr_input, epitope_input, feature_input], outputs=output)
    return model


# ## Hyperparameter Tuning

# Custom F1 score metric
def f1(precision, recall):
    if precision + recall == 0:
        return 0.0
    return round((2 * precision * recall) / (precision + recall), 6)

def f2(precision, recall):
    if precision + recall == 0:
        return 0.0
    return round(5 * (precision * recall) / ((4 * precision) + recall), 6)

def safe_log_metric(run, key, value):
    if not np.isnan(value) and not np.isinf(value):
        run[key].append(value)
    else:
        print(f"Skipping logging {key}: {value} is not valid.")

def objective_with_logging(trial):
    ff_dim = trial.suggest_int('ff_dim', 16, 200)
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64, 128, 256])
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2)
    activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'leaky_relu'])
    l2_reg = trial.suggest_float('l2_regularization', 0.0, 0.01)
    num_layers = trial.suggest_int('num_layers', 1, 10)
    num_heads = trial.suggest_int('num_heads', 2, 50)
    embed_numerical = trial.suggest_categorical('embed_numerical', ['PLE', 'Periodic'])

    optimizer_name = 'sgd'
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, clipnorm=1.0)

    # Create a trial-level Neptune run
    run_trial_level = neptune.init_run(
        project=PROJECT,
        name=f"{MODEL_TAG} {GROUP_TAG} {trial.number}",
        api_token=os.getenv("NEPTUNE_API_KEY"),
        tags=[MODEL_TAG, EMBEDDING_TAG, "new"]
    )

    # Log study name and trial metadata to trial-level run
    run_trial_level["sys/group_tags"].add([GROUP_TAG])
    run_trial_level["trial/number"] = trial.number
    run_trial_level["trial/parameters"] = {
        "ff_dim": ff_dim,
        "dropout_rate": dropout_rate,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "activation": activation,
        "optimizer_name": optimizer_name,
        "l2_reg": l2_reg,
        "num_layers": num_layers,   
        "num_heads": num_heads,
        "embed_numerical": embed_numerical,
        "seed": SEED,
    }

    
    # Create the model
    model = create_model(
        embed_dim=EMBEDDING,
        ff_dim=ff_dim,
        feature_dim=12,
        dropout_rate=dropout_rate,
        activation=activation,
        l2_reg=l2_reg,
        num_layers=num_layers,
        num_heads=num_heads,
        embed_numerical=embed_numerical
    )
    
    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', 
                 tf.keras.metrics.AUC(name='pr_auc', curve='PR'),
                 tf.keras.metrics.AUC(name='roc_auc', curve='ROC'), 
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall'),
                ]
    )
    
    # Define Early Stopping callback
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )
    
    # Train the model
    history = model.fit(
        train_generator(training_data, batch_size, BALANCED),
        validation_data=(
            {
                "TCR_Input": tf.convert_to_tensor(X_validation_tcr),
                "Epitope_Input": tf.convert_to_tensor(X_validation_epitope),
                "Physicochemical_Features": tf.convert_to_tensor(X_validation_features)
            },
            tf.convert_to_tensor(y_validation_labels)
        ),
        epochs=EPOCHS,
        steps_per_epoch=int(NUM_POSITIVE_SAMPLES/batch_size),
        batch_size=batch_size,
        verbose=0,
        callbacks=[early_stopping]
    )
    
    # Log metrics during training
    for epoch in range(len(history.history['loss'])):
        safe_log_metric(run_trial_level, "train/accuracy", history.history['accuracy'][epoch])
        safe_log_metric(run_trial_level, "train/loss", history.history['loss'][epoch])
        safe_log_metric(run_trial_level, "val/accuracy", history.history['val_accuracy'][epoch])
        safe_log_metric(run_trial_level, "val/loss", history.history['val_loss'][epoch])
        safe_log_metric(run_trial_level, "val/pr_auc", history.history['val_pr_auc'][epoch])
        safe_log_metric(run_trial_level, "val/roc_auc", history.history['val_roc_auc'][epoch])
        safe_log_metric(run_trial_level, "val/precision", history.history['val_precision'][epoch])
        safe_log_metric(run_trial_level, "val/recall", history.history['val_recall'][epoch])
        safe_log_metric(run_trial_level, "val/f1_score", f1(history.history['val_precision'][epoch], history.history['val_recall'][epoch]))
        safe_log_metric(run_trial_level, "val/f2_score", f2(history.history['val_precision'][epoch], history.history['val_recall'][epoch]))

    # Calculate the score (validation roc auc)
    validation_roc_auc = max(history.history['val_roc_auc'])
    
    # Log the score to trial-level run
    run_trial_level["trial/score"] = validation_roc_auc

    # Stop trial-level run
    run_trial_level.stop()

    # Clear TensorFlow session to free GPU memory
    tf.keras.backend.clear_session()
    gc.collect()

    return validation_roc_auc


# ## Experiments

if EXPERIMENT:
    # Create the Optuna study with TPE sampler
    load_dotenv()
    run_study_level = neptune.init_run(
        project=PROJECT,
        name=f"Study {MODEL_TAG} {GROUP_TAG} {EMBEDDING} {EMBEDDING_TYPE}",
        api_token=os.getenv("NEPTUNE_API_KEY"),
        tags=["study"],
    )
    
    # Disable logging to output
    neptune_logger = logging.getLogger("neptune")
    neptune_logger.setLevel(logging.WARNING)
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=SEED)
    )
    
    
    
    run_study_level["sys/group_tags"].add([study.study_name])
    
    neptune_callback = NeptuneCallback(run_study_level)
    
    study.optimize(
        objective_with_logging,
        n_trials=N_TRIALS,
        callbacks=[neptune_callback]
    )
    
    run_study_level.stop()


# ## Train best Model

from neptune.integrations.tensorflow_keras import NeptuneCallback

if not EXPERIMENT:    
    load_dotenv()
    run = neptune.init_run(
        project=PROJECT,
        name=f"Best {MODEL_TAG} {GROUP_TAG}",
        api_token=os.getenv("NEPTUNE_API_KEY"),
        tags=[MODEL_TAG, GROUP_TAG, EMBEDDING_TAG, "best", "new"],
    )

    run["sys/group_tags"].add([GROUP_TAG])
    
    # Disable logging to output
    neptune_logger = logging.getLogger("neptune")
    neptune_logger.setLevel(logging.WARNING)
    
    run["parameters"] = {
        "ff_dim": FF_DIM,
        "dropout_rate": DROPOUT,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "activation": ACTIVATION,
        "optimizer_name": OPTIMIZER,
        "l2_reg": L2_REG,
        "num_layers": NUM_LAYERS,
        "num_heads": NUM_HEADS,
        "embed_numerical": EMBED_NUMERICAL
    }
    
    # Create the model
    model = create_model(
        embed_dim=EMBEDDING,
        ff_dim=FF_DIM,
        feature_dim=12,
        dropout_rate=DROPOUT,
        activation=ACTIVATION,
        l2_reg=L2_REG,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        embed_numerical=EMBED_NUMERICAL
    )
    
    # Compile the model
    model.compile(
        optimizer=OPTIMIZER,
        loss='binary_crossentropy',
        metrics=['accuracy', 
                 tf.keras.metrics.AUC(name='pr_auc', curve='PR'), 
                 tf.keras.metrics.AUC(name='roc_auc', curve='ROC'), 
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall'),
                 ]
    )
    
    # Define Early Stopping callback
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    neptune_callback = NeptuneCallback(run=run, base_namespace="metrics")
    
    # Train the model
    history = model.fit(
        train_generator(training_data, BATCH_SIZE, BALANCED),
        validation_data=(
            {
                "TCR_Input": tf.convert_to_tensor(X_validation_tcr),
                "Epitope_Input": tf.convert_to_tensor(X_validation_epitope),
                "Physicochemical_Features": tf.convert_to_tensor(X_validation_features)
            },
            tf.convert_to_tensor(y_validation_labels)
        ),
        epochs=EPOCHS,
        steps_per_epoch=int(NUM_POSITIVE_SAMPLES/BATCH_SIZE),
        batch_size=BATCH_SIZE,
        verbose=0,
        callbacks=[early_stopping, neptune_callback]
    )
    
    # Log metrics during training
    for epoch in range(len(history.history['loss'])):
        safe_log_metric(run, "train/accuracy", history.history['accuracy'][epoch])
        safe_log_metric(run, "train/loss", history.history['loss'][epoch])
        safe_log_metric(run, "val/accuracy", history.history['val_accuracy'][epoch])
        safe_log_metric(run, "val/loss", history.history['val_loss'][epoch])
        safe_log_metric(run, "val/pr_auc", history.history['val_pr_auc'][epoch])
        safe_log_metric(run, "val/roc_auc", history.history['val_roc_auc'][epoch])
        safe_log_metric(run, "val/precision", history.history['val_precision'][epoch])
        safe_log_metric(run, "val/recall", history.history['val_recall'][epoch])
        safe_log_metric(run, "val/f1_score", f1(history.history['val_precision'][epoch], history.history['val_recall'][epoch]))
        safe_log_metric(run, "val/f2_score", f2(history.history['val_precision'][epoch], history.history['val_recall'][epoch]))

    # Test the model
    test_metrics = model.evaluate(
        {
            "TCR_Input": tf.convert_to_tensor(X_test_tcr),
            "Epitope_Input": tf.convert_to_tensor(X_test_epitope),
            "Physicochemical_Features": tf.convert_to_tensor(X_test_features)
        },
        tf.convert_to_tensor(y_test_labels),
        verbose=0,
        return_dict=True
    )
    
    # Log test metrics
    for name in test_metrics:
        safe_log_metric(run, f"test/{name}", test_metrics[name])

    safe_log_metric(run, "test/f1_score", f1(test_metrics['precision'], test_metrics['recall']))
    safe_log_metric(run, "test/f2_score", f2(test_metrics['precision'], test_metrics['recall']))


# ## Evaluate TPP

if not EXPERIMENT:
    # Ensure raw sequences are decoded from bytes (e.g., b'CLVGMI') to proper strings (e.g., 'CLVGMI')
    def decode_if_bytes(val):
        if isinstance(val, bytes):
            return val.decode('utf-8')
        elif isinstance(val, str) and val.startswith("b'") and val.endswith("'"):
            return val[2:-1]  # Remove b'' notation
        return str(val)

    # Build sets of seen sequences (decoded)
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
    
    # Then use iloc instead of loc
    df_tpp1 = df_test.iloc[tpp1]
    df_tpp2 = df_test.iloc[tpp2]
    df_tpp3 = df_test.iloc[tpp3]
    df_tpp4 = df_test.iloc[tpp4]

    print(f"TPP1: {len(df_tpp1)}, TPP2: {len(df_tpp2)}, TPP3: {len(df_tpp3)}, TPP4: {len(df_tpp4)}")

    def predict_tpp(num_tpp, df_tpp):
        X_test_tcr, X_test_epitope, X_test_features, y_test_labels = preprocess_tpp(df_tpp)
        y_pred_probs  = model.predict(
            {
                "TCR_Input": tf.convert_to_tensor(X_test_tcr),
                "Epitope_Input": tf.convert_to_tensor(X_test_epitope),
                "Physicochemical_Features": tf.convert_to_tensor(X_test_features)
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
    model.save(f'{MODEL_TAG}_model_{sample}_{EMBEDDING}_{EMBEDDING_TYPE}.keras')
