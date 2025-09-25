#!/usr/bin/env python
# coding: utf-8

import h5py
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping
import keras
from sklearn.metrics.pairwise import cosine_similarity
import sys

path = '/scratch/'

def load_hdf5_TCR_in_chunks(h5_path, chunk_size=5000):
    with h5py.File(h5_path, 'r') as f:
        total_samples = f['TCR'].shape[0]

        for i in range(0, total_samples, chunk_size):
            end = min(i + chunk_size, total_samples)

            # Load chunk of embeddings
            TCR_chunk = f['TCR'][i:end]

            # Convert to pandas DataFrame
            df_chunk = pd.DataFrame()
            df_chunk['TCR'] = list(TCR_chunk)

            yield df_chunk

def load_data_tcr(filename):
    df = pd.DataFrame()
    
    for df_chunk in load_hdf5_TCR_in_chunks(f'{path}{filename}', chunk_size=5000):
        df = pd.concat([df, df_chunk], sort=False)

    return df

def load_hdf5_epitope_in_chunks(h5_path, chunk_size=5000):
    with h5py.File(h5_path, 'r') as f:
        total_samples = f['epitope'].shape[0]

        for i in range(0, total_samples, chunk_size):
            end = min(i + chunk_size, total_samples)

            # Load chunk of embeddings
            epitope_chunk = f['epitope'][i:end]

            # Convert to pandas DataFrame
            df_chunk = pd.DataFrame()
            df_chunk['epitope'] = list(epitope_chunk)

            yield df_chunk

def load_data_epitope(filename):
    df = pd.DataFrame()
    
    for df_chunk in load_hdf5_epitope_in_chunks(f'{path}{filename}', chunk_size=5000):
        df = pd.concat([df, df_chunk], sort=False)

    return df

def prepare_data(sequence):    
    X_train = np.stack(df_train[sequence].values)
    X_validation = np.stack(df_validation[sequence].values)
    X_test = np.stack(df_test[sequence].values)

    return X_train, X_validation, X_test


# Positional Encoding
def positional_encoding(length, depth):
    positions = np.arange(length)[:, np.newaxis]
    depths = np.arange(depth)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (depths // 2)) / np.float32(depth))
    angle_rads = positions * angle_rates

    pos_encoding = np.zeros((length, depth))
    pos_encoding[:, 0::2] = np.sin(angle_rads[:, 0::2])
    pos_encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])

    return tf.constant(pos_encoding, dtype=tf.float32)

# Transformer Encoder Block
def transformer_encoder_block(x, head_size=64, num_heads=4, ff_dim=256, dropout=0.1):
    # Self-attention
    attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(x, x)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    # Feed-forward
    ff = layers.Dense(ff_dim, activation='relu')(x)
    ff = layers.Dense(x.shape[-1])(ff)
    x = layers.Add()([x, ff])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    return x

# Autoencoder Model
def build_autoencoder(max_length=26, input_dim=1024, bottleneck_dim=64):
    input_layer = layers.Input(shape=(max_length, input_dim))

    # Positional Encoding
    pos_encoding = positional_encoding(max_length, input_dim)
    x = input_layer + pos_encoding

    # Encoder
    x = transformer_encoder_block(x, head_size=64, num_heads=4, ff_dim=512)
    encoded = layers.TimeDistributed(layers.Dense(bottleneck_dim))(x)

    # Decoder
    x = transformer_encoder_block(encoded, head_size=32, num_heads=2, ff_dim=256)
    decoded = layers.TimeDistributed(layers.Dense(input_dim))(x)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    encoder = Model(inputs=input_layer, outputs=encoded)

    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder, encoder


def reconstruction_error(X, X_rec):
    # Mean squared error per sample
    return np.mean((X - X_rec) ** 2, axis=(1, 2)) 


def cosine_sim(X, X_rec):
    X_flat = X.reshape(X.shape[0], -1)
    Xr_flat = X_rec.reshape(X_rec.shape[0], -1)

    X_flat = np.nan_to_num(X_flat, nan=0.0)
    Xr_flat = np.nan_to_num(Xr_flat, nan=0.0)

    sims = [cosine_similarity([a], [b])[0][0] for a, b in zip(X_flat, Xr_flat)]
    return np.array(sims)


def evaluate_autoencoder(model, X, label="Set"):
    X_rec = model.predict(X, verbose = 0)
    mse = reconstruction_error(X, X_rec)
    cos = cosine_sim(X, X_rec)

    print(f"Evaluation on {label}:")
    print(f"MSE:  Mean={mse.mean():.6f}, Std={mse.std():.6f}")
    print(f"Cosine Similarity: Mean={cos.mean():.6f}, Std={cos.std():.6f}")

    return cos.mean()


def train_best_encoder(sequence, embedding):
    if sequence == 'TCR':
        max_length = 26
    else:
        max_length = 24

    # Find best hyperparameters
    batch_sizes = [32, 64, 128]
    best_cos = 0
    best_batch = 0
    
    for batch_size in batch_sizes:
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3, restore_best_weights=True)
        autoencoder, encoder = build_autoencoder(max_length=max_length, input_dim=1024, bottleneck_dim=embedding)
        autoencoder.fit(
            X_train, X_train,
            epochs=20,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(X_validation, X_validation),
            verbose=0,
            callbacks=es
        )
        
        cos = evaluate_autoencoder(autoencoder, X_test, "Test")
        if best_cos < cos:
            best_cos = cos
            best_batch = batch_size
    
    print(f"Batch size: {best_batch}, Cosine similarity: {best_cos}")

    # Train best autoencoder
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3, restore_best_weights=True)
    autoencoder, encoder = build_autoencoder(max_length=max_length, input_dim=1024, bottleneck_dim=embedding)
    autoencoder.fit(
        X_train, X_train,
        epochs=75,
        batch_size=best_batch,
        shuffle=True,
        validation_data=(X_validation, X_validation),
        verbose=0,
        callbacks=es
    )

    encoder.save(f"transformer_encoder_{sequence}_{embedding}.keras")
    autoencoder.save(f"transformer_autoencoder_{sequence}_{embedding}.keras")


SEQUENCE = sys.argv[1]
EMBEDDING = int(sys.argv[2])

if SEQUENCE == 'TCR':
    df_train = load_data_tcr(f'train_ProtBERT_{SEQUENCE}_1024.h5')
    X_train = np.stack(df_train[SEQUENCE].values)
    del df_train
    
    df_validation = load_data_tcr(f'validation_ProtBERT_{SEQUENCE}_1024.h5')
    X_validation = np.stack(df_validation[SEQUENCE].values)
    del df_validation
    
    df_test = load_data_tcr(f'test_ProtBERT_{SEQUENCE}_1024.h5')
    X_test = np.stack(df_test[SEQUENCE].values)
    del df_test
else:
    df_train = load_data_epitope(f'train_ProtBERT_{SEQUENCE}_1024.h5')
    X_train = np.stack(df_train[SEQUENCE].values)
    del df_train

    df_validation = load_data_epitope(f'validation_ProtBERT_{SEQUENCE}_1024.h5')
    X_validation = np.stack(df_validation[SEQUENCE].values)
    del df_validation

    df_test = load_data_epitope(f'test_ProtBERT_{SEQUENCE}_1024.h5')
    X_test = np.stack(df_test[SEQUENCE].values)
    del df_test

train_best_encoder(SEQUENCE, EMBEDDING)

