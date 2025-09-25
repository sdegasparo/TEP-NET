#!/usr/bin/env python
# coding: utf-8

import h5py
from h5py import string_dtype
import pandas as pd
import numpy as np
from joblib import load
from tensorflow.keras.models import load_model
import keras
import sys
from tqdm import tqdm

path = '/scratch/'

def load_hdf5_TCR_in_chunks(h5_path, chunk_size=5000):
    with h5py.File(h5_path, 'r') as f:
        total_samples = f['TCR'].shape[0]
        meta_keys = list(f['meta'].keys())

        for i in range(0, total_samples, chunk_size):
            end = min(i + chunk_size, total_samples)

            # Load chunk of embeddings
            TCR_chunk = f['TCR'][i:end]

            # Load metadata chunk
            meta_chunk = {}
            for key in meta_keys:
                data = f[f'meta/{key}'][i:end]
                if isinstance(data[0], bytes):
                    meta_chunk[key] = [x.decode('utf-8') for x in data]
                else:
                    meta_chunk[key] = data

            # Convert to pandas DataFrame
            df_chunk = pd.DataFrame(meta_chunk)
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
        meta_keys = list(f['meta'].keys())

        for i in range(0, total_samples, chunk_size):
            end = min(i + chunk_size, total_samples)

            # Load chunk of embeddings
            epitope_chunk = f['epitope'][i:end]

            # Load metadata chunk
            meta_chunk = {}
            for key in meta_keys:
                data = f[f'meta/{key}'][i:end]
                if isinstance(data[0], bytes):
                    meta_chunk[key] = [x.decode('utf-8') for x in data]
                else:
                    meta_chunk[key] = data

            df_chunk = pd.DataFrame(meta_chunk)
            df_chunk['epitope'] = list(epitope_chunk)

            yield df_chunk

def load_data_epitope(filename):
    df = pd.DataFrame()
    
    for df_chunk in load_hdf5_epitope_in_chunks(f'{path}{filename}', chunk_size=5000):
        df = pd.concat([df, df_chunk], sort=False)

    return df

def save_tcr_to_h5(df, hdf5_path, embedding, chunk_size=5000):
    first_chunk = df.iloc[:chunk_size]
    TCR_chunk = np.stack(first_chunk['TCR'])
    meta_chunk = first_chunk.drop(columns=['TCR'])

    with h5py.File(hdf5_path, 'w') as f:
        TCR_ds = f.create_dataset('TCR', shape=(0, 26, embedding), maxshape=(None, 26, embedding),
                                  dtype='float32', compression='gzip', chunks=True)

        meta_dsets = {}
        for col in meta_chunk.columns:
            dtype = string_dtype('utf-8') if pd.api.types.is_object_dtype(meta_chunk[col]) else meta_chunk[col].dtype
            meta_dsets[col] = f.create_dataset(f'meta/{col}', shape=(0,), maxshape=(None,), dtype=dtype,
                                               compression='gzip', chunks=True)

        n_rows = len(TCR_chunk)
        TCR_ds.resize(n_rows, axis=0)
        TCR_ds[:n_rows] = TCR_chunk

        for col in meta_chunk.columns:
            meta_dsets[col].resize(n_rows, axis=0)
            meta_dsets[col][:n_rows] = meta_chunk[col].astype(str).values if pd.api.types.is_object_dtype(meta_chunk[col]) else meta_chunk[col].values

        for start in tqdm(range(chunk_size, len(df), chunk_size), desc="Processing TCR chunks"):
            end = min(start + chunk_size, len(df))
            chunk = df.iloc[start:end]
            TCR_chunk = np.stack(chunk['TCR'])
            meta_chunk = chunk.drop(columns=['TCR'])

            n_new = len(TCR_chunk)
            n_total = TCR_ds.shape[0] + n_new

            TCR_ds.resize(n_total, axis=0)
            TCR_ds[-n_new:] = TCR_chunk

            for col in meta_chunk.columns:
                meta_dsets[col].resize(n_total, axis=0)
                meta_dsets[col][-n_new:] = meta_chunk[col].astype(str).values if pd.api.types.is_object_dtype(meta_chunk[col]) else meta_chunk[col].values

def save_epitope_to_h5(df, hdf5_path, embedding, chunk_size=5000):
    first_chunk = df.iloc[:chunk_size]
    epitope_chunk = np.stack(first_chunk['epitope'])
    meta_chunk = first_chunk.drop(columns=['epitope'])

    with h5py.File(hdf5_path, 'w') as f:
        epitope_ds = f.create_dataset('epitope', shape=(0, 24, embedding), maxshape=(None, 24, embedding),
                                      dtype='float32', compression='gzip', chunks=True)

        meta_dsets = {}
        for col in meta_chunk.columns:
            dtype = string_dtype('utf-8') if pd.api.types.is_object_dtype(meta_chunk[col]) else meta_chunk[col].dtype
            meta_dsets[col] = f.create_dataset(f'meta/{col}', shape=(0,), maxshape=(None,), dtype=dtype,
                                               compression='gzip', chunks=True)

        n_rows = len(epitope_chunk)
        epitope_ds.resize(n_rows, axis=0)
        epitope_ds[:n_rows] = epitope_chunk

        for col in meta_chunk.columns:
            meta_dsets[col].resize(n_rows, axis=0)
            meta_dsets[col][:n_rows] = meta_chunk[col].astype(str).values if pd.api.types.is_object_dtype(meta_chunk[col]) else meta_chunk[col].values

        for start in tqdm(range(chunk_size, len(df), chunk_size), desc="Processing epitope chunks"):
            end = min(start + chunk_size, len(df))
            chunk = df.iloc[start:end]
            epitope_chunk = np.stack(chunk['epitope'])
            meta_chunk = chunk.drop(columns=['epitope'])

            n_new = len(epitope_chunk)
            n_total = epitope_ds.shape[0] + n_new

            epitope_ds.resize(n_total, axis=0)
            epitope_ds[-n_new:] = epitope_chunk

            for col in meta_chunk.columns:
                meta_dsets[col].resize(n_total, axis=0)
                meta_dsets[col][-n_new:] = meta_chunk[col].astype(str).values if pd.api.types.is_object_dtype(meta_chunk[col]) else meta_chunk[col].values

def reduce_embedding(embedding, pca):
    return pca.transform(embedding)

def reduce_encoder(embedding, encoder):
    embedding = np.array(embedding)
    if embedding.ndim == 2:
        embedding = np.expand_dims(embedding, axis=0)
    reduced = encoder.predict(embedding, verbose=0)
    return reduced[0]


EMBEDDING = int(sys.argv[1])
SEQUENCE = sys.argv[2]
MODEL = sys.argv[3]


if SEQUENCE == 'TCR':
    df = load_data_tcr(f'{SEQUENCE}_ProtBERT_1024.h5')
else:
    df = load_data_epitope(f'{SEQUENCE}_ProtBERT_1024.h5')

if MODEL == 'pca':
    pca = load(f'{path}{MODEL}_{SEQUENCE}_{EMBEDDING}.bin')
    df[SEQUENCE] = df[SEQUENCE].apply(lambda e: reduce_embedding(e, pca))
else:
    encoder =  keras.saving.load_model(f'{path}{MODEL}_{SEQUENCE}_{EMBEDDING}.keras')
    df[SEQUENCE] = df[SEQUENCE].apply(lambda e: reduce_encoder(e, encoder))
    

if SEQUENCE == 'TCR':
    save_tcr_to_h5(df, f'{path}{SEQUENCE}_ProtBERT_{EMBEDDING}_{MODEL}.h5', EMBEDDING)
else:
    save_epitope_to_h5(df, f'{path}{SEQUENCE}_ProtBERT_{EMBEDDING}_{MODEL}.h5', EMBEDDING)

