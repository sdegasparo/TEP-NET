#!/usr/bin/env python
# coding: utf-8

# # ProtBERT matching

import h5py
from h5py import string_dtype
from ast import literal_eval
import numpy as np
import h5py
from h5py import string_dtype
from ast import literal_eval
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys

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
                # Force decode even if h5py says it's already 'str' dtype
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
                # Force decode even if h5py says it's already 'str' dtype
                if isinstance(data[0], bytes):
                    meta_chunk[key] = [x.decode('utf-8') for x in data]
                else:
                    meta_chunk[key] = data

            # Convert to pandas DataFrame
            df_chunk = pd.DataFrame(meta_chunk)
            df_chunk['epitope'] = list(epitope_chunk)

            yield df_chunk

def load_data_epitope(filename):
    df = pd.DataFrame()
    
    for df_chunk in load_hdf5_epitope_in_chunks(f'{path}{filename}', chunk_size=5000):
        df = pd.concat([df, df_chunk], sort=False)

    return df


def dataframe_to_h5(df, hdf5_path, embedding, chunk_size=5000):   
    # Preprocess the first chunk
    first_chunk = df.iloc[:chunk_size]
    TCR_chunk = np.stack(first_chunk['TCR'])
    epitope_chunk = np.stack(first_chunk['epitope'])
    meta_chunk = first_chunk.drop(columns=['TCR', 'epitope'])

    with h5py.File(hdf5_path, 'w') as f:
        N_max = None
        TCR_ds = f.create_dataset('TCR', shape=(0, 26, embedding), maxshape=(N_max, 26, embedding),
                                  dtype='float32', compression='gzip', chunks=True)
        epitope_ds = f.create_dataset('epitope', shape=(0, 24, embedding), maxshape=(N_max, 24, embedding),
                                      dtype='float32', compression='gzip', chunks=True)
        
        meta_dsets = {}
        for col in meta_chunk.columns:
            values = meta_chunk[col].values
            dtype = string_dtype('utf-8') if pd.api.types.is_object_dtype(values) else values.dtype
            meta_dsets[col] = f.create_dataset(f'meta/{col}', shape=(0,), maxshape=(None,),
                                               dtype=dtype, compression='gzip', chunks=True)

        # Write first chunk
        n_rows = len(TCR_chunk)
        TCR_ds.resize(n_rows, axis=0)
        epitope_ds.resize(n_rows, axis=0)
        TCR_ds[:n_rows] = TCR_chunk
        epitope_ds[:n_rows] = epitope_chunk

        for col in meta_chunk.columns:
            meta_dsets[col].resize(n_rows, axis=0)
            meta_dsets[col][:n_rows] = meta_chunk[col].astype(str).values if pd.api.types.is_object_dtype(meta_chunk[col]) else meta_chunk[col].values

        # Process the rest of the DataFrame
        for start in tqdm(range(chunk_size, len(df), chunk_size), desc="Processing chunks"):
            end = min(start + chunk_size, len(df))
            chunk = df.iloc[start:end]

            TCR_chunk = np.stack(chunk['TCR'])
            epitope_chunk = np.stack(chunk['epitope'])
            meta_chunk = chunk.drop(columns=['TCR', 'epitope'])

            n_new = len(TCR_chunk)
            n_total = TCR_ds.shape[0] + n_new

            TCR_ds.resize(n_total, axis=0)
            epitope_ds.resize(n_total, axis=0)
            TCR_ds[-n_new:] = TCR_chunk
            epitope_ds[-n_new:] = epitope_chunk

            for col in meta_chunk.columns:
                meta_dsets[col].resize(n_total, axis=0)
                meta_dsets[col][-n_new:] = meta_chunk[col].astype(str).values if pd.api.types.is_object_dtype(meta_chunk[col]) else meta_chunk[col].values


EMBEDDING = int(sys.argv[1])
EMBEDDING_TYPE = sys.argv[2]

df_tcr = load_data_tcr(f'TCR_ProtBERT_{EMBEDDING}_{EMBEDDING_TYPE}.h5')
df_epitope = load_data_epitope(f'epitope_ProtBERT_{EMBEDDING}_{EMBEDDING_TYPE}.h5')


# Train
df_train = pd.read_csv(f'{path}train_raw.csv').rename(columns={'TCR': 'TCR_raw', 'epitope': 'epitope_raw'})
df_train = pd.merge(df_train, df_epitope, on='epitope_raw', how='left')
df_train = pd.merge(df_train, df_tcr, on='TCR_raw', how='left')
dataframe_to_h5(df_train, f'train_ProtBERT_{EMBEDDING}_{EMBEDDING_TYPE}.h5', EMBEDDING)
del df_train

# Validation
df_validation = pd.read_csv(f'{path}validation_raw.csv').rename(columns={'TCR': 'TCR_raw', 'epitope': 'epitope_raw'})
df_validation = pd.merge(df_validation, df_epitope, on='epitope_raw', how='left')
df_validation = pd.merge(df_validation, df_tcr, on='TCR_raw', how='left')
dataframe_to_h5(df_validation, f'validation_ProtBERT_{EMBEDDING}_{EMBEDDING_TYPE}.h5', EMBEDDING)
del df_validation

# Test
df_test = pd.read_csv(f'{path}test_raw.csv').rename(columns={'TCR': 'TCR_raw', 'epitope': 'epitope_raw'})
df_test = pd.merge(df_test, df_epitope, on='epitope_raw', how='left')
df_test = pd.merge(df_test, df_tcr, on='TCR_raw', how='left')
dataframe_to_h5(df_test, f'test_ProtBERT_{EMBEDDING}_{EMBEDDING_TYPE}.h5', EMBEDDING)
