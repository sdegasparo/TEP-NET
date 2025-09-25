#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import peptides
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import numpy as np
import torch
from transformers import BertModel, BertTokenizer
from joblib import load
import h5py
from h5py import string_dtype
from ast import literal_eval
from tqdm import tqdm
import sys

path = '/scratch/'


INDEX = sys.argv[1]
SEQUENCE = sys.argv[2]
FILE = f'{SEQUENCE}_raw_{INDEX}.csv'

if SEQUENCE == 'TCR':
    MAX_LEN = 26
else:
    MAX_LEN = 24


df = pd.read_csv(f'{path}{FILE}')


standard_scaler = load(f'{path}std_scaler_{SEQUENCE}.bin')
minmax_scaler = load(f'{path}minmax_scaler_{SEQUENCE}.bin')


# BioPython
def calculate_physicochemical_properties(sequence):
    analysis = ProteinAnalysis(sequence)
    properties = {
        'hydrophobicity': analysis.gravy(),
        'aromaticity': analysis.aromaticity(),
        'isoelectric_point': analysis.isoelectric_point(),
        'instability_index': analysis.instability_index()
    }
    return properties

# Peptides
def calculate_peptide_descriptors(sequence):
    peptide_descriptors = peptides.Peptide(sequence).descriptors()
    meaningful_peptide_descriptors = {
        'KF7': peptide_descriptors.get("KF7"),
        'KF1': peptide_descriptors.get("KF1")
    }
    return meaningful_peptide_descriptors


# Load ProtBERT model
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
model = BertModel.from_pretrained("Rostlab/prot_bert")

# Prepare embeddings for all sequences
def extract_embeddings(sequence, max_len):
    sequence_prepared = " ".join(sequence)
    encoded_input = tokenizer(sequence_prepared, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    embedding = model_output.last_hidden_state.squeeze(0).numpy()  # shape: (seq_len, 1024)

    seq_len = embedding.shape[0]
    if seq_len < max_len:
        padding = np.zeros((max_len - seq_len, embedding.shape[1]))
        embedding_padded = np.vstack([embedding, padding])
    else:
        embedding_padded = embedding[:max_len]

    return embedding_padded.tolist()


# Apply Peptides
descriptors_df = df[f'{SEQUENCE}_raw'].apply(calculate_peptide_descriptors).apply(pd.Series)
descriptors_df = descriptors_df.add_prefix(f'{SEQUENCE}_')
df = pd.concat([df, descriptors_df], axis=1)

# Apply BioPython
properties_df = df[f'{SEQUENCE}_raw'].apply(calculate_physicochemical_properties).apply(pd.Series)
properties_df = properties_df.add_prefix(f'{SEQUENCE}_')
df = pd.concat([df, properties_df], axis=1)


# Feature scaling
if SEQUENCE == 'TCR':
    gaussian_features = ['TCR_KF7', 'TCR_KF1']
    not_gaussian_features = ['TCR_hydrophobicity', 'TCR_aromaticity', 'TCR_isoelectric_point', 'TCR_instability_index']
    df[gaussian_features] = standard_scaler.transform(df[gaussian_features])
    df[not_gaussian_features] = minmax_scaler.transform(df[not_gaussian_features])
else:
    not_gaussian_features = ['epitope_KF7', 'epitope_KF1', 'epitope_hydrophobicity', 'epitope_aromaticity', 'epitope_isoelectric_point', 'epitope_instability_index']
    df[not_gaussian_features] = minmax_scaler.transform(df[not_gaussian_features])

# ProtBERT embedding
df[SEQUENCE] = df[f'{SEQUENCE}_raw'].apply(extract_embeddings, max_len=MAX_LEN)

# # Convert to h5

def save_tcr_to_h5(df, hdf5_path, chunk_size=5000):
    first_chunk = df.iloc[:chunk_size]
    TCR_chunk = np.stack(first_chunk['TCR'])
    meta_chunk = first_chunk.drop(columns=['TCR'])

    with h5py.File(hdf5_path, 'w') as f:
        TCR_ds = f.create_dataset('TCR', shape=(0, 26, 1024), maxshape=(None, 26, 1024),
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


def save_epitope_to_h5(df, hdf5_path, chunk_size=5000):
    first_chunk = df.iloc[:chunk_size]
    epitope_chunk = np.stack(first_chunk['epitope'])
    meta_chunk = first_chunk.drop(columns=['epitope'])

    with h5py.File(hdf5_path, 'w') as f:
        epitope_ds = f.create_dataset('epitope', shape=(0, 24, 1024), maxshape=(None, 24, 1024),
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


if SEQUENCE == 'TCR':
    save_tcr_to_h5(df, f'{path}{SEQUENCE}_ProtBERT_1024_{INDEX}.h5')
else:
    save_epitope_to_h5(df, f'{path}{SEQUENCE}_ProtBERT_1024_{INDEX}.h5')

