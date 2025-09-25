#!/usr/bin/env python
# coding: utf-8

# # PCA Fitting

import pandas as pd
import numpy as np
from sklearn.decomposition import IncrementalPCA
from joblib import dump
import sys

path = '/scratch/'

def fit_incremental_pca(sequence, dim, batch_size=1000):
    pca = IncrementalPCA(n_components=int(dim))
    file_path = f'{path}{sequence}_ProtBERT.csv'
    reader = pd.read_csv(file_path, chunksize=batch_size)
    for chunk in reader:
        chunk[sequence] = chunk[sequence].apply(eval).apply(np.array)
        batch = np.concatenate(chunk[sequence].tolist(), axis=0)
        pca.partial_fit(batch)

    dump(pca, f'{path}pca_{sequence}_{dim}.bin', compress=True)
    dump(pca, f'pca_{sequence}_{dim}_direct.bin', compress=True)

fit_incremental_pca(sys.argv[1], sys.argv[2])