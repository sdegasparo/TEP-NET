#!/usr/bin/env python
# coding: utf-8

# # Feature Enrichment epiTCR

import pandas as pd
import numpy as np
import sys

path = ''

df_train = pd.read_csv(f"{path}train_raw.csv")
df_validation = pd.read_csv(f"{path}validation_raw.csv")
df_test = pd.read_csv(f"{path}test_raw.csv")


blosum62_20aa = {
    'A': np.array((4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0)),
    'R': np.array((-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3)),
    'N': np.array((-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3)),
    'D': np.array((-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3)),
    'C': np.array((0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1)),
    'Q': np.array((-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2)),
    'E': np.array((-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2)),
    'G': np.array((0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3)),
    'H': np.array((-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3)),
    'I': np.array((-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3)),
    'L': np.array((-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1)),
    'K': np.array((-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2)),
    'M': np.array((-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1)),
    'F': np.array((-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1)),
    'P': np.array((-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2)),
    'S': np.array((1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2)),
    'T': np.array((0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0)),
    'W': np.array((-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3)),
    'Y': np.array((-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1)),
    'V': np.array((0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4))
}

def enccodeListBlosumMaxLen(aa_seqs, blosum, max_seq_len):
    sequences = []
    for seq in aa_seqs:
        seq = seq.upper()
        e_seq = np.zeros((len(seq), len(blosum["A"])))
        count = 0
        for aa in seq:
            if aa in blosum:
                e_seq[count] = blosum[aa]
                count += 1
            else:
                sys.stderr.write(f"Unknown amino acid in peptides: {aa}, encoding aborted!\n")
                sys.exit(2)
                
        sequences.append(e_seq)
    n_seqs = len(aa_seqs)
    n_features = sequences[0].shape[1]

    enc_aa_seq = np.zeros((n_seqs, max_seq_len, n_features))
    for i in range(n_seqs):
        enc_aa_seq[i, :sequences[i].shape[0], :n_features] = sequences[i]

    return enc_aa_seq


def dataRepresentationBlosum62(df):
    encoding = blosum62_20aa
    TCR_len = 26
    epitope_len = 24

    m_tcr = enccodeListBlosumMaxLen(df.TCR, encoding, TCR_len)
    m_tcr = m_tcr.reshape(len(m_tcr), TCR_len*20)

    m_epitope = enccodeListBlosumMaxLen(df.epitope, encoding, epitope_len)
    m_epitope = m_epitope.reshape(len(m_epitope), epitope_len*20)


    df_res1 = pd.DataFrame(m_tcr)
    df_res2 = pd.DataFrame(m_epitope)

    res = pd.concat([df_res1, df_res2, df.TCR, df.epitope], axis=1)
    res.columns = ["F" + str(i + 1) for i in range(res.shape[1])]

    label = df['binding']
    res = pd.concat([res, label], axis=1)
    res = res.rename(columns={"F1001": "TCR_raw", "F1002": "epitope_raw"})

    return res


df_train = dataRepresentationBlosum62(df_train)
df_train.to_csv(f"{path}train_epiTCR.csv", index=False)


df_test = dataRepresentationBlosum62(df_test)
df_test.to_csv(f"{path}test_epiTCR.csv", index=False)


df_validation = dataRepresentationBlosum62(df_validation)
df_validation.to_csv(f"{path}validation_epiTCR.csv", index=False)

df_tpp1 = pd.read_csv(f'{path}tpp1_raw.csv')
df_tpp2 = pd.read_csv(f'{path}tpp2_raw.csv')
df_tpp3 = pd.read_csv(f'{path}tpp3_raw.csv')
df_tpp4 = pd.read_csv(f'{path}tpp4_raw.csv')

df_tpp1 = dataRepresentationBlosum62(df_tpp1)
df_tpp1.to_csv(f"{path}tpp1_epiTCR.csv", index=False)

df_tpp2 = dataRepresentationBlosum62(df_tpp2)
df_tpp2.to_csv(f"{path}tpp2_epiTCR.csv", index=False)

df_tpp3 = dataRepresentationBlosum62(df_tpp3)
df_tpp3.to_csv(f"{path}tpp3_epiTCR.csv", index=False)

df_tpp4 = dataRepresentationBlosum62(df_tpp4)
df_tpp4.to_csv(f"{path}tpp4_epiTCR.csv", index=False)