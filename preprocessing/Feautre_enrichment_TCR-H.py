#!/usr/bin/env python
# coding: utf-8

# # Feature enrichment TCR-H

import pandas as pd
import peptides
path = ''

df_tcr_h = pd.read_csv(f"{path}sample_train_data_TCR-H.csv")


peptide = peptides.Peptide("CASSARSGELFF")
columns = df_tcr_h.columns
dict_keys = set(peptide.descriptors().keys())

processed_columns = set()
for col in columns:
    if col.startswith("epitope_"):
        processed_columns.add(col.replace("epitope_", ""))
    elif col.startswith("cdr3_"):
        processed_columns.add(col.replace("cdr3_", ""))
        
used_keys = dict_keys & processed_columns
unused_keys = dict_keys - used_keys
not_peptides_descriptors = processed_columns - dict_keys


def calculate_peptide_descriptors(sequence):
    peptide = peptides.Peptide(sequence)

    descriptors = {}
    descriptors['aliphatic_index'] = peptide.aliphatic_index()
    descriptors['boman'] = peptide.boman()
    descriptors['charge'] = peptide.charge(pH=7.4)
    descriptors['hydrophobic_moment'] = peptide.hydrophobic_moment()
    descriptors['mz'] = peptide.mz()
    descriptors['molecular'] = peptide.molecular_weight()
    descriptors['hydrophobicity'] = peptide.hydrophobicity()
    descriptors['isoelectric_point'] = peptide.isoelectric_point()
    descriptors['instability_index'] = peptide.instability_index()

    peptide_descriptors = peptide.descriptors()
    for key in used_keys:
        descriptors[key] = peptide_descriptors.get(key)
    return descriptors

def enrich_df(df):
    tcr_descriptors_df = df['TCR'].apply(calculate_peptide_descriptors).apply(pd.Series)
    tcr_descriptors_df = tcr_descriptors_df.add_prefix('TCR_')
    epitope_descriptors_df = df['epitope'].apply(calculate_peptide_descriptors).apply(pd.Series)
    epitope_descriptors_df = epitope_descriptors_df.add_prefix('epitope_')
    
    return pd.concat([df, tcr_descriptors_df, epitope_descriptors_df], axis=1)


df_train = pd.read_csv(f"{path}train_raw.csv")
df_test = pd.read_csv(f"{path}test_raw.csv")
df_validation = pd.read_csv(f"{path}validation_raw.csv")

df_train = enrich_df(df_train)
df_train.to_csv(f"{path}train_tcr_h.csv", index=False)

df_test = enrich_df(df_test)
df_test.to_csv(f"{path}test_tcr_h.csv", index=False)

df_validation = enrich_df(df_validation)
df_validation.to_csv(f"{path}validation_tcr_h.csv", index=False)

df_tpp1 = pd.read_csv(f'{path}tpp1_raw.csv')
df_tpp2 = pd.read_csv(f'{path}tpp2_raw.csv')
df_tpp3 = pd.read_csv(f'{path}tpp3_raw.csv')
df_tpp4 = pd.read_csv(f'{path}tpp4_raw.csv')


df_tpp1 = enrich_df(df_tpp1)
df_tpp1.to_csv(f"{path}tpp1_tcr-h.csv", index=False)

df_tpp2 = enrich_df(df_tpp2)
df_tpp2.to_csv(f"{path}tpp2_tcr-h.csv", index=False)

df_tpp3 = enrich_df(df_tpp3)
df_tpp3.to_csv(f"{path}tpp3_tcr-h.csv", index=False)

df_tpp4 = enrich_df(df_tpp4)
df_tpp4.to_csv(f"{path}tpp4_tcr-h.csv", index=False)
