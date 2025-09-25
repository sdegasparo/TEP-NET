#!/usr/bin/env python
# coding: utf-8

# # TCR-H Model

import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score, make_scorer, roc_curve, auc, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.metrics import confusion_matrix
import joblib
from cuml.svm import SVC as cuSVC


path = '/scratch/'

df_train = pd.read_csv(f'{path}train_tcr-h.csv')
df_test = pd.read_csv(f'{path}test_tcr-h.csv')


# ## Preprocess Data

var_columns = [c for c in df_train.columns if c not in('TCR', 'epitope', 'binding')]
# Nylonase,Class
X_train = df_train.loc[:, var_columns]
y_train = df_train.loc[:, 'binding']

X_test = df_test.loc[:, var_columns]
y_test = df_test.loc[:, 'binding']

correlation_matrix = X_train.corr()

def correlation(dataset, threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr

corr_features = correlation(X_train, 0.8)
print(len(set(corr_features)))

corr_features = list(corr_features)

print("Removing correlated features ")
X_train = X_train.drop(corr_features,axis=1)
X_test = X_test.drop(corr_features,axis=1)

# Split the data into feature and target variables
y_train = df_train['binding']
y_test = df_test['binding']


model = cuSVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced')
model.fit(X_train, y_train)


# ## Test

y_pred = model.predict(X_test)


print("ROC-AUC", roc_auc_score(y_test, y_pred))
print("Accuracy", accuracy_score(y_test, y_pred))
print("Precision", precision_score(y_test, y_pred))
print("Recall", recall_score(y_test, y_pred))
print("F1-Score", f1_score(y_test, y_pred))


# ## Evaluate TPP
def f1_score_tpp(tp, fp, fn):
    if tp + fp + fn == 0:
        return 0
    return (2 * tp) / ((2 * tp) + fp + fn)


df_tpp1 = pd.read_csv(f'{path}tpp1_tcr-h.csv')
df_tpp2 = pd.read_csv(f'{path}tpp2_tcr-h.csv')
df_tpp3 = pd.read_csv(f'{path}tpp3_tcr-h.csv')
df_tpp4 = pd.read_csv(f'{path}tpp4_tcr-h.csv')

# Print category sizes
print(f"TPP1: {len(df_tpp1)}, TPP2: {len(df_tpp2)}, TPP3: {len(df_tpp3)}, TPP4: {len(df_tpp4)}")

def predict_tpp(num_tpp, df_tpp):
    X_test = df_tpp.loc[:, var_columns]
    X_test = X_test.drop(corr_features,axis=1)
    y_test = df_tpp['binding']
    y_pred = model.predict(X_test)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(f"{num_tpp}: True Positives (TP): {tp}, True Negatives (TN): {tn}, False Positives (FP): {fp}, False Negatives (FN): {fn}")

    return f1_score_tpp(tp, fp, fn)

tpp1 = predict_tpp("TPP1", df_tpp1)
tpp2 = predict_tpp("TPP2", df_tpp2)
tpp3 = predict_tpp("TPP3", df_tpp3)
tpp4 = predict_tpp("TPP4", df_tpp4)

dictionary = {'epiTCR': [tpp1, tpp2, tpp3, tpp4]}
print(dictionary)

# Save model
joblib.dump(model, "TCR-H.joblib")

