#!/usr/bin/env python
# coding: utf-8

# # epiTCR Random Forest 

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, precision_recall_curve, f1_score, precision_score, recall_score, auc
from sklearn.metrics import confusion_matrix
import joblib


path = '/scratch/'

df_train = pd.read_csv(f'{path}train_epiTCR.csv')
df_test = pd.read_csv(f'{path}test_epiTCR.csv')


# ## Preprocess Data

X_train = df_train.drop('binding', axis=1)
y_train = df_train['binding']
X_test = df_test.drop('binding', axis=1)
y_test = df_test['binding']


model = RandomForestClassifier(bootstrap=False, max_features=15, n_estimators=300, n_jobs=4, random_state=42)
model.fit(X_train, y_train)


# ## Test

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

precision, recall, _ = precision_recall_curve(y_test, y_proba)
pr_auc = auc(recall, precision)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_proba))
print("Precision", precision_score(y_test, y_pred))
print("Recall", recall_score(y_test, y_pred))
print("PR-AUC:", pr_auc)
print("F1-Score", f1_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# ## Evaluate TPP
def f1_score_tpp(tp, fp, fn):
    if tp + fp + fn == 0:
        return 0
    return (2 * tp) / ((2 * tp) + fp + fn)


df_tpp1 = pd.read_csv(f'{path}tpp1_epiTCR.csv')
df_tpp2 = pd.read_csv(f'{path}tpp2_epiTCR.csv')
df_tpp3 = pd.read_csv(f'{path}tpp3_epiTCR.csv')
df_tpp4 = pd.read_csv(f'{path}tpp4_epiTCR.csv')

print(f"TPP1: {len(df_tpp1)}, TPP2: {len(df_tpp2)}, TPP3: {len(df_tpp3)}, TPP4: {len(df_tpp4)}")

def predict_tpp(num_tpp, df_tpp):
    X_test = df_tpp.drop('binding', axis=1)
    y_test = df_tpp['binding']
    y_pred_probs = model.predict_proba(X_test)[:, 1]
    
    y_pred = (y_pred_probs > 0.5).astype(int)
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
joblib.dump(model, "epiTCR.joblib")

