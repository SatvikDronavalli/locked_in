#!/usr/bin/env python
"""
Stacking meta-classifier: logistic regression on two confidence scores.
Run after CNN_LSTM.py and SVM_Model.py have produced fusion artifacts.
"""
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

FUSION = Path("AI_Models") / "_fusion"
cnn_probs = np.load(FUSION / "cnn_probs.npy")
svm_probs = np.load(FUSION / "svm_probs.npy")
y_test    = np.load(FUSION / "y_test.npy")

meta_X = np.column_stack([cnn_probs[:24], svm_probs])
print(len(y_test))
meta_clf = LogisticRegression().fit(meta_X, y_test)

meta_probs = meta_clf.predict_proba(meta_X)[:, 1]
meta_pred  = (meta_probs >= 0.5).astype(int)

print("\n=== STACKED META-MODEL ===")
print(f"Accuracy : {accuracy_score(y_test, meta_pred):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, meta_probs):.4f}")
print(f"F1-Score : {f1_score(y_test, meta_pred):.4f}")
print("Weights (GRF, IMU):", meta_clf.coef_.round(3))