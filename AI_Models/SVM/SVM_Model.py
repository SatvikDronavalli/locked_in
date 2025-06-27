"""
SVM training script for IMU features
▪ Saves model   ➜ svm_imu.joblib  (same folder)
▪ Drops fusion files in   AI_Models/_fusion/
   – svm_probs.npy
"""
import numpy as np, joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.inspection import permutation_importance
from IMU_Data.IMU_data import X_train, X_test, y_train, y_test, params

# ── Scale features ─────────────────────────────────────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── Grid-search SVM (probability=True) ─────────────────────────────
param_grid = {'C':[0.1,1,10,100],
              'gamma':['scale',0.01,0.1,1],
              'kernel':['rbf']}
grid = GridSearchCV(
    SVC(probability=True), param_grid,
    cv=5, scoring="f1", n_jobs=-1, verbose=1)
grid.fit(X_train_s, y_train)
svm_model = grid.best_estimator_

# ── Metrics ────────────────────────────────────────────────────────
y_pred = svm_model.predict(X_test_s)
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"F1-score : {f1_score(y_test, y_pred):.4f}")

# Permutation importance plot (unchanged, optional)
'''perm = permutation_importance(svm_model, X_test_s, y_test)
plt.barh(params[perm.importances_mean.argsort()],
         perm.importances_mean[perm.importances_mean.argsort()])
plt.xlabel("Permutation Importance")
plt.show() '''

# ── Save model & scaler ────────────────────────────────────────────
joblib.dump(svm_model, "svm_imu.joblib")
joblib.dump(scaler,    "imu_scaler.joblib")

# ── NEW: write fusion probabilities ────────────────────────────────
FUSION_DIR = Path(__file__).resolve().parent.parent / "_fusion"
FUSION_DIR.mkdir(exist_ok=True)
svm_probs = svm_model.predict_proba(X_test_s)[:, 1]
np.save(FUSION_DIR / "svm_probs.npy", svm_probs)

print("✅ SVM training complete; fusion files saved to", FUSION_DIR)
