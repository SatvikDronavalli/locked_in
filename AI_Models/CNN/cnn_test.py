from tensorflow.keras.models import load_model
from Force_Data.force_data import final_x_test,final_y_test
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path

# Load the best model after training
best_model = load_model('best_fall_risk_cnn_lstm.keras')

# Evaluate the best model on the testing data
y_pred_probs = best_model.predict(final_x_test)  # Get probabilities (output of sigmoid layer)
y_pred = (y_pred_probs > 0.5).astype(int)  # Convert probabilities to binary (0 or 1) using a threshold of 0.5
y_true = final_y_test  # Ground truth labels

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# Print results
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# ── NEW: write fusion artifacts ────────────────────────────────────
FUSION_DIR = Path(__file__).resolve().parent.parent / "_fusion"
FUSION_DIR.mkdir(exist_ok=True)

# per-sample probability of class 1 on SAME test set used by SVM
cnn_probs = best_model.predict(final_x_test, verbose=0).ravel()
np.save(FUSION_DIR / "cnn_probs.npy", cnn_probs)

# save labels once (over-write OK) so stack_meta.py can read them
np.save(FUSION_DIR / "y_test.npy", final_y_test)

print("✅ CNN-LSTM training complete; fusion files saved to", FUSION_DIR)