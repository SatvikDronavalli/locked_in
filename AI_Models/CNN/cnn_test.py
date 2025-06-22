from tensorflow.keras.models import load_model
from Force_Data.force_data import final_x_test,final_y_test
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the best model after training
best_model = load_model('best_fall_risk_model.keras')

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
  
