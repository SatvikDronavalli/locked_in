from tensorflow.keras.models import load_model
from Force_Data import gait_x_test, gait_y_test
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Load the best model after training
best_model = load_model('best_fall_risk_model.keras')

# Evaluate the best model on the testing data
y_pred_probs = best_model.predict(gait_x_test)
y_true = np.argmax(y_pred_probs, axis=1)
accuracy = accuracy_score(y_true, gait_y_test)
precision = precision_score(y_true, gait_y_test, average="weighted")  # Change to 'macro' or 'micro' if needed
recall = recall_score(y_true, gait_y_test, average="weighted")
f1 = f1_score(y_true, gait_y_test, average="weighted")

# Print results
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
