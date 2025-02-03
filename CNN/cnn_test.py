from tensorflow.keras.models import load_model
from force_data import gait_x_test, gait_y_test
# Load the best model after training
best_model = load_model('best_fall_risk_model.keras')

# Evaluate the best model on the testing data
test_loss, test_accuracy = best_model.evaluate(gait_x_test, gait_y_test)  # Pass y_test here
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')
