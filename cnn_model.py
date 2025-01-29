from Force_Data import GRF_right_train, GRF_left_train, GRF_right_test, GRF_left_test
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model

# Combine GRF training and testing data
X_train_combined = np.stack([GRF_right_train, GRF_left_train], axis=-1)
X_test_combined = np.stack([GRF_right_test, GRF_left_test], axis=-1)

# Define the model for GRF data only (no y_train or y_test involved)
model = Sequential([
    Conv1D(filters=32, kernel_size=3, activation='relu', 
           input_shape=(X_train_combined.shape[1], X_train_combined.shape[2])),
    Dropout(0.2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Assuming a binary classification task
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=['accuracy'])

# Define checkpoints and early stopping (no validation data)
checkpoint = ModelCheckpoint('best_fall_risk_model.h5', save_best_only=True, monitor='loss', mode='min')
early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

# Remove validation_split (since you're not using validation data right now)
print(X_train_combined.shape)  # Should be (num_samples, timesteps, 2)
print(type(X_train_combined))  # Should be <class 'numpy.ndarray'>
print(np.isnan(X_train_combined).any())  # Should be False
print(np.isinf(X_train_combined).any())  # Should be False

# Train the model with just the GRF data
history = model.fit(
    X_train_combined, 
    epochs=25,
    batch_size=32,
    callbacks=[checkpoint, early_stopping]
)

# Save the training history to a CSV file
pd.DataFrame(history.history).to_csv('training_history.csv', index=False)

# Load the best model after training
best_model = load_model('best_fall_risk_model.h5')

# Evaluate the best model on the testing data (no y_test required)
test_loss = best_model.evaluate(X_test_combined)  # No y_test here
print(f'Test Loss: {test_loss}')
