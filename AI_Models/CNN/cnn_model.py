from Force_Data.force_data import final_x_train,final_y_train
import pandas as pd
# GPU path: C:\Users\Satvik\AppData\Local\Temp\CUDA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv1D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model
# Define the model for GRF data
model = Sequential([
    Input(shape=(101,6)),
    Conv1D(filters=32, kernel_size=3, activation='relu'),
    Dropout(0.2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification output
])
# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=['accuracy'])

# Define checkpoints and early stopping
checkpoint = ModelCheckpoint('best_fall_risk_model.keras', save_best_only=True, monitor='loss', mode='min')
early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
print("hello world")

# Train the model with GRF data and labels
history = model.fit(
    final_x_train,
    final_y_train,  # Use actual labels here
    epochs=200,
    batch_size=32,
    callbacks=[checkpoint, early_stopping]
)

# Save the training history to a CSV file
pd.DataFrame(history.history).to_csv('training_history.csv', index=False)
