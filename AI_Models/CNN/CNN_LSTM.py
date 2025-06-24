"""
CNN-LSTM training script for GRF data
▪ Saves model   ➜ best_fall_risk_cnn_lstm.keras  (same folder)
▪ Drops fusion files in   AI_Models/_fusion/
   – cnn_probs.npy   (P(fall=1) for test set)
   – y_test.npy      (labels, saved once)
"""

from pathlib import Path
import numpy as np, pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, Dropout, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from Force_Data.force_data import (
    final_x_train, final_y_train,
    final_x_test,  final_y_test
)

# ── Build model (unchanged) ─────────────────────────────────────────
model = Sequential([
    Input(shape=(101, 6)),
    Conv1D(32, 3, activation="relu"),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(64, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer=Adam(1e-3),
              loss="binary_crossentropy",
              metrics=["accuracy"])

ckpt = ModelCheckpoint(
    "best_fall_risk_cnn_lstm.keras",
    save_best_only=True, monitor="loss", mode="min")
es   = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)

history = model.fit(
    final_x_train, final_y_train,
    epochs=300, batch_size=32,
    callbacks=[ckpt, es], verbose=2
)

pd.DataFrame(history.history).to_csv("training_history_cnn_lstm.csv",
                                     index=False)

# ── NEW: write fusion artifacts ────────────────────────────────────
FUSION_DIR = Path(__file__).resolve().parent.parent / "_fusion"
FUSION_DIR.mkdir(exist_ok=True)

# per-sample probability of class 1 on SAME test set used by SVM
cnn_probs = model.predict(final_x_test, verbose=0).ravel()
np.save(FUSION_DIR / "cnn_probs.npy", cnn_probs)

# save labels once (over-write OK) so stack_meta.py can read them
np.save(FUSION_DIR / "y_test.npy", final_y_test)

print("✅ CNN-LSTM training complete; fusion files saved to", FUSION_DIR)
