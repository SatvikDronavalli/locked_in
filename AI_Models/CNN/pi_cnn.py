from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
best_model = load_model("best_fall_risk_model.keras")
pi_force_1 = pd.read_csv("collected_force_left.csv")
pi_force_2 = pd.read_csv("collected_force_right.csv")
pi_test = np.stack([pi_force_1,pi_force_2],-1)
preds = best_model.predict(pi_test)
for i in range(1,11):
    print(f"Trial {i} yielded a fall risk of {round(preds[i-1][0]*100,2)}%")
