import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import resample

# Load the dataset
file_name = "HC101_SelfPace.csv"
patient = file_name.split("_")[0]
df = pd.read_csv(file_name)
metadata = pd.read_csv("CONTROLS - Demographic+Clinical - datasetV1.csv")
# print(metadata[metadata.columns[8]])
weight = int(metadata.loc[metadata[metadata.columns[0]] == patient][metadata.columns[8]].tolist()[0])*9.81
# Remove 'Standing' periods
df = df[df["GeneralEvent"] != "Standing"].reset_index(drop=True)

# Identify all sections where the subject is walking straight (i.e., 'Walk' events only)
walk_df = df[df["GeneralEvent"] == "Walk"].reset_index(drop=True)

# Parameters for step detection
threshold = 50
min_length = 30
max_length = 120

cases = []
extracted_curves = []
r_acc_x = []
r_acc_y = []
r_acc_z = []
r_gyr_x = []
r_gyr_y = []
r_gyr_z = []
r_cop_x = []
r_cop_y = []
def normalize_to_101_points(curve):
    """
    Interpolates a 1D array (e.g., a GRF curve) to have exactly 101 points.
    """
    original_length = len(curve)
    original_x = np.linspace(0, 1, original_length)
    target_x = np.linspace(0, 1, 101)

    f = interp1d(original_x, curve, kind='linear')  # or 'cubic' for smoother interpolation
    normalized_curve = f(target_x)

    return normalized_curve

while True:
    stepStarted = False
    started_idx = 0
    ended_idx = 0
    found_step = False

    for i in range(0, len(walk_df["RTotalForce"]) - 1):
        # Checks if a step started
        if not stepStarted and pd.notna(walk_df.iloc[i]["RTotalForce"]) and pd.notna(walk_df.iloc[i+1]["RTotalForce"]) and \
           walk_df.iloc[i]["RTotalForce"] < threshold and walk_df.iloc[i+1]["RTotalForce"] >= threshold:
            stepStarted = True
            started_idx = i
            ended_idx = i
            found_step = True
        # Checks if a step ended
        elif stepStarted and pd.notna(walk_df.iloc[i]["RTotalForce"]) and pd.notna(walk_df.iloc[i+1]["RTotalForce"]) and \
             walk_df.iloc[i]["RTotalForce"] < threshold and walk_df.iloc[i+1]["RTotalForce"] < threshold:
            break
        elif stepStarted:
            ended_idx += 1

    if not found_step or ended_idx <= started_idx or ended_idx + 2 > len(walk_df):
        break

    new_df = walk_df.iloc[started_idx:ended_idx+2].reset_index(drop=True)
    r_acc_x.append(normalize_to_101_points(new_df["Rinsole:Acc_X"]))
    r_acc_y.append(normalize_to_101_points(new_df["Rinsole:Acc_X"]))
    r_acc_z.append(normalize_to_101_points(new_df["Rinsole:Acc_X"]))
    r_gyr_x.append(normalize_to_101_points(new_df["Rinsole:Acc_X"]))
    r_gyr_y.append(normalize_to_101_points(new_df["Rinsole:Acc_X"]))
    r_gyr_z.append(normalize_to_101_points(new_df["Rinsole:Acc_X"]))
    r_cop_x.append(normalize_to_101_points(new_df["Rinsole:Acc_X"]))
    r_cop_x.append(normalize_to_101_points(new_df["Rinsole:Acc_X"]))
    if not new_df["RTotalForce"].isna().any() and max_length >= len(new_df) >= min_length:
        cases.append((started_idx, ended_idx+2))
        extracted_curves.append(new_df["RTotalForce"].values / weight)
    walk_df = walk_df.iloc[ended_idx+2:].reset_index(drop=True)
    if len(walk_df) <= 100:
        break


normalized_curves = [normalize_to_101_points(c) for c in extracted_curves]

if extracted_curves[-1][-1] > threshold:
    extracted_curves.pop()
for i in r_acc_x:
    plt.plot(i)
    plt.show()

print(f"Extracted {len(extracted_curves)} valid walking force curves.")
''