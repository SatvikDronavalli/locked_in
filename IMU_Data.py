# Setup
import pandas as pd
from pathlib import Path
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
pd.set_option('display.max_colwidth',None)
pd.set_option('display.max_rows',None)
pd.set_option('display.width',None)
pd.set_option('display.max_columns',None)
gait_file = Path("IMU_Data") / "05_gait_parameters.xlsx"
demographics_file = Path("IMU_Data") / "01_demography_sppb_mmse.xlsx"
gait_params = pd.read_excel(gait_file)
demographics = pd.read_excel(demographics_file)
demographics.rename(columns={"Number of self-reported falls in the last month": "at risk of falls"}, inplace=True)
# Setting excess negative results equal to None
index = 0
rows_removed = 0
while rows_removed < 20:
    if demographics.loc[index]["at risk of falls"] == 0:
        demographics.loc[index,"at risk of falls"] = None
        rows_removed += 1
    index += 1
demographics.drop(44,inplace=True)
demographics.drop(45,inplace=True)
# Filling in empty name values
curr_name = None
for i in range(0, len(gait_params)):
    if type(gait_params.loc[i,"Name"]) != type(float()):
        curr_name = gait_params.loc[i,"Name"]
    else:
        gait_params.loc[i,"Name"] = curr_name
dem_index = 0 
# Marking non-excess patients in the gait_parameters dataset as at risk or not at risk of falls
gait_params["at risk of falls"] = [None] * len(gait_params)
for i in range(0, len(gait_params)):
    if gait_params.loc[i,"Name"] != demographics.loc[dem_index,"Subject ID"]:
        dem_index += 1
    gait_params.loc[i,"at risk of falls"] = demographics.loc[dem_index,"at risk of falls"]
# Removing excess negative values
for i in range(0, len(gait_params)):
    if gait_params.loc[i,"at risk of falls"] not in range(0,2):
        gait_params.drop(i,inplace=True)
print(f"Final Length = {len(gait_params)}")
# Dropping irrelevant columns and balancing dataset with SMOTE
gait_params.drop(columns=["Left_Limp_Index", "Right_Limp_Index", "Left_Foot_Off", "Right_Foot_Off", "Gait Duration after data crop"],inplace=True)
X = gait_params.drop(columns=["at risk of falls", "Name"]) # Independent Variable
y = gait_params["at risk of falls"].astype(int) # Dependent Variable
smote = SMOTE(sampling_strategy={1:100},random_state=42)
X_resampled, y_resampled = smote.fit_resample(X,y)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
