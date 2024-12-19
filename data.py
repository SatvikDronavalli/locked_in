import pandas as pd
from pathlib import Path
gait_file = Path("IMU_Data") / "05_gait_parameters.xlsx"
demographics_file = Path ("IMU_Data") / "01_demography_sppb_mmse.xlsx"
gait_params = pd.read_excel(gait_file)
demographics = pd.read_excel(demographics_file)
demographics.rename(columns={"self_reported_falls": "at risk of falls"}, inplace=True)
demographics = demographics[demographics["at risk fo falls"] != 0].iloc[20:]
valid_IDs = demographics["subject_id"].tolist()
demographics = demographics[demographics["subject_id"].isin(valid_IDs)]
y = gait_params["at risk of falls"]
