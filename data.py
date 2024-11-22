import pandas
demographics = pandas.read_csv("01_demography_sppb_mmse.xlsx")
gait_params = pandas.read_csv("05_gait_parameters.xlsx")
demographics.rename(columns={"self_reported_falls": "at risk of falls"}, inplace=True)
demographics = demographics[demographics["at risk fo falls"] != 0].iloc[20:]
valid_IDs = demographics["subject_id"].tolist()
demographics = demographics[demographics["subject_id"].isin(valid_IDs)]
y = gait_params["at risk of falls"]

