import pandas as pd
from pathlib import Path
pd.set_option('display.max_colwidth',None)
pd.set_option('display.max_rows',None)
pd.set_option('display.width',None)
pd.set_option('display.max_columns',None)
GRF_right_path = Path("Force_Data") / "GRF_F_V_PRO_right.csv"
GRF_left_path = Path("Force_Data") / "GRF_F_V_PRO_left.csv"
metadata_path = Path("Force_Data") / "GRF_metadata.csv"
GRF_right = pd.read_csv(GRF_right_path)
GRF_left = pd.read_csv(GRF_left_path)
metadata = pd.read_csv(metadata_path)
metadata = metadata.drop_duplicates(subset=["SUBJECT_ID"])
metadata = metadata[metadata["CLASS_LABEL"].isin(["HC","H"])]
metadata = metadata.drop(metadata[metadata["CLASS_LABEL"] == "H"].iloc[len(metadata[metadata["CLASS_LABEL"] == "H"]) // 2:].index)
metadata["at risk of falls"] = [None] * len(metadata)
metadata["at risk of falls"] = metadata["CLASS_LABEL"].apply(lambda x: 0 if x == "HC" else 1)
valid_IDs = metadata["SUBJECT_ID"].tolist()
GRF_right = GRF_right[GRF_right["SUBJECT_ID"].isin(valid_IDs)]
GRF_left = GRF_left[GRF_left["SUBJECT_ID"].isin(valid_IDs)]
GRF_right = GRF_right.merge(metadata[["SUBJECT_ID","at risk of falls"]], on="SUBJECT_ID", how="inner")
GRF_left = GRF_left.merge(metadata[["SUBJECT_ID","at risk of falls"]], on="SUBJECT_ID", how="inner")
GRF_right = GRF_right.merge(metadata[["SUBJECT_ID","TRAIN"]], on="SUBJECT_ID", how="inner")
GRF_left = GRF_left.merge(metadata[["SUBJECT_ID","TRAIN"]], on="SUBJECT_ID", how="inner")
GRF_right_train = GRF_right[GRF_right["TRAIN"] == 1]
GRF_right_test = GRF_right[GRF_right["TRAIN"] == 0]
GRF_left_train = GRF_left[GRF_left["TRAIN"] == 1]
GRF_left_test = GRF_left[GRF_left["TRAIN"] == 0]
