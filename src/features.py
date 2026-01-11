from typing import List, Tuple
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAW_NUM = [
    "temperature", "vibration", "voltage", "load", "humidity",
    "usage_hours", "time_since_maintenance"
]
RAW_CAT = ["component_type"]
LABEL = "next_failed"
ID_TS = ["component_id", "timestamp", "failed"]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # deviations from nominal
    df["temp_dev"] = df["temperature"] - 40.0
    df["volt_dev"] = df["voltage"] - 220.0
    df["vib_sq"] = df["vibration"] ** 2
    df["load_x_vib"] = df["load"] * df["vibration"]
    # normalized usage
    df["usage_norm"] = df["usage_hours"] / (df.groupby("component_id")["usage_hours"].transform("max") + 1e-6)
    # time since maintenance capped
    df["tsm_capped"] = np.clip(df["time_since_maintenance"], 0, 200)
    return df


ENGINEERED_NUM = RAW_NUM + ["temp_dev", "volt_dev", "vib_sq", "load_x_vib", "usage_norm", "tsm_capped"]


def build_preprocess_pipeline(cat_cols: List[str] = RAW_CAT, num_cols: List[str] = ENGINEERED_NUM) -> ColumnTransformer:
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    ct = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])
    return ct


def split_features_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    X = df.drop(columns=[LABEL])
    y = df[LABEL]
    return X, y, df[ID_TS]
