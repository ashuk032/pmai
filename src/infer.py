import os
import joblib
import pandas as pd
from typing import Tuple

from src.features import add_engineered_features
from src.utils import risk_to_priority, estimate_ttf_hours

MODEL_PATH = "models/model.pkl"


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model file not found. Run training first.")
    return joblib.load(MODEL_PATH)


def score(df: pd.DataFrame) -> pd.DataFrame:
    model = load_model()
    df_eng = add_engineered_features(df)
    # select same feature cols as train (drop id/time-only)
    cols = [c for c in df_eng.columns if c not in ["timestamp", "component_id", "failed", "next_failed"]]
    proba = model.predict_proba(df_eng[cols])[:, 1]
    out = df_eng.copy()
    out["risk"] = proba
    out["priority"] = out["risk"].apply(risk_to_priority)
    out["ttf_hours"] = out["risk"].apply(estimate_ttf_hours)
    return out
