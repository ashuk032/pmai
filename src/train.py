import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from src.data_gen import generate_synthetic, save_dataset
from src.features import add_engineered_features, build_preprocess_pipeline, split_features_labels
from src.utils import ensure_dirs, save_json

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False


def load_or_generate(path: str = "data/data.csv") -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["timestamp"])    
    df = generate_synthetic()
    save_dataset(df, path)
    return df


def train_and_evaluate(random_state: int = 42):
    ensure_dirs()
    df = load_or_generate()

    df_feat = add_engineered_features(df)

    # supervised on snapshots
    X_all, y_all, meta = split_features_labels(df_feat)

    # select feature columns for preprocessing
    feat_cols = [c for c in X_all.columns if c not in ["timestamp", "component_id", "failed"]]
    X = X_all[feat_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_all, test_size=0.2, random_state=random_state, stratify=y_all
    )

    preprocess = build_preprocess_pipeline(
        cat_cols=["component_type"],
        num_cols=[c for c in feat_cols if c != "component_type"]
    )

    models = []
    rf = Pipeline([
        ("prep", preprocess),
        ("clf", RandomForestClassifier(n_estimators=300, max_depth=None, random_state=random_state, n_jobs=-1))
    ])
    models.append(("random_forest", rf))

    if HAS_XGB:
        xgb = Pipeline([
            ("prep", preprocess),
            ("clf", XGBClassifier(
                n_estimators=400, max_depth=6, learning_rate=0.08,
                subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                random_state=random_state, n_jobs=-1, eval_metric="logloss"
            ))
        ])
        models.append(("xgboost", xgb))

    metrics_all = {}
    best_name, best_model, best_auc = None, None, -1.0
    for name, pipe in models:
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        acc = float(accuracy_score(y_test, pred))
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)
        try:
            auc = float(roc_auc_score(y_test, proba))
        except Exception:
            auc = float("nan")
        cm = confusion_matrix(y_test, pred).tolist()
        metrics_all[name] = {
            "accuracy": acc,
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "roc_auc": auc,
            "confusion_matrix": cm
        }
        if auc > best_auc:
            best_auc = auc
            best_name = name
            best_model = pipe

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/model.pkl")
    save_json("reports/metrics.json", {"models": metrics_all, "best_model": best_name})
    print("Saved best model:", best_name)
    print("Metrics:", metrics_all[best_name])


if __name__ == "__main__":
    train_and_evaluate()
