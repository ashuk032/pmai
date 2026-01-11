import json
import os
from typing import Dict


def ensure_dirs():
    for d in ["data", "models", "reports"]:
        os.makedirs(d, exist_ok=True)


def save_json(path: str, obj: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def risk_to_priority(risk: float) -> str:
    if risk >= 0.7:
        return "High"
    if risk >= 0.4:
        return "Medium"
    return "Low"


def estimate_ttf_hours(risk: float) -> float:
    # Simple heuristic: higher risk => shorter TTF
    # avoid div-by-zero; map risk in (0,1] to ~[20, 500] hours
    r = max(1e-3, min(0.999, risk))
    return float(max(6.0, 500.0 * (1.0 - r) ** 1.5 + 20.0 * (1.0 - r)))
