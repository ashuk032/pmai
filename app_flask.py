from flask import Flask, render_template, redirect, url_for
import os
import pandas as pd

from src.data_gen import generate_synthetic
from src.train import train_and_evaluate
from src.infer import score

app = Flask(__name__)


def get_latest_scored(num_components: int = 25, days: int = 120, freq: int = 60) -> pd.DataFrame:
    df = generate_synthetic(num_components=num_components, days=days, freq_minutes=freq)
    latest = df.sort_values(["component_id", "timestamp"]).groupby("component_id").tail(1)
    scored = score(latest)
    return scored


def get_component_history_scored(component_id: int, days: int = 120, freq: int = 60) -> pd.DataFrame:
    df = generate_synthetic(num_components=30, days=days, freq_minutes=freq)
    hist = df[df["component_id"] == component_id].sort_values("timestamp")
    if hist.empty:
        # fallback if the specific component id is not present in generated set
        hist = df[df["component_id"] == df["component_id"].min()].sort_values("timestamp")
    hist_scored = score(hist)
    return hist_scored


@app.route("/")
def index():
    scored = get_latest_scored()
    table = scored[["component_id", "component_type", "risk", "priority", "ttf_hours"]]

    # Data for Chart.js (sorted by risk desc)
    chart_df = table.sort_values("risk", ascending=False)
    labels = chart_df["component_id"].astype(str).tolist()
    risks = (chart_df["risk"] * 100).round(1).tolist()

    return render_template(
        "index.html",
        rows=table.sort_values("risk", ascending=False).to_dict(orient="records"),
        labels=labels,
        risks=risks,
    )


@app.route("/component/<int:component_id>")
def component_detail(component_id: int):
    hist_scored = get_component_history_scored(component_id)

    # Risk time series
    ts = hist_scored["timestamp"].dt.strftime("%Y-%m-%d %H:%M").tolist()
    risk_series = (hist_scored["risk"] * 100).round(1).tolist()

    # Sensor trends
    temperature = hist_scored["temperature"].round(2).tolist()
    vibration = hist_scored["vibration"].round(4).tolist()
    load = hist_scored["load"].round(3).tolist()

    meta = hist_scored.iloc[-1].to_dict()

    return render_template(
        "component.html",
        component_id=int(meta.get("component_id", component_id)),
        component_type=str(meta.get("component_type", "unknown")),
        ts=ts,
        risk_series=risk_series,
        temperature=temperature,
        vibration=vibration,
        load=load,
    )


@app.route("/retrain")
def retrain():
    train_and_evaluate()
    return redirect(url_for("index"))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
