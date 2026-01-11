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


@app.route("/components")
def components_list():
    scored = get_latest_scored()
    table = scored[["component_id", "component_type", "risk", "priority", "ttf_hours"]].copy()

    # Filters
    f_type = request.args.get("type", "")
    f_prio = request.args.get("priority", "")
    try:
        th_high = float(request.args.get("thresh_high", 0.7))
        th_med = float(request.args.get("thresh_med", 0.4))
        if th_med > th_high:
            th_med = th_high
    except Exception:
        th_high, th_med = 0.7, 0.4
    def map_prio(r: float) -> str:
        if r >= th_high:
            return "High"
        if r >= th_med:
            return "Medium"
        return "Low"
    table["priority"] = table["risk"].apply(map_prio)
    if f_type:
        table = table[table["component_type"] == f_type]
    if f_prio:
        table = table[table["priority"] == f_prio]

    # Pagination
    page = max(1, int(request.args.get("page", 1)))
    per_page = min(50, max(5, int(request.args.get("per_page", 15))))
    total_items = int(len(table))
    total_pages = max(1, (total_items + per_page - 1) // per_page)
    page = min(page, total_pages)
    start = (page - 1) * per_page
    end = start + per_page
    page_rows = table.sort_values("risk", ascending=False).iloc[start:end]

    types = sorted(scored["component_type"].unique().tolist())

    return render_template(
        "components.html",
        rows=page_rows.to_dict(orient="records"),
        page=page,
        per_page=per_page,
        total_pages=total_pages,
        total_items=total_items,
        types=types,
        f_type=f_type,
        f_prio=f_prio,
        th_high=th_high,
        th_med=th_med,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
