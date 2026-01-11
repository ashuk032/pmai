# AI-Driven Predictive Maintenance (Airport Infrastructure)

Quickstart for hackathon demo: end-to-end pipeline (synthetic data → train → evaluate → Streamlit dashboard).

## Setup

1. Create venv and install deps
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run baseline

- Generate data, train, evaluate
```
python -m src/train.py
```
- Launch Streamlit dashboard
```
python -m streamlit run app.py
```

## Minimal Flask frontend (Tailwind + Chart.js)

Use this if you prefer a classic web app instead of Streamlit.

```
python -m app_flask.py
```

Then open http://localhost:5000

Routes:
- `/` Overview: risk table and risk distribution bar chart
- `/component/<id>` Component detail: risk over time + key sensor trends
- `/retrain` Retrains the model on synthetic data

## Outputs
- data/data.csv — synthetic dataset
- models/model.pkl — best trained classifier (sklearn API)
- reports/metrics.json — metrics

## Notes
- Features include scaling + basic engineered indicators (usage-normalized, deviations).
- Risk score (0–100%), TTF estimate (heuristic), priority banding.
