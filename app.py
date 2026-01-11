import os
import time
import pandas as pd
import streamlit as st

from src.data_gen import generate_synthetic
from src.train import train_and_evaluate, load_or_generate
from src.infer import score

st.set_page_config(page_title="Airport Predictive Maintenance", layout="wide")

st.title("Airport Infrastructure Predictive Maintenance")
st.caption("Risk scoring, TTF estimation, and maintenance prioritization")

col1, col2, col3 = st.columns(3)
with col1:
    num_components = st.slider("Components", 5, 50, 25, 1)
with col2:
    days = st.slider("Days simulated", 30, 180, 120, 10)
with col3:
    freq = st.selectbox("Sampling frequency (minutes)", [15, 30, 60], index=2)

placeholder = st.empty()

@st.cache_data(show_spinner=False)
def get_data(num_components:int, days:int, freq:int):
    return generate_synthetic(num_components=num_components, days=days, freq_minutes=freq)

@st.cache_resource(show_spinner=False)
def ensure_model():
    # try to load_or_generate data and (re)train if model missing
    if not os.path.exists("models/model.pkl"):
        with st.spinner("Training model on synthetic data..."):
            train_and_evaluate()
            time.sleep(0.2)
    return True

# Controls
train_btn = st.button("(Re)Train Model on Synthetic Data", type="primary")
if train_btn:
    with st.spinner("Training..."):
        train_and_evaluate()
    st.success("Training complete.")

ensure_model()

# Data + scoring
with st.spinner("Simulating data and scoring..."):
    df = get_data(num_components, days, freq)
    # take latest snapshot per component for dashboard summary
    latest = df.sort_values(["component_id", "timestamp"]).groupby("component_id").tail(1)
    scored = score(latest)

st.subheader("Component Risk Overview")
st.dataframe(scored[["component_id", "component_type", "risk", "priority", "ttf_hours"]].sort_values("risk", ascending=False), use_container_width=True)

# Charts
st.subheader("Risk Distribution")
st.bar_chart(scored.set_index("component_id")["risk"].sort_values(ascending=False))

# Detail selection
st.subheader("Component Details")
comp_id = st.selectbox("Select component", scored["component_id"].unique())
comp_type = scored.loc[scored["component_id"]==comp_id, "component_type"].iloc[0]
st.write(f"Type: {comp_type}")

hist = df[df["component_id"]==comp_id].sort_values("timestamp")
hist_scored = score(hist)

colA, colB = st.columns(2)
with colA:
    st.line_chart(hist_scored.set_index("timestamp")["risk"], height=250)
    st.caption("Risk over time")
with colB:
    st.line_chart(hist_scored.set_index("timestamp")[["temperature", "vibration", "load"]], height=250)
    st.caption("Key sensor trends")

st.info("Tip: Use the retrain button to refresh the model on newly generated synthetic data.")
