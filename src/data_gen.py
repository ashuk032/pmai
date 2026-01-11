import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RNG = np.random.default_rng(42)

COMPONENT_TYPES = [
    "runway_light", "baggage_unit", "escalator", "hvac", "power_subsystem"
]

SENSORS = ["temperature", "vibration", "voltage", "load", "humidity"]


def _simulate_component_series(component_id: int, days: int = 120, freq_minutes: int = 60):
    start = datetime.now() - timedelta(days=days)
    timestamps = pd.date_range(start=start, periods=days * (60 // (freq_minutes if freq_minutes else 60)) * 24,
                               freq=f"{freq_minutes}min")
    comp_type = RNG.choice(COMPONENT_TYPES)
    base_temp = RNG.normal(30, 5)
    base_vib = abs(RNG.normal(0.5, 0.1))
    base_volt = RNG.normal(220, 5)
    base_load = abs(RNG.normal(0.6, 0.2))
    base_hum = RNG.normal(50, 10)

    usage_hours = np.cumsum(np.ones(len(timestamps)) * (freq_minutes / 60.0))
    last_maint = 0.0
    last_maint_arr = []

    rows = []
    for i, ts in enumerate(timestamps):
        # drift + noise
        temperature = base_temp + 0.02 * usage_hours[i] + RNG.normal(0, 0.8)
        vibration = base_vib + 0.001 * usage_hours[i] + abs(RNG.normal(0, 0.02))
        voltage = base_volt + RNG.normal(0, 1.0)
        load = np.clip(base_load + RNG.normal(0, 0.05), 0, 1.5)
        humidity = np.clip(base_hum + RNG.normal(0, 2.0), 5, 95)

        # stochastic maintenance resets
        if RNG.random() < 0.0015:
            last_maint = usage_hours[i]

        last_maint_arr.append(last_maint)

        # failure risk as function of deviation + time since maint
        dev = (
            0.6 * max(0, temperature - 45) +
            20 * max(0, vibration - 0.8) +
            0.5 * abs(voltage - 220) +
            30 * max(0, load - 1.0) +
            0.1 * max(0, humidity - 70)
        )
        time_since_maint = usage_hours[i] - last_maint
        hazard = 0.002 + 0.0005 * time_since_maint + 0.0005 * dev
        fail = RNG.random() < hazard

        rows.append({
            "timestamp": ts,
            "component_id": component_id,
            "component_type": comp_type,
            "temperature": temperature,
            "vibration": vibration,
            "voltage": voltage,
            "load": load,
            "humidity": humidity,
            "usage_hours": usage_hours[i],
            "time_since_maintenance": time_since_maint,
            "failed": int(fail)
        })

        if fail:
            # after failure, maintenance occurs and resets degradation
            last_maint = usage_hours[i]

    df = pd.DataFrame(rows)
    return df


def generate_synthetic(num_components: int = 25, days: int = 120, freq_minutes: int = 60) -> pd.DataFrame:
    frames = []
    for cid in range(num_components):
        frames.append(_simulate_component_series(cid, days=days, freq_minutes=freq_minutes))
    df = pd.concat(frames, ignore_index=True)

    # Create supervised label: next-period failure indicator
    df = df.sort_values(["component_id", "timestamp"])  # ensure order
    df["next_failed"] = df.groupby("component_id")["failed"].shift(-1).fillna(0).astype(int)

    return df


def save_dataset(df: pd.DataFrame, path: str = "data/data.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


if __name__ == "__main__":
    data = generate_synthetic()
    save_dataset(data)
    print(f"Saved synthetic dataset with shape: {data.shape}")
