# data_simulator.py
import pandas as pd
import numpy as np
import datetime

def get_live_data():
    now = datetime.datetime.now()
    data = {
        "timestamp": now,
        "oil_temp": np.random.normal(72, 5),
        "hydraulic_temp": np.random.normal(65, 4),
        "bearing_temp": np.random.normal(85, 6),
        "vibration": np.random.normal(5.0, 1.0),
        "units_produced": np.random.randint(300, 500),
        "defect_rate": round(np.random.uniform(0.5, 2.0), 2),
        "production_efficiency": round(np.random.uniform(85, 98), 2),
        "energy_usage": round(np.random.uniform(150, 250), 2),
        "energy_cost": round(np.random.uniform(20, 35), 2),
        "machine_type": np.random.choice(['CNC', 'Lathe', 'Milling'])
    }
    return pd.DataFrame([data])

def generate_fake_history(n=50, freq='H'):
    timestamps = pd.date_range(end=pd.Timestamp.now(), periods=n, freq=freq)
    data = []

    for ts in timestamps:
        row = {
            "timestamp": ts,
            "oil_temp": np.random.normal(72, 5),
            "hydraulic_temp": np.random.normal(65, 4),
            "bearing_temp": np.random.normal(85, 6),
            "vibration": np.random.normal(5.0, 1.0),
            "units_produced": np.random.randint(300, 500),
            "defect_rate": round(np.random.uniform(0.5, 2.0), 2),
            "production_efficiency": round(np.random.uniform(85, 98), 2),
            "energy_usage": round(np.random.uniform(150, 250), 2),
            "energy_cost": round(np.random.uniform(20, 35), 2),
            "machine_type": np.random.choice(['CNC', 'Lathe', 'Milling'])
        }
        data.append(row)

    return pd.DataFrame(data)
