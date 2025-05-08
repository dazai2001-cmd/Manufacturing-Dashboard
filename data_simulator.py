import pandas as pd
import numpy as np
import datetime

def get_live_data():
    now = datetime.datetime.now()
    machine_types = ['CNC', 'Lathe', 'Milling']
    data = []

    for m_type in machine_types:
        # 5% chance to simulate a spike
        spike = np.random.rand() < 0.05

        row = {
            "timestamp": now,
            "machine_type": m_type,
            "oil_temp": 115,
            "hydraulic_temp": np.random.normal(65, 4),
            "bearing_temp": np.random.normal(85, 6) + (35 if spike else 0),
            "vibration": np.random.normal(5.0, 1.0) + (6 if spike else 0),
            "units_produced": np.random.randint(1, 4),
            "defect_rate": round(np.random.uniform(0.5, 2.0), 2),
            "production_efficiency": round(np.random.uniform(85, 98), 2),
            "energy_usage": round(np.random.uniform(1, 4), 2),
            "energy_cost": round(np.random.uniform(0.2, 0.6), 2),
        }
        data.append(row)

    return pd.DataFrame(data)
