import pandas as pd
import numpy as np
import datetime

def get_live_data():
    now = datetime.datetime.now()
    machine_types = ['CNC', 'Lathe', 'Milling']

    expected_output = {
        "CNC": 4,
        "Lathe": 3,
        "Milling": 2
    }

    data = []

    for m_type in machine_types:
        # Simulate actual output
        actual_output = np.random.randint(1, expected_output[m_type] + 1)

        # Calculate efficiency
        efficiency = round((actual_output / expected_output[m_type]) * 100, 2)

        # Random defect rate
        defect_rate = round(np.random.uniform(0.5, 2.0), 2)

        # Occasional alert-triggering spikes
        spike = np.random.rand() < 0.05

        row = {
            "timestamp": now,
            "machine_type": m_type,
            "oil_temp": np.random.normal(72, 5) + (40 if spike else 0),
            "hydraulic_temp": np.random.normal(65, 4),
            "bearing_temp": np.random.normal(85, 6) + (35 if spike else 0),
            "vibration": np.random.normal(5.0, 1.0) + (6 if spike else 0),
            "units_produced": actual_output,
            "defect_rate": defect_rate,
            "production_efficiency": efficiency,
            "energy_usage": round(np.random.uniform(1, 4), 2),
            "energy_cost": round(np.random.uniform(0.2, 0.6), 2)
        }

        data.append(row)

    return pd.DataFrame(data)
