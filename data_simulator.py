import datetime

import numpy as np
import pandas as pd


def _machine_state(health_state, machine_type):
    if machine_type not in health_state:
        health_state[machine_type] = {
            "bearing_wear": np.random.uniform(0.05, 0.18),
            "lubrication_stress": np.random.uniform(0.03, 0.14),
            "hydraulic_stress": np.random.uniform(0.03, 0.12),
        }
    return health_state[machine_type]


def _advance_state(state):
    state["bearing_wear"] = min(1.0, state["bearing_wear"] + np.random.uniform(0.003, 0.018))
    state["lubrication_stress"] = min(1.0, state["lubrication_stress"] + np.random.uniform(0.002, 0.015))
    state["hydraulic_stress"] = min(1.0, state["hydraulic_stress"] + np.random.uniform(0.001, 0.011))

    if np.random.rand() < 0.04:
        stressed_component = np.random.choice(list(state.keys()))
        state[stressed_component] = min(1.0, state[stressed_component] + 0.12)


def get_live_data(health_state=None):
    now = datetime.datetime.now()
    machine_types = ["CNC", "Lathe", "Milling"]
    if health_state is None:
        health_state = {}

    expected_output = {
        "CNC": 4,
        "Lathe": 3,
        "Milling": 2,
    }

    data = []

    for m_type in machine_types:
        state = _machine_state(health_state, m_type)
        _advance_state(state)

        actual_output = np.random.randint(1, expected_output[m_type] + 1)
        wear_drag = (state["bearing_wear"] + state["lubrication_stress"]) / 2
        efficiency = max(35, round((actual_output / expected_output[m_type]) * 100 - wear_drag * 16, 2))
        defect_rate = round(np.random.uniform(0.5, 1.4) + state["bearing_wear"] * 1.2, 2)

        data.append({
            "timestamp": now,
            "machine_type": m_type,
            "oil_temp": np.random.normal(70, 3) + state["lubrication_stress"] * 38,
            "hydraulic_temp": np.random.normal(63, 3) + state["hydraulic_stress"] * 30,
            "bearing_temp": np.random.normal(82, 4) + state["bearing_wear"] * 38,
            "vibration": np.random.normal(4.6, 0.65) + state["bearing_wear"] * 6.2,
            "units_produced": actual_output,
            "defect_rate": defect_rate,
            "production_efficiency": efficiency,
            "energy_usage": round(np.random.uniform(1, 4) + wear_drag * 1.8, 2),
            "energy_cost": round(np.random.uniform(0.2, 0.6), 2),
        })

    return pd.DataFrame(data)
