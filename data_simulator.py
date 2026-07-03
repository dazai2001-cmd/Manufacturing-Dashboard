import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


DATASET_PATH = Path(__file__).resolve().parent / "data" / "ai4i2020.csv"
TRAIN_FRACTION = 0.70
TEST_FRACTION = 0.30
SPLIT_RANDOM_STATE = 42
EXPECTED_OUTPUT = {
    "H": 4,
    "M": 3,
    "L": 2,
}
FAILURE_LABELS = {
    "TWF": "Tool wear failure",
    "HDF": "Heat dissipation failure",
    "PWF": "Power failure",
    "OSF": "Overstrain failure",
    "RNF": "Random failure",
}


def _kelvin_to_celsius(value):
    return float(value) - 273.15


def _clamp(value, lower, upper):
    return max(lower, min(upper, value))


@lru_cache(maxsize=1)
def load_ai4i_dataset():
    if not DATASET_PATH.exists():
        return pd.DataFrame()

    df = pd.read_csv(DATASET_PATH)
    df = df.rename(columns={
        "UDI": "udi",
        "Product ID": "product_id",
        "Type": "product_type",
        "Air temperature [K]": "air_temperature_k",
        "Process temperature [K]": "process_temperature_k",
        "Rotational speed [rpm]": "rotational_speed_rpm",
        "Torque [Nm]": "torque_nm",
        "Tool wear [min]": "tool_wear_min",
        "Machine failure": "machine_failure",
    })
    df["machine_type"] = df["product_type"]
    df["air_temp_c"] = df["air_temperature_k"].map(_kelvin_to_celsius)
    df["process_temp_c"] = df["process_temperature_k"].map(_kelvin_to_celsius)
    df["power_kw"] = df["torque_nm"] * df["rotational_speed_rpm"] / 9550
    df["failure_type"] = df.apply(_failure_type, axis=1)
    return df


@lru_cache(maxsize=2)
def load_ai4i_split(split):
    dataset = load_ai4i_dataset()
    if dataset.empty:
        return pd.DataFrame()

    train_df, test_df = train_test_split(
        dataset,
        test_size=TEST_FRACTION,
        random_state=SPLIT_RANDOM_STATE,
        stratify=dataset["machine_failure"],
    )
    train_df = train_df.sort_values("udi").copy()
    test_df = test_df.sort_values("udi").copy()
    train_df["dataset_split"] = "train"
    test_df["dataset_split"] = "test"

    if split == "train":
        return train_df.reset_index(drop=True)
    if split == "test":
        return test_df.reset_index(drop=True)
    return dataset.copy().reset_index(drop=True)


def _failure_type(row):
    failures = [label for col, label in FAILURE_LABELS.items() if int(row.get(col, 0)) == 1]
    if failures:
        return ", ".join(failures)
    return "None"


def _dashboard_row_from_ai4i(row, now):
    machine = row["machine_type"]
    expected = EXPECTED_OUTPUT[machine]
    failure = int(row["machine_failure"])
    tool_wear = float(row["tool_wear_min"])
    torque = float(row["torque_nm"])
    rpm = float(row["rotational_speed_rpm"])
    power_kw = float(row["power_kw"])
    process_temp_c = float(row["process_temp_c"])
    air_temp_c = float(row["air_temp_c"])
    torque_load = _clamp(torque / 65, 0, 1.4)
    wear_load = _clamp(tool_wear / 250, 0, 1.2)
    rpm_instability = _clamp(abs(rpm - 1500) / 700, 0, 1.4)

    oil_temp = process_temp_c + 42 + torque_load * 18 + wear_load * 10
    hydraulic_temp = air_temp_c + 34 + power_kw * 2.8 + rpm_instability * 5
    bearing_temp = process_temp_c + 38 + torque_load * 15 + wear_load * 18
    vibration = 2.2 + rpm_instability * 2.4 + torque_load * 1.2 + wear_load * 3.4

    if failure:
        oil_temp += 7
        hydraulic_temp += 5
        bearing_temp += 9
        vibration += 1.2

    unit_penalty = 1 if failure else 0
    actual_output = max(1, expected - unit_penalty - int(wear_load > 0.9))
    efficiency = _clamp((actual_output / expected) * 100 - wear_load * 12 - failure * 10, 20, 100)
    defect_rate = 0.4 + wear_load * 2.2 + torque_load * 0.6 + failure * 2.8
    energy_usage = max(0.5, power_kw * 0.45 + torque_load * 1.1)

    return {
        "timestamp": now,
        "machine_type": machine,
        "product_type": row["product_type"],
        "product_id": row["product_id"],
        "udi": int(row["udi"]),
        "air_temperature_k": row["air_temperature_k"],
        "process_temperature_k": row["process_temperature_k"],
        "air_temp_c": air_temp_c,
        "process_temp_c": process_temp_c,
        "rotational_speed_rpm": rpm,
        "torque_nm": torque,
        "tool_wear_min": tool_wear,
        "power_kw": power_kw,
        "machine_failure": failure,
        "failure_type": row["failure_type"],
        "dataset_split": row.get("dataset_split", "test"),
        "TWF": int(row["TWF"]),
        "HDF": int(row["HDF"]),
        "PWF": int(row["PWF"]),
        "OSF": int(row["OSF"]),
        "RNF": int(row["RNF"]),
        "oil_temp": oil_temp,
        "hydraulic_temp": hydraulic_temp,
        "bearing_temp": bearing_temp,
        "vibration": vibration,
        "units_produced": actual_output,
        "defect_rate": round(defect_rate, 2),
        "production_efficiency": round(efficiency, 2),
        "energy_usage": round(energy_usage, 2),
        "energy_cost": round(energy_usage * 0.18, 2),
    }


def _get_ai4i_live_data(replay_state):
    now = datetime.datetime.now()
    dataset = load_ai4i_split("test")
    cursors = replay_state.setdefault("ai4i_cursors", {})
    rows = []

    for product_type in ["H", "M", "L"]:
        machine_rows = dataset[dataset["product_type"] == product_type].reset_index(drop=True)
        cursor = cursors.get(product_type)
        if cursor is None:
            failure_positions = np.flatnonzero(machine_rows["machine_failure"].to_numpy() == 1)
            cursor = max(int(failure_positions[0]) - 8, 0) if len(failure_positions) else 0
        row = machine_rows.iloc[cursor % len(machine_rows)]
        cursors[product_type] = cursor + 1
        rows.append(_dashboard_row_from_ai4i(row, now))

    return pd.DataFrame(rows)


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


def _get_synthetic_live_data(health_state):
    now = datetime.datetime.now()
    machine_types = ["H", "M", "L"]
    data = []

    for machine in machine_types:
        state = _machine_state(health_state, machine)
        _advance_state(state)

        actual_output = np.random.randint(1, EXPECTED_OUTPUT[machine] + 1)
        wear_drag = (state["bearing_wear"] + state["lubrication_stress"]) / 2
        efficiency = max(35, round((actual_output / EXPECTED_OUTPUT[machine]) * 100 - wear_drag * 16, 2))
        defect_rate = round(np.random.uniform(0.5, 1.4) + state["bearing_wear"] * 1.2, 2)

        data.append({
            "timestamp": now,
            "machine_type": machine,
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


def get_live_data(health_state=None):
    if health_state is None:
        health_state = {}
    if not load_ai4i_split("test").empty:
        return _get_ai4i_live_data(health_state)
    return _get_synthetic_live_data(health_state)
