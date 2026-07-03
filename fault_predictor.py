from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from data_simulator import TEST_FRACTION, TRAIN_FRACTION, load_ai4i_dataset, load_ai4i_split


MODEL_NAME = "Random Forest"
MODEL_TARGET = "Target: Machine failure"
AI4I_FEATURES = [
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
]
DERIVED_FEATURES = [
    "oil_temp",
    "hydraulic_temp",
    "bearing_temp",
    "vibration",
]
FAULT_THRESHOLDS = {
    "oil_temp": 110,
    "vibration": 10,
    "bearing_temp": 115,
}


@lru_cache(maxsize=1)
def _train_ai4i_model():
    data = load_ai4i_split("train")
    if data.empty:
        return None

    training_data = data.dropna(subset=AI4I_FEATURES + ["machine_failure"])
    if training_data["machine_failure"].nunique() < 2:
        return None

    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=42,
    )
    model.fit(training_data[AI4I_FEATURES], training_data["machine_failure"])
    return model


@lru_cache(maxsize=1)
def get_model_diagnostics():
    model = _train_ai4i_model()
    train_df = load_ai4i_split("train")
    test_df = load_ai4i_split("test")
    if model is None or train_df.empty or test_df.empty:
        return {
            "model_name": MODEL_NAME,
            "target": MODEL_TARGET,
            "train_rows": 0,
            "test_rows": 0,
            "split": "Unavailable",
            "accuracy": None,
            "recall": None,
            "precision": None,
        }

    test_data = test_df.dropna(subset=AI4I_FEATURES + ["machine_failure"])
    y_true = test_data["machine_failure"].astype(int)
    y_pred = model.predict(test_data[AI4I_FEATURES]).astype(int)
    true_positive = int(((y_true == 1) & (y_pred == 1)).sum())
    false_positive = int(((y_true == 0) & (y_pred == 1)).sum())
    false_negative = int(((y_true == 1) & (y_pred == 0)).sum())
    accuracy = float((y_true == y_pred).mean())
    recall = true_positive / max(true_positive + false_negative, 1)
    precision = true_positive / max(true_positive + false_positive, 1)

    return {
        "model_name": MODEL_NAME,
        "target": MODEL_TARGET,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "split": f"{int(TRAIN_FRACTION * 100)}% train / {int(TEST_FRACTION * 100)}% test",
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
    }


def _train_threshold_fallback(history_df):
    if not all(column in history_df.columns for column in DERIVED_FEATURES):
        return None

    data = history_df.copy()
    data["fault"] = (
        (data["oil_temp"] > FAULT_THRESHOLDS["oil_temp"])
        | (data["vibration"] > FAULT_THRESHOLDS["vibration"])
        | (data["bearing_temp"] > FAULT_THRESHOLDS["bearing_temp"])
    ).astype(int)
    data = data.dropna(subset=DERIVED_FEATURES + ["fault"])
    if data["fault"].nunique() < 2:
        return None

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=6,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=42,
    )
    model.fit(data[DERIVED_FEATURES], data["fault"])
    return model


def _ai4i_explanation(selected_row):
    explanations = []
    tool_wear = float(selected_row.get("tool_wear_min", 0))
    torque = float(selected_row.get("torque_nm", 0))
    rpm = float(selected_row.get("rotational_speed_rpm", 0))
    air_temp = float(selected_row.get("air_temperature_k", 0))
    process_temp = float(selected_row.get("process_temperature_k", 0))
    power_kw = torque * rpm / 9550 if rpm else 0

    if tool_wear >= 180:
        explanations.append("high tool wear")
    if process_temp - air_temp >= 10.5:
        explanations.append("high process-to-air temperature gap")
    if power_kw < 3.0 or power_kw > 9.0:
        explanations.append("abnormal power load")
    if torque * tool_wear >= 11000:
        explanations.append("overstrain from torque and tool wear")

    return ", ".join(explanations) if explanations else "No strong AI4I failure drivers detected"


def _threshold_explanation(selected_row):
    reasons = []
    for col, thresh in FAULT_THRESHOLDS.items():
        if selected_row[col] > thresh:
            reasons.append(f"{col.replace('_', ' ').title()} above {thresh}")
    return ", ".join(reasons) if reasons else "No critical readings detected"


def predict_fault(selected_row: pd.Series, history_df: pd.DataFrame):
    """Predict failure probability and estimated downtime for one reading."""
    if all(feature in selected_row for feature in AI4I_FEATURES) and not load_ai4i_dataset().empty:
        model = _train_ai4i_model()
        features = AI4I_FEATURES
        explanation = _ai4i_explanation(selected_row)
    else:
        model = _train_threshold_fallback(history_df)
        features = DERIVED_FEATURES
        explanation = _threshold_explanation(selected_row)

    if model is None:
        return 0.0, 0.5, "Insufficient data to train predictive model"

    feature_frame = selected_row[features].astype(float).to_frame().T
    probability = float(model.predict_proba(feature_frame)[0, 1])
    if np.isnan(probability):
        return 0.0, 0.5, "Prediction unavailable for this reading"

    downtime = max(0.5, probability * 6)
    return probability, downtime, explanation
