import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data_simulator import FAILURE_LABELS, load_ai4i_dataset


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


def _model_training_frame(history_df):
    ai4i_df = load_ai4i_dataset()
    if not ai4i_df.empty:
        return ai4i_df, AI4I_FEATURES, "machine_failure"
    if all(column in history_df.columns for column in DERIVED_FEATURES):
        data = history_df.copy()
        data["fault"] = (
            (data["oil_temp"] > FAULT_THRESHOLDS["oil_temp"])
            | (data["vibration"] > FAULT_THRESHOLDS["vibration"])
            | (data["bearing_temp"] > FAULT_THRESHOLDS["bearing_temp"])
        ).astype(int)
        return data, DERIVED_FEATURES, "fault"
    return pd.DataFrame(), [], ""


def _train_model(history_df):
    data, features, target = _model_training_frame(history_df)
    if data.empty or not features or target not in data:
        return None, features
    data = data.dropna(subset=features + [target])
    if data[target].nunique() < 2:
        return None, features
    try:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        )
        model.fit(data[features], data[target])
        return model, features
    except Exception:
        return None, features


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

    labeled_failures = [
        label for code, label in FAILURE_LABELS.items()
        if int(selected_row.get(code, 0)) == 1
    ]
    if labeled_failures:
        explanations.append("dataset label: " + ", ".join(labeled_failures))

    return ", ".join(explanations) if explanations else "No strong AI4I failure drivers detected"


def _threshold_explanation(selected_row):
    reasons = []
    for col, thresh in FAULT_THRESHOLDS.items():
        if selected_row[col] > thresh:
            reasons.append(f"{col.replace('_', ' ').title()} above {thresh}")
    return ", ".join(reasons) if reasons else "No critical readings detected"


def predict_fault(selected_row: pd.Series, history_df: pd.DataFrame):
    """Predict fault probability and downtime for the selected machine."""
    model, features = _train_model(history_df)
    if model is None or not features:
        return 0.0, 0.5, "Insufficient data to train predictive model"

    feature_frame = selected_row[features].astype(float).to_frame().T
    probability = float(model.predict_proba(feature_frame)[0, 1])
    downtime = max(0.5, probability * 6)

    if all(feature in selected_row for feature in AI4I_FEATURES):
        explanation = _ai4i_explanation(selected_row)
    else:
        explanation = _threshold_explanation(selected_row)

    if np.isnan(probability):
        return 0.0, 0.5, "Prediction unavailable for this reading"
    return probability, downtime, explanation
