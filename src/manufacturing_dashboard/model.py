from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from manufacturing_dashboard.data import load_ai4i_dataset, load_ai4i_split
from manufacturing_dashboard.training import (
    AI4I_FEATURES,
    METRICS_REPORT_PATH,
    MODEL_NAME,
    MODEL_TARGET,
    load_metrics_report,
    load_saved_artifact,
)


DEFAULT_FAILURE_THRESHOLD = 0.5
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
def _runtime_ai4i_artifact():
    artifact = load_saved_artifact()
    if artifact is not None:
        return artifact

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
    return {
        "model": model,
        "model_name": MODEL_NAME,
        "target": MODEL_TARGET,
        "features": AI4I_FEATURES,
        "threshold": DEFAULT_FAILURE_THRESHOLD,
        "feature_medians": training_data[AI4I_FEATURES].median().to_dict(),
    }


@lru_cache(maxsize=1)
def get_model_diagnostics():
    report = load_metrics_report()
    if report:
        test_metrics = report.get("test_metrics", {})
        return {
            "model_name": report.get("model_name", MODEL_NAME),
            "target": report.get("target", MODEL_TARGET),
            "train_rows": report.get("train_rows", 0),
            "validation_rows": report.get("validation_rows", 0),
            "test_rows": report.get("test_rows", 0),
            "split": report.get("split", "70% train / 30% test"),
            "threshold": report.get("chosen_threshold", DEFAULT_FAILURE_THRESHOLD),
            "threshold_strategy": report.get("threshold_strategy", ""),
            "accuracy": test_metrics.get("accuracy"),
            "recall": test_metrics.get("recall"),
            "precision": test_metrics.get("precision"),
            "f1": test_metrics.get("f1"),
            "f2": test_metrics.get("f2"),
            "roc_auc": test_metrics.get("roc_auc"),
            "model_comparison": report.get("model_comparison", []),
            "feature_importance": report.get("feature_importance", []),
            "report_path": str(METRICS_REPORT_PATH),
        }

    artifact = _runtime_ai4i_artifact()
    train_df = load_ai4i_split("train")
    test_df = load_ai4i_split("test")
    if artifact is None or train_df.empty or test_df.empty:
        return {
            "model_name": MODEL_NAME,
            "target": MODEL_TARGET,
            "train_rows": 0,
            "test_rows": 0,
            "split": "Unavailable",
            "threshold": DEFAULT_FAILURE_THRESHOLD,
            "accuracy": None,
            "recall": None,
            "precision": None,
            "f1": None,
            "f2": None,
            "roc_auc": None,
            "model_comparison": [],
            "feature_importance": [],
        }

    test_data = test_df.dropna(subset=AI4I_FEATURES + ["machine_failure"])
    y_true = test_data["machine_failure"].astype(int)
    probabilities = artifact["model"].predict_proba(test_data[AI4I_FEATURES])[:, 1]
    threshold = artifact.get("threshold", DEFAULT_FAILURE_THRESHOLD)
    y_pred = (probabilities >= threshold).astype(int)
    true_positive = int(((y_true == 1) & (y_pred == 1)).sum())
    false_positive = int(((y_true == 0) & (y_pred == 1)).sum())
    false_negative = int(((y_true == 1) & (y_pred == 0)).sum())
    accuracy = float((y_true == y_pred).mean())
    recall = true_positive / max(true_positive + false_negative, 1)
    precision = true_positive / max(true_positive + false_positive, 1)

    return {
        "model_name": artifact.get("model_name", MODEL_NAME),
        "target": MODEL_TARGET,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "split": "70% train / 30% test",
        "threshold": threshold,
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "f1": None,
        "f2": None,
        "roc_auc": None,
        "model_comparison": [],
        "feature_importance": [],
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


def _ai4i_rules_explanation(selected_row):
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

    return explanations


def _threshold_explanation(selected_row):
    reasons = []
    for col, thresh in FAULT_THRESHOLDS.items():
        if selected_row[col] > thresh:
            reasons.append(f"{col.replace('_', ' ').title()} above {thresh}")
    return ", ".join(reasons) if reasons else "No critical readings detected"


def _feature_label(feature):
    return {
        "air_temperature_k": "air temperature",
        "process_temperature_k": "process temperature",
        "rotational_speed_rpm": "rotational speed",
        "torque_nm": "torque",
        "tool_wear_min": "tool wear",
    }.get(feature, feature.replace("_", " "))


def _format_feature_value(feature, value):
    if feature.endswith("_k"):
        return f"{value:.1f} K"
    if feature == "rotational_speed_rpm":
        return f"{value:.0f} rpm"
    if feature == "torque_nm":
        return f"{value:.1f} Nm"
    if feature == "tool_wear_min":
        return f"{value:.0f} min"
    return f"{value:.2f}"


def _local_feature_contributions(model, selected_row, features, baseline_values, probability):
    row_frame = selected_row[features].astype(float).to_frame().T
    contributions = []

    for feature in features:
        baseline_frame = row_frame.copy()
        baseline_value = float(baseline_values.get(feature, row_frame.iloc[0][feature]))
        baseline_frame.loc[:, feature] = baseline_value
        baseline_probability = float(model.predict_proba(baseline_frame)[0, 1])
        contribution = probability - baseline_probability
        contributions.append({
            "feature": feature,
            "label": _feature_label(feature),
            "value": float(row_frame.iloc[0][feature]),
            "baseline": baseline_value,
            "contribution": float(contribution),
            "direction": "increased" if contribution >= 0 else "reduced",
        })

    return sorted(contributions, key=lambda item: abs(item["contribution"]), reverse=True)


def _ai4i_explanation(selected_row, artifact, probability):
    model = artifact["model"]
    features = artifact.get("features", AI4I_FEATURES)
    baselines = artifact.get("feature_medians", {})
    contributions = _local_feature_contributions(model, selected_row, features, baselines, probability)
    positive_drivers = [item for item in contributions if item["contribution"] > 0.005]
    rules = _ai4i_rules_explanation(selected_row)

    if positive_drivers:
        driver_text = "; ".join(
            f"{item['label']} at {_format_feature_value(item['feature'], item['value'])} "
            f"raised probability by {item['contribution'] * 100:.1f} pp"
            for item in positive_drivers[:3]
        )
    else:
        driver_text = "No feature strongly raised the failure probability against the training baseline"

    if rules:
        driver_text += ". Rule checks also flagged: " + ", ".join(rules)

    return driver_text, contributions


def predict_fault(selected_row: pd.Series, history_df: pd.DataFrame):
    """Predict failure probability and estimated downtime for one reading."""
    using_ai4i = all(feature in selected_row for feature in AI4I_FEATURES) and not load_ai4i_dataset().empty
    threshold = DEFAULT_FAILURE_THRESHOLD
    contributions = []
    model_name = MODEL_NAME

    if using_ai4i:
        artifact = _runtime_ai4i_artifact()
        if artifact is None:
            return {
                "probability": 0.0,
                "downtime_hours": 0.5,
                "explanation": "Insufficient data to train predictive model",
                "threshold": threshold,
                "predicted_failure": False,
                "feature_contributions": contributions,
                "model_name": model_name,
            }
        model = artifact["model"]
        features = artifact.get("features", AI4I_FEATURES)
        threshold = float(artifact.get("threshold", DEFAULT_FAILURE_THRESHOLD))
        model_name = artifact.get("model_name", MODEL_NAME)
        feature_frame = selected_row[features].astype(float).to_frame().T
        probability = float(model.predict_proba(feature_frame)[0, 1])
        explanation, contributions = _ai4i_explanation(selected_row, artifact, probability)
    else:
        model = _train_threshold_fallback(history_df)
        if model is None:
            return {
                "probability": 0.0,
                "downtime_hours": 0.5,
                "explanation": "Insufficient data to train predictive model",
                "threshold": threshold,
                "predicted_failure": False,
                "feature_contributions": contributions,
                "model_name": "Runtime Threshold Model",
            }
        features = DERIVED_FEATURES
        feature_frame = selected_row[features].astype(float).to_frame().T
        probability = float(model.predict_proba(feature_frame)[0, 1])
        explanation = _threshold_explanation(selected_row)
        model_name = "Runtime Threshold Model"

    if np.isnan(probability):
        probability = 0.0
        explanation = "Prediction unavailable for this reading"

    downtime = max(0.5, probability * 6)
    return {
        "probability": probability,
        "downtime_hours": downtime,
        "explanation": explanation,
        "threshold": threshold,
        "predicted_failure": probability >= threshold,
        "feature_contributions": contributions,
        "model_name": model_name,
    }
