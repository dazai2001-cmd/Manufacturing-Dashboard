import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# Thresholds used to label data for the simple predictive model
FAULT_THRESHOLDS = {
    "oil_temp": 110,
    "vibration": 10,
    "bearing_temp": 115,
}

def _prepare_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with a derived 'fault' label column."""
    labeled = df.copy()
    labeled["fault"] = (
        (labeled["oil_temp"] > FAULT_THRESHOLDS["oil_temp"]) |
        (labeled["vibration"] > FAULT_THRESHOLDS["vibration"]) |
        (labeled["bearing_temp"] > FAULT_THRESHOLDS["bearing_temp"])
    ).astype(int)
    return labeled

def _train_model(df: pd.DataFrame):
    """Train a simple logistic regression model if possible."""
    data = _prepare_training_data(df)
    X = data[["oil_temp", "hydraulic_temp", "bearing_temp", "vibration"]]
    y = data["fault"]
    if y.nunique() < 2:
        return None
    try:
        model = LogisticRegression()
        model.fit(X, y)
        return model
    except Exception:
        return None

def predict_fault(selected_row: pd.Series, history_df: pd.DataFrame):
    """Predict fault probability and downtime for the selected machine.

    Returns (probability, downtime_hours, explanation).
    """
    model = _train_model(history_df)
    features = selected_row[["oil_temp", "hydraulic_temp", "bearing_temp", "vibration"]].values.reshape(1, -1)
    # Basic explanation based on threshold breaches
    reasons = []
    for col, thresh in FAULT_THRESHOLDS.items():
        if selected_row[col] > thresh:
            reasons.append(f"{col.replace('_', ' ').title()} above {thresh}")
    if model is None:
        probability = 0.0
    else:
        probability = float(model.predict_proba(features)[0, 1])
    downtime = max(0.5, probability * 5)
    if reasons:
        explanation = ", ".join(reasons)
    elif model is None:
        explanation = "Insufficient data to train predictive model"
    else:
        explanation = "No critical readings detected"
    return probability, downtime, explanation
