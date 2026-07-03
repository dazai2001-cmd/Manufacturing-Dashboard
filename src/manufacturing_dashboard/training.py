import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from manufacturing_dashboard.data import (
    SPLIT_RANDOM_STATE,
    TEST_FRACTION,
    TRAIN_FRACTION,
    load_ai4i_split,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODEL_ARTIFACT_PATH = MODELS_DIR / "failure_model.joblib"
METRICS_REPORT_PATH = REPORTS_DIR / "model_metrics.json"
MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"
MODEL_NAME = "Random Forest"
MODEL_TARGET = "Target: Machine failure"
AI4I_FEATURES = [
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
]


def _candidate_models():
    return {
        "logistic_regression": {
            "name": "Logistic Regression",
            "model": Pipeline([
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=SPLIT_RANDOM_STATE,
                    ),
                ),
            ]),
        },
        "random_forest": {
            "name": "Random Forest",
            "model": RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=5,
                class_weight="balanced_subsample",
                random_state=SPLIT_RANDOM_STATE,
            ),
        },
        "gradient_boosting": {
            "name": "Histogram Gradient Boosting",
            "model": HistGradientBoostingClassifier(
                max_iter=250,
                learning_rate=0.05,
                max_leaf_nodes=21,
                l2_regularization=0.05,
                random_state=SPLIT_RANDOM_STATE,
            ),
            "use_sample_weight": True,
        },
    }


def _clean_training_frame(df):
    required = AI4I_FEATURES + ["machine_failure"]
    return df.dropna(subset=required).copy()


def _fit_candidate(candidate, x_train, y_train):
    model = candidate["model"]
    if candidate.get("use_sample_weight"):
        sample_weight = compute_sample_weight("balanced", y_train)
        model.fit(x_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(x_train, y_train)
    return model


def _predict_probability(model, x_frame):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_frame)[:, 1]
    decision = model.decision_function(x_frame)
    return 1 / (1 + np.exp(-decision))


def _metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 3),
        "accuracy": round(float(accuracy_score(y_true, predictions)), 4),
        "precision": round(float(precision_score(y_true, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, predictions, zero_division=0)), 4),
        "f2": round(float(_f2_score(y_true, predictions)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 4),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def _f2_score(y_true, predictions):
    precision = precision_score(y_true, predictions, zero_division=0)
    recall = recall_score(y_true, predictions, zero_division=0)
    beta_squared = 4
    denominator = beta_squared * precision + recall
    if denominator == 0:
        return 0.0
    return (1 + beta_squared) * precision * recall / denominator


def _choose_threshold(y_true, probabilities):
    best = {"threshold": 0.5, "f2": -1, "recall": -1, "precision": -1}
    for threshold in np.linspace(0.05, 0.95, 181):
        predictions = (probabilities >= threshold).astype(int)
        precision = precision_score(y_true, predictions, zero_division=0)
        recall = recall_score(y_true, predictions, zero_division=0)
        f2 = _f2_score(y_true, predictions)
        candidate = {
            "threshold": float(threshold),
            "f2": float(f2),
            "recall": float(recall),
            "precision": float(precision),
        }
        if (candidate["f2"], candidate["recall"], candidate["precision"]) > (
            best["f2"],
            best["recall"],
            best["precision"],
        ):
            best = candidate
    return best["threshold"]


def _feature_importance(model):
    estimator = model
    if isinstance(model, Pipeline):
        estimator = model.named_steps.get("classifier", model)
    importances = getattr(estimator, "feature_importances_", None)
    coefficients = getattr(estimator, "coef_", None)
    if importances is not None:
        values = importances
    elif coefficients is not None:
        values = np.abs(coefficients[0])
    else:
        return []

    total = float(np.sum(values)) or 1.0
    return [
        {
            "feature": feature,
            "importance": round(float(value / total), 4),
        }
        for feature, value in sorted(
            zip(AI4I_FEATURES, values),
            key=lambda item: item[1],
            reverse=True,
        )
    ]


def train_and_save_artifacts():
    train_df = _clean_training_frame(load_ai4i_split("train"))
    test_df = _clean_training_frame(load_ai4i_split("test"))
    if train_df.empty or test_df.empty:
        raise RuntimeError("AI4I train/test data is unavailable.")

    train_base, validation_df = train_test_split(
        train_df,
        test_size=0.20,
        random_state=SPLIT_RANDOM_STATE,
        stratify=train_df["machine_failure"],
    )

    x_train = train_base[AI4I_FEATURES]
    y_train = train_base["machine_failure"].astype(int)
    x_validation = validation_df[AI4I_FEATURES]
    y_validation = validation_df["machine_failure"].astype(int)
    x_test = test_df[AI4I_FEATURES]
    y_test = test_df["machine_failure"].astype(int)

    comparison_rows = []
    trained_candidates = {}
    for key, candidate in _candidate_models().items():
        model = _fit_candidate(candidate, x_train, y_train)
        validation_probabilities = _predict_probability(model, x_validation)
        threshold = _choose_threshold(y_validation, validation_probabilities)
        validation_metrics = _metrics(y_validation, validation_probabilities, threshold)
        trained_candidates[key] = {
            "key": key,
            "name": candidate["name"],
            "model": model,
            "threshold": threshold,
            "validation_metrics": validation_metrics,
        }
        comparison_rows.append({
            "model_key": key,
            "model_name": candidate["name"],
            **validation_metrics,
        })

    best_candidate = max(
        trained_candidates.values(),
        key=lambda item: (
            item["validation_metrics"]["f2"],
            item["validation_metrics"]["recall"],
            item["validation_metrics"]["precision"],
        ),
    )

    final_candidate = _candidate_models()[best_candidate["key"]]
    final_model = _fit_candidate(
        final_candidate,
        train_df[AI4I_FEATURES],
        train_df["machine_failure"].astype(int),
    )
    test_probabilities = _predict_probability(final_model, x_test)
    test_metrics = _metrics(y_test, test_probabilities, best_candidate["threshold"])

    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    feature_medians = train_df[AI4I_FEATURES].median().to_dict()
    artifact = {
        "model": final_model,
        "model_key": best_candidate["key"],
        "model_name": best_candidate["name"],
        "target": MODEL_TARGET,
        "features": AI4I_FEATURES,
        "threshold": float(best_candidate["threshold"]),
        "feature_medians": {key: float(value) for key, value in feature_medians.items()},
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    joblib.dump(artifact, MODEL_ARTIFACT_PATH)

    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["f2", "recall", "precision"],
        ascending=[False, False, False],
    )
    comparison.to_csv(MODEL_COMPARISON_PATH, index=False)

    report = {
        "model_name": best_candidate["name"],
        "model_key": best_candidate["key"],
        "target": MODEL_TARGET,
        "features": AI4I_FEATURES,
        "chosen_threshold": round(float(best_candidate["threshold"]), 3),
        "threshold_strategy": "Selected on validation data to maximize F2 score, weighting recall higher than precision.",
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(validation_df)),
        "test_rows": int(len(test_df)),
        "split": f"{int(TRAIN_FRACTION * 100)}% train / {int(TEST_FRACTION * 100)}% held-out test",
        "random_state": SPLIT_RANDOM_STATE,
        "class_balance": {
            "train_failure_rate": round(float(train_df["machine_failure"].mean()), 4),
            "test_failure_rate": round(float(test_df["machine_failure"].mean()), 4),
        },
        "validation_metrics": best_candidate["validation_metrics"],
        "test_metrics": test_metrics,
        "model_comparison": comparison.drop(columns=["threshold"]).to_dict(orient="records"),
        "feature_importance": _feature_importance(final_model),
        "artifact_path": str(MODEL_ARTIFACT_PATH.relative_to(PROJECT_ROOT)),
        "comparison_path": str(MODEL_COMPARISON_PATH.relative_to(PROJECT_ROOT)),
        "trained_at_utc": artifact["trained_at_utc"],
    }
    METRICS_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def load_saved_artifact():
    if MODEL_ARTIFACT_PATH.exists():
        return joblib.load(MODEL_ARTIFACT_PATH)
    return None


def load_metrics_report():
    if METRICS_REPORT_PATH.exists():
        return json.loads(METRICS_REPORT_PATH.read_text(encoding="utf-8"))
    return None
