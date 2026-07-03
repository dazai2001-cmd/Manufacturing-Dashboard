import math

import pandas as pd


SENSOR_LIMITS = {
    "oil_temp": {"watch": 85, "critical": 110, "label": "oil temperature"},
    "hydraulic_temp": {"watch": 75, "critical": 95, "label": "hydraulic temperature"},
    "bearing_temp": {"watch": 95, "critical": 115, "label": "bearing temperature"},
    "vibration": {"watch": 7.0, "critical": 10.0, "label": "vibration"},
}


def _scale(value, start, end):
    if end <= start:
        return 0
    return max(0, min(1, (value - start) / (end - start)))


def _fmt(value, suffix="", precision=1):
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.{precision}f}{suffix}"


def _trend(machine_history, column, lookback=6):
    values = machine_history[column].dropna().tail(lookback)
    if len(values) < 3:
        return 0
    return float(values.iloc[-1] - values.iloc[0])


def _recent_average(machine_history, column, lookback=8):
    values = machine_history[column].dropna().tail(lookback)
    if values.empty:
        return math.nan
    return float(values.mean())


def _risk_band(score):
    if score >= 75:
        return "Critical"
    if score >= 50:
        return "High"
    if score >= 30:
        return "Watch"
    return "Normal"


def _time_to_service(score):
    if score >= 75:
        return "Now"
    if score >= 50:
        return "Within 24 hours"
    if score >= 30:
        return "This week"
    return "Routine interval"


def _top_cause(causes):
    if not causes:
        return {
            "reason": "No immediate maintenance driver detected",
            "action": "Continue normal operation and review the next scheduled inspection.",
            "priority": 0,
        }
    return max(causes, key=lambda item: item["points"])


def calculate_maintenance_insights(live_df, history_df):
    """Return explainable predictive and prescriptive maintenance insights."""
    if live_df.empty:
        return pd.DataFrame()

    history = history_df.copy()
    if "timestamp" in history:
        history["timestamp"] = pd.to_datetime(history["timestamp"], errors="coerce")
        history = history.sort_values("timestamp")

    insights = []

    for _, row in live_df.iterrows():
        machine = row["machine_type"]
        machine_history = history[history["machine_type"] == machine].copy()
        score = 0
        causes = []
        evidence = []

        oil_temp = float(row["oil_temp"])
        hydraulic_temp = float(row["hydraulic_temp"])
        bearing_temp = float(row["bearing_temp"])
        vibration = float(row["vibration"])
        efficiency = float(row["production_efficiency"])
        defect_rate = float(row["defect_rate"])
        energy_per_unit = float(row["energy_usage"]) / max(float(row["units_produced"]), 1)

        bearing_vibration_points = (
            24 * _scale(bearing_temp, 92, 118)
            + 28 * _scale(vibration, 6.5, 11.5)
            + 8 * _scale(_trend(machine_history, "bearing_temp"), 4, 18)
            + 8 * _scale(_trend(machine_history, "vibration"), 0.8, 4.0)
        )
        if bearing_vibration_points >= 10:
            causes.append({
                "reason": "Bearing wear, imbalance, or shaft misalignment",
                "action": "Inspect bearings, check alignment, and schedule vibration analysis before the next production run.",
                "points": bearing_vibration_points,
                "priority": 3,
            })
            evidence.append(
                f"Bearing temp {_fmt(bearing_temp, 'C')} and vibration {_fmt(vibration, ' mm/s', 2)} are trending toward wear thresholds."
            )

        lubrication_points = (
            28 * _scale(oil_temp, 82, 112)
            + 10 * _scale(_trend(machine_history, "oil_temp"), 4, 18)
            + 8 * _scale(vibration, 7.0, 11.0)
        )
        if lubrication_points >= 10:
            causes.append({
                "reason": "Lubrication breakdown or cooling restriction",
                "action": "Check oil level, oil quality, filters, coolant flow, and heat exchanger performance.",
                "points": lubrication_points,
                "priority": 3,
            })
            evidence.append(f"Oil temp is {_fmt(oil_temp, 'C')} with recent heat gain of {_fmt(_trend(machine_history, 'oil_temp'), 'C')}.")

        hydraulic_points = (
            22 * _scale(hydraulic_temp, 72, 98)
            + 8 * _scale(_trend(machine_history, "hydraulic_temp"), 3, 16)
        )
        if hydraulic_points >= 8:
            causes.append({
                "reason": "Hydraulic fluid overheating or pump strain",
                "action": "Inspect hydraulic fluid, pump load, reservoir cooling, and blocked return filters.",
                "points": hydraulic_points,
                "priority": 2,
            })
            evidence.append(f"Hydraulic temp is {_fmt(hydraulic_temp, 'C')} and may indicate pump or cooling stress.")

        quality_points = (
            12 * _scale(defect_rate, 1.2, 3.0)
            + 14 * _scale(85 - efficiency, 0, 40)
        )
        if quality_points >= 8:
            causes.append({
                "reason": "Tool wear or process drift affecting output quality",
                "action": "Inspect tooling, recalibrate offsets, and verify material feed or fixture setup.",
                "points": quality_points,
                "priority": 2,
            })
            evidence.append(f"Efficiency is {_fmt(efficiency, '%')} with defect rate {_fmt(defect_rate, '%', 2)}.")

        recent_energy_per_unit = _recent_average(
            machine_history.assign(
                energy_per_unit=machine_history["energy_usage"] / machine_history["units_produced"].clip(lower=1)
            ),
            "energy_per_unit",
        )
        energy_points = 10 * _scale(energy_per_unit - recent_energy_per_unit, 0.25, 1.25)
        if energy_points >= 5:
            causes.append({
                "reason": "Rising energy per unit suggests mechanical drag or inefficient load",
                "action": "Check lubrication, belt tension, spindle load, and motor current draw.",
                "points": energy_points,
                "priority": 1,
            })
            evidence.append(
                f"Energy per unit is {_fmt(energy_per_unit, ' kWh/unit', 2)} versus recent average {_fmt(recent_energy_per_unit, ' kWh/unit', 2)}."
            )

        for column, limits in SENSOR_LIMITS.items():
            value = float(row[column])
            if value >= limits["critical"]:
                score += 18
                evidence.append(f"Critical {limits['label']} threshold exceeded: {_fmt(value, precision=2)}.")
            elif value >= limits["watch"]:
                score += 8

        score += sum(cause["points"] for cause in causes)
        score = int(round(min(score, 100)))
        top_cause = _top_cause(causes)
        confidence = int(round(min(95, 48 + score * 0.45 + len(machine_history) * 1.2)))

        insights.append({
            "machine_type": machine,
            "maintenance_risk_pct": score,
            "risk_band": _risk_band(score),
            "predicted_reason": top_cause["reason"],
            "prescribed_action": top_cause["action"],
            "priority": top_cause["priority"],
            "confidence_pct": confidence,
            "time_to_service": _time_to_service(score),
            "evidence": " ".join(dict.fromkeys(evidence)) if evidence else "Current readings are within normal operating patterns.",
        })

    return pd.DataFrame(insights).sort_values(
        ["priority", "maintenance_risk_pct"], ascending=[False, False]
    )
