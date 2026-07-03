import math

import pandas as pd


SENSOR_LIMITS = {
    "oil_temp": {"watch": 85, "critical": 110, "label": "oil temperature"},
    "hydraulic_temp": {"watch": 75, "critical": 95, "label": "hydraulic temperature"},
    "bearing_temp": {"watch": 95, "critical": 115, "label": "bearing temperature"},
    "vibration": {"watch": 7.0, "critical": 10.0, "label": "vibration"},
}
FORECAST_HORIZON_READINGS = 6
HOURLY_DOWNTIME_COST = 850
PREVENTIVE_MAINTENANCE_COST = 650
CORRECTIVE_REPAIR_COST = 2400


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


def _trend_per_reading(machine_history, column, lookback=8):
    values = machine_history[column].dropna().tail(lookback)
    if len(values) < 3:
        return 0
    return float((values.iloc[-1] - values.iloc[0]) / (len(values) - 1))


def _recent_average(machine_history, column, lookback=8):
    values = machine_history[column].dropna().tail(lookback)
    if values.empty:
        return math.nan
    return float(values.mean())


def _recent_peak(machine_history, column, lookback=8):
    values = machine_history[column].dropna().tail(lookback)
    if values.empty:
        return math.nan
    return float(values.max())


def _forecast_value(machine_history, column, current_value, horizon=FORECAST_HORIZON_READINGS):
    trend = max(0, _trend_per_reading(machine_history, column))
    projected_value = current_value + trend * horizon
    recent_peak = _recent_peak(machine_history, column)
    if pd.isna(recent_peak):
        return projected_value
    # Keep recent stress visible so a one-refresh recovery does not erase risk.
    return max(projected_value, current_value * 0.7 + recent_peak * 0.3)


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


def _cost_impact(score, priority):
    failure_likelihood = min(0.95, max(0.02, score / 100))
    estimated_downtime_hours = 1.0 + failure_likelihood * 7.0 + priority * 0.75
    expected_failure_cost = (
        failure_likelihood * CORRECTIVE_REPAIR_COST
        + estimated_downtime_hours * HOURLY_DOWNTIME_COST
    )
    expected_preventive_cost = PREVENTIVE_MAINTENANCE_COST + max(priority, 1) * 125
    cost_avoided = max(0, expected_failure_cost - expected_preventive_cost)
    return {
        "failure_likelihood": failure_likelihood,
        "estimated_downtime_hours": estimated_downtime_hours,
        "expected_failure_cost": expected_failure_cost,
        "expected_preventive_cost": expected_preventive_cost,
        "estimated_cost_avoided": cost_avoided,
    }


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
        predicted_oil_temp = _forecast_value(machine_history, "oil_temp", oil_temp)
        predicted_hydraulic_temp = _forecast_value(machine_history, "hydraulic_temp", hydraulic_temp)
        predicted_bearing_temp = _forecast_value(machine_history, "bearing_temp", bearing_temp)
        predicted_vibration = _forecast_value(machine_history, "vibration", vibration)
        efficiency = float(row["production_efficiency"])
        defect_rate = float(row["defect_rate"])
        energy_per_unit = float(row["energy_usage"]) / max(float(row["units_produced"]), 1)

        bearing_vibration_points = (
            24 * _scale(predicted_bearing_temp, 92, 118)
            + 28 * _scale(predicted_vibration, 6.5, 11.5)
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
                f"Bearing temp is projected to {_fmt(predicted_bearing_temp, 'C')} and vibration to {_fmt(predicted_vibration, ' mm/s', 2)} within {FORECAST_HORIZON_READINGS} readings."
            )

        lubrication_points = (
            28 * _scale(predicted_oil_temp, 82, 112)
            + 10 * _scale(_trend(machine_history, "oil_temp"), 4, 18)
            + 8 * _scale(predicted_vibration, 7.0, 11.0)
        )
        if lubrication_points >= 10:
            causes.append({
                "reason": "Lubrication breakdown or cooling restriction",
                "action": "Check oil level, oil quality, filters, coolant flow, and heat exchanger performance.",
                "points": lubrication_points,
                "priority": 3,
            })
            evidence.append(f"Oil temp is projected to {_fmt(predicted_oil_temp, 'C')} from current {_fmt(oil_temp, 'C')}.")

        hydraulic_points = (
            22 * _scale(predicted_hydraulic_temp, 72, 98)
            + 8 * _scale(_trend(machine_history, "hydraulic_temp"), 3, 16)
        )
        if hydraulic_points >= 8:
            causes.append({
                "reason": "Hydraulic fluid overheating or pump strain",
                "action": "Inspect hydraulic fluid, pump load, reservoir cooling, and blocked return filters.",
                "points": hydraulic_points,
                "priority": 2,
            })
            evidence.append(f"Hydraulic temp is projected to {_fmt(predicted_hydraulic_temp, 'C')} and may indicate pump or cooling stress.")

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

        if {"tool_wear_min", "torque_nm", "rotational_speed_rpm", "air_temperature_k", "process_temperature_k"}.issubset(row.index):
            tool_wear = float(row["tool_wear_min"])
            torque = float(row["torque_nm"])
            rpm = float(row["rotational_speed_rpm"])
            air_temp = float(row["air_temperature_k"])
            process_temp = float(row["process_temperature_k"])
            power_kw = torque * rpm / 9550
            temp_gap = process_temp - air_temp

            tool_wear_points = 22 * _scale(tool_wear, 150, 240)
            if tool_wear_points >= 7:
                causes.append({
                    "reason": "Tool wear is approaching failure range",
                    "action": "Plan tool replacement and verify cutting parameters before the next batch.",
                    "points": tool_wear_points,
                    "priority": 3,
                })
                evidence.append(f"AI4I tool wear is {_fmt(tool_wear, ' min', 0)}.")

            heat_points = 18 * _scale(temp_gap, 8.5, 12.0)
            if heat_points >= 7:
                causes.append({
                    "reason": "Heat dissipation stress",
                    "action": "Inspect cooling, airflow, and process temperature control.",
                    "points": heat_points,
                    "priority": 2,
                })
                evidence.append(f"Process temperature is {_fmt(temp_gap, ' K')} above air temperature.")

            high_power_points = 16 * _scale(power_kw, 8.5, 10.5)
            low_power_points = 16 * _scale(3.5 - power_kw, 0, 1.5)
            power_points = max(high_power_points, low_power_points)
            if power_points >= 7:
                causes.append({
                    "reason": "Power load is outside the normal operating band",
                    "action": "Check torque, spindle speed, motor load, and feed settings.",
                    "points": power_points,
                    "priority": 2,
                })
                evidence.append(f"Estimated AI4I power load is {_fmt(power_kw, ' kW', 2)}.")

            overstrain_points = 20 * _scale(torque * tool_wear, 8500, 12000)
            if overstrain_points >= 7:
                causes.append({
                    "reason": "Overstrain risk from torque and accumulated tool wear",
                    "action": "Reduce load or replace the tool before running another high-torque job.",
                    "points": overstrain_points,
                    "priority": 3,
                })
                evidence.append(f"Torque-wear product is {_fmt(torque * tool_wear, precision=0)}.")

        for column, limits in SENSOR_LIMITS.items():
            predicted_values = {
                "oil_temp": predicted_oil_temp,
                "hydraulic_temp": predicted_hydraulic_temp,
                "bearing_temp": predicted_bearing_temp,
                "vibration": predicted_vibration,
            }
            value = predicted_values[column]
            if value >= limits["critical"]:
                score += 18
                evidence.append(f"Projected {limits['label']} reaches critical range: {_fmt(value, precision=2)}.")
            elif value >= limits["watch"]:
                score += 8

        score += sum(cause["points"] for cause in causes)
        score = int(round(min(score, 100)))
        if score < 30:
            causes = []
            evidence = []
        top_cause = _top_cause(causes)
        confidence = int(round(min(95, 48 + score * 0.45 + len(machine_history) * 1.2)))
        cost = _cost_impact(score, top_cause["priority"])

        insights.append({
            "machine_type": machine,
            "maintenance_risk_pct": score,
            "risk_band": _risk_band(score),
            "predicted_reason": top_cause["reason"],
            "prescribed_action": top_cause["action"],
            "priority": top_cause["priority"],
            "confidence_pct": confidence,
            "time_to_service": _time_to_service(score),
            "forecast_horizon": f"Next {FORECAST_HORIZON_READINGS} readings",
            "projected_oil_temp": predicted_oil_temp,
            "projected_hydraulic_temp": predicted_hydraulic_temp,
            "projected_bearing_temp": predicted_bearing_temp,
            "projected_vibration": predicted_vibration,
            "estimated_downtime_hours": round(cost["estimated_downtime_hours"], 1),
            "expected_failure_cost": int(round(cost["expected_failure_cost"])),
            "expected_preventive_cost": int(round(cost["expected_preventive_cost"])),
            "estimated_cost_avoided": int(round(cost["estimated_cost_avoided"])),
            "evidence": " ".join(dict.fromkeys(evidence)) if evidence else "Forecasted readings remain within normal operating patterns.",
        })

    return pd.DataFrame(insights).sort_values(
        ["priority", "maintenance_risk_pct"], ascending=[False, False]
    )
