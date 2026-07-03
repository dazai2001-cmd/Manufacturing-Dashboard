import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_simulator import DATASET_PATH, get_live_data, load_ai4i_dataset
from fault_predictor import AI4I_FEATURES, MODEL_NAME, MODEL_TARGET, predict_fault
from maintenance_analytics import calculate_maintenance_insights


REFRESH_INTERVAL_SECONDS = 10

st.set_page_config("Manufacturing Dashboard", layout="wide")

st.markdown("""
    <style>
    body, .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: white;
        font-family: 'Segoe UI', sans-serif;
    }
    h1 {
        font-size: 36px !important;
        font-weight: bold !important;
        color: #ffffff !important;
        margin-bottom: 0.5rem !important;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 1.2rem;
        margin: 0.5rem 0;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .metric-value { font-size: 28px; font-weight: 600; color: #ffffff; }
    .metric-label { font-size: 14px; color: #bbbbbb; }
    .gauge-title {
        text-align: center;
        font-size: 16px;
        color: white;
        margin-top: -12px;
    }
    .alert-box {
        background-color: #ff4d4d;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    }
    .analytics-card {
        background: rgba(255, 255, 255, 0.07);
        border: 1px solid rgba(255,255,255,0.13);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0 1rem 0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.28);
    }
    .analytics-title {
        color: #ffffff;
        font-weight: 700;
        font-size: 16px;
        margin-bottom: 0.35rem;
    }
    .analytics-copy {
        color: #e8e5ff;
        font-size: 14px;
        line-height: 1.45;
    }
    .stPlotlyChart {
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    }
    table {
        background-color: rgba(255,255,255,0.05) !important;
        color: white;
    }
    .block-container { padding: 2rem; }
    </style>
""", unsafe_allow_html=True)


def create_gauge(value, green_range, orange_range, red_range, unit="deg C"):
    if value <= green_range[1]:
        bar_color = "#d0aaff"
    elif value <= orange_range[1]:
        bar_color = "#b07bfa"
    else:
        bar_color = "#9446ff"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": f" {unit}", "font": {"size": 24, "color": "white"}},
        domain={"x": [0, 1], "y": [0, 1]},
        gauge={
            "axis": {"range": [None, red_range[1]], "tickcolor": "#888", "tickwidth": 1.5},
            "bar": {"color": bar_color, "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0.1)",
            "borderwidth": 0,
            "steps": [],
            "threshold": {"line": {"color": "white", "width": 4}, "thickness": 0.75, "value": red_range[0]},
        },
    ))
    fig.update_traces(gauge_shape="angular")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        height=250,
        margin=dict(t=20, b=10, l=0, r=0),
    )
    return fig


if "history_df" not in st.session_state:
    st.session_state.history_df = pd.DataFrame()
if "refresh_live_data" not in st.session_state:
    st.session_state.refresh_live_data = True
if "machine_health_state" not in st.session_state:
    st.session_state.machine_health_state = {}

if st.session_state.refresh_live_data or "live_df" not in st.session_state:
    st.session_state.live_df = get_live_data(st.session_state.machine_health_state)
    st.session_state.history_df = pd.concat(
        [st.session_state.history_df, st.session_state.live_df],
        ignore_index=True,
    )
    st.session_state.refresh_live_data = False

live_df = st.session_state.live_df.copy()
history_df = st.session_state.history_df.copy()
maintenance_insights = calculate_maintenance_insights(live_df, history_df)

using_ai4i = {"machine_failure", "tool_wear_min", "torque_nm", "process_temp_c"}.issubset(history_df.columns)
avg_predicted_risk = round(maintenance_insights["maintenance_risk_pct"].mean(), 1)

st.title("Manufacturing Operations Dashboard")
if not load_ai4i_dataset().empty:
    st.caption(f"Data source: UCI AI4I 2020 Predictive Maintenance replay ({DATASET_PATH.name})")
st.markdown("### Overall Manufacturing KPIs")

kpi_cols = st.columns(5)
if using_ai4i:
    records_replayed = len(history_df)
    observed_failure_rate = history_df["machine_failure"].mean() * 100
    avg_tool_wear = history_df["tool_wear_min"].mean()
    avg_torque = history_df["torque_nm"].mean()
    avg_process_temp = history_df["process_temp_c"].mean()
    with kpi_cols[0]:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{records_replayed}</div><div class='metric-label'>AI4I Records Replayed</div></div>", unsafe_allow_html=True)
    with kpi_cols[1]:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{observed_failure_rate:.1f}%</div><div class='metric-label'>Observed Failure Rate</div></div>", unsafe_allow_html=True)
    with kpi_cols[2]:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{avg_tool_wear:.0f} min</div><div class='metric-label'>Avg Tool Wear</div></div>", unsafe_allow_html=True)
    with kpi_cols[3]:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{avg_torque:.1f} Nm</div><div class='metric-label'>Avg Torque</div></div>", unsafe_allow_html=True)
    with kpi_cols[4]:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{avg_predicted_risk:.1f}%</div><div class='metric-label'>Avg Predicted Risk</div></div>", unsafe_allow_html=True)
else:
    total_units = history_df["units_produced"].sum()
    avg_eff = round(history_df["production_efficiency"].mean(), 2)
    avg_defect = round(history_df["defect_rate"].mean(), 2)
    total_energy = round(history_df["energy_usage"].sum(), 2)
    total_cost = round(history_df["energy_cost"].sum(), 2)
    with kpi_cols[0]:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{total_units}</div><div class='metric-label'>Total Units Produced</div></div>", unsafe_allow_html=True)
    with kpi_cols[1]:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{avg_eff:.1f}%</div><div class='metric-label'>Avg Efficiency</div></div>", unsafe_allow_html=True)
    with kpi_cols[2]:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{avg_defect:.2f}%</div><div class='metric-label'>Avg Defect Rate</div></div>", unsafe_allow_html=True)
    with kpi_cols[3]:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{total_energy:.1f} kWh</div><div class='metric-label'>Total Energy Used</div></div>", unsafe_allow_html=True)
    with kpi_cols[4]:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>${total_cost:.2f}</div><div class='metric-label'>Total Energy Cost</div></div>", unsafe_allow_html=True)

machine_options = live_df["machine_type"].unique().tolist()
selected_machine = st.selectbox("Select Machine for Gauges & Alerts", machine_options)
selected_row = live_df[live_df["machine_type"] == selected_machine].iloc[0]
selected_machine_df = history_df[history_df["machine_type"] == selected_machine]
selected_insight = maintenance_insights[maintenance_insights["machine_type"] == selected_machine].iloc[0]

st.markdown("### Selected Machine Stats (Total & Avg)")

machine_kpi_cols = st.columns(5)
if using_ai4i:
    selected_risk = float(selected_insight["maintenance_risk_pct"])
    selected_failure_rate = selected_machine_df["machine_failure"].mean() * 100
    latest_failure_type = str(selected_row.get("failure_type", "None"))
    with machine_kpi_cols[0]:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{len(selected_machine_df)}</div><div class='metric-label'>Records Replayed</div></div>", unsafe_allow_html=True)
    with machine_kpi_cols[1]:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{selected_failure_rate:.1f}%</div><div class='metric-label'>Observed Failure Rate</div></div>", unsafe_allow_html=True)
    with machine_kpi_cols[2]:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{selected_machine_df['tool_wear_min'].mean():.0f} min</div><div class='metric-label'>Avg Tool Wear</div></div>", unsafe_allow_html=True)
    with machine_kpi_cols[3]:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{selected_row['power_kw']:.2f} kW</div><div class='metric-label'>Latest Power Load</div></div>", unsafe_allow_html=True)
    with machine_kpi_cols[4]:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{selected_risk:.0f}%</div><div class='metric-label'>Predicted Risk</div></div>", unsafe_allow_html=True)
    if latest_failure_type != "None":
        st.markdown(f"<div class='alert-box'>{selected_machine} replay row is labeled: {latest_failure_type}</div>", unsafe_allow_html=True)
else:
    with machine_kpi_cols[0]:
        total_units_m = selected_machine_df["units_produced"].sum()
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{total_units_m}</div><div class='metric-label'>Total Units Produced</div></div>", unsafe_allow_html=True)
    with machine_kpi_cols[1]:
        avg_eff_m = selected_machine_df["production_efficiency"].mean()
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{avg_eff_m:.1f}%</div><div class='metric-label'>Avg Efficiency</div></div>", unsafe_allow_html=True)
    with machine_kpi_cols[2]:
        avg_defect_m = selected_machine_df["defect_rate"].mean()
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{avg_defect_m:.2f}%</div><div class='metric-label'>Avg Defect Rate</div></div>", unsafe_allow_html=True)
    with machine_kpi_cols[3]:
        total_energy_m = selected_machine_df["energy_usage"].sum()
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{total_energy_m:.2f} kWh</div><div class='metric-label'>Total Energy Used</div></div>", unsafe_allow_html=True)
    with machine_kpi_cols[4]:
        total_cost_m = selected_machine_df["energy_cost"].sum()
        st.markdown(f"<div class='metric-card'><div class='metric-value'>${total_cost_m:.2f}</div><div class='metric-label'>Total Energy Cost</div></div>", unsafe_allow_html=True)

probability, downtime_hours, explanation = predict_fault(selected_row, history_df)
st.markdown("### AI Fault Prediction")
ai_cols = st.columns(3)
with ai_cols[0]:
    st.markdown(f"<div class='metric-card'><div class='metric-value'>{probability * 100:.1f}%</div><div class='metric-label'>Failure Probability</div></div>", unsafe_allow_html=True)
with ai_cols[1]:
    st.markdown(f"<div class='metric-card'><div class='metric-value'>{downtime_hours:.1f} hrs</div><div class='metric-label'>Estimated Downtime</div></div>", unsafe_allow_html=True)
with ai_cols[2]:
    st.markdown(f"<div class='metric-card'><div class='metric-value'>{MODEL_NAME}</div><div class='metric-label'>{MODEL_TARGET}</div></div>", unsafe_allow_html=True)
st.caption("Model features: " + ", ".join(AI4I_FEATURES if using_ai4i else ["oil_temp", "hydraulic_temp", "bearing_temp", "vibration"]))
st.markdown(f"**Explanation:** {explanation}")

st.markdown("### Predictive Prescriptive Maintenance Analytics")

analytics_cols = st.columns(4)
with analytics_cols[0]:
    st.markdown(f"<div class='metric-card'><div class='metric-value'>{selected_insight['maintenance_risk_pct']}%</div><div class='metric-label'>Predicted Maintenance Risk</div></div>", unsafe_allow_html=True)
with analytics_cols[1]:
    st.markdown(f"<div class='metric-card'><div class='metric-value'>{selected_insight['risk_band']}</div><div class='metric-label'>Risk Band</div></div>", unsafe_allow_html=True)
with analytics_cols[2]:
    st.markdown(f"<div class='metric-card'><div class='metric-value'>{selected_insight['confidence_pct']}%</div><div class='metric-label'>Prediction Confidence</div></div>", unsafe_allow_html=True)
with analytics_cols[3]:
    st.markdown(f"<div class='metric-card'><div class='metric-value'>{selected_insight['time_to_service']}</div><div class='metric-label'>Service Window</div></div>", unsafe_allow_html=True)

st.markdown(
    f"""
    <div class='analytics-card'>
        <div class='analytics-title'>Likely reason: {selected_insight['predicted_reason']}</div>
        <div class='analytics-copy'><strong>Forecast horizon:</strong> {selected_insight['forecast_horizon']}</div>
        <div class='analytics-copy'><strong>Evidence:</strong> {selected_insight['evidence']}</div>
        <div class='analytics-copy'><strong>Recommended action:</strong> {selected_insight['prescribed_action']}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

analytics_row = st.columns([1, 1])
with analytics_row[0]:
    maintenance_chart_df = (
        maintenance_insights.set_index("machine_type")
        .loc[machine_options]
        .reset_index()
    )
    risk_chart = px.bar(
        maintenance_chart_df,
        x="machine_type",
        y="maintenance_risk_pct",
        color="risk_band",
        color_discrete_map={
            "Normal": "#69d2a3",
            "Watch": "#f7d154",
            "High": "#ff9f43",
            "Critical": "#ff4d4d",
        },
        title="Predicted Maintenance Risk by Machine",
        labels={"maintenance_risk_pct": "Risk (%)", "machine_type": "Machine"},
        category_orders={"machine_type": machine_options},
    )
    risk_chart.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        title_font=dict(size=16),
        yaxis=dict(range=[0, 100]),
        margin=dict(t=40, b=20),
    )
    st.plotly_chart(risk_chart, use_container_width=True)

with analytics_row[1]:
    st.dataframe(
        maintenance_insights[[
            "machine_type",
            "maintenance_risk_pct",
            "risk_band",
            "predicted_reason",
            "forecast_horizon",
            "time_to_service",
            "prescribed_action",
        ]].rename(columns={
            "machine_type": "Machine",
            "maintenance_risk_pct": "Risk %",
            "risk_band": "Band",
            "predicted_reason": "Predicted Reason",
            "forecast_horizon": "Forecast Horizon",
            "time_to_service": "Service Window",
            "prescribed_action": "Recommended Action",
        }),
        hide_index=True,
        use_container_width=True,
    )

alert_conditions = []
if selected_row["oil_temp"] > 110:
    alert_conditions.append(f"{selected_machine} - Oil temperature is critically high.")
if selected_row["vibration"] > 10:
    alert_conditions.append(f"{selected_machine} - Abnormal vibration levels detected.")
if selected_row["bearing_temp"] > 115:
    alert_conditions.append(f"{selected_machine} - Bearing temperature is too high.")
if selected_insight["maintenance_risk_pct"] >= 50:
    alert_conditions.append(
        f"Prescriptive analytics recommends maintenance for {selected_machine}: {selected_insight['predicted_reason']}"
    )

sound_enabled = st.checkbox("Enable Sound Alerts", value=False, help="Play audio when alerts are triggered")

if alert_conditions:
    if sound_enabled:
        st.audio("alert.mp3", format="audio/mp3", autoplay=True)
    for msg in alert_conditions:
        st.markdown(f"<div class='alert-box'>{msg}</div>", unsafe_allow_html=True)

gauge_cols_1 = st.columns(2)
gauge_cols_2 = st.columns(2)
with gauge_cols_1[0]:
    st.plotly_chart(create_gauge(selected_row["oil_temp"], (40, 70), (70, 85), (85, 120)), use_container_width=True)
    st.markdown("<div class='gauge-title'>Oil Temp</div>", unsafe_allow_html=True)
with gauge_cols_1[1]:
    st.plotly_chart(create_gauge(selected_row["hydraulic_temp"], (30, 60), (60, 75), (75, 120)), use_container_width=True)
    st.markdown("<div class='gauge-title'>Hydraulic Temp</div>", unsafe_allow_html=True)
with gauge_cols_2[0]:
    st.plotly_chart(create_gauge(selected_row["vibration"], (0.0, 4.0), (4.0, 7.0), (7.0, 15.0), unit="mm/s"), use_container_width=True)
    st.markdown("<div class='gauge-title'>Vibration</div>", unsafe_allow_html=True)
with gauge_cols_2[1]:
    st.plotly_chart(create_gauge(selected_row["bearing_temp"], (40, 80), (80, 95), (95, 120)), use_container_width=True)
    st.markdown("<div class='gauge-title'>Bearing Temp</div>", unsafe_allow_html=True)

chart_row = st.columns(2)
with chart_row[0]:
    today = pd.Timestamp.now().normalize()
    today_df = history_df[history_df["timestamp"] >= today]
    daily_prod = today_df.groupby("machine_type")["units_produced"].sum().reset_index()
    fig_bar = px.bar(daily_prod, x="machine_type", y="units_produced", title="Daily Production by Machine Type")
    fig_bar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        title_font=dict(size=16),
        margin=dict(t=40, b=20),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with chart_row[1]:
    st.markdown(f"### Efficiency Trend for {selected_machine}")
    history_df["timestamp"] = pd.to_datetime(history_df["timestamp"])
    machine_df = history_df[history_df["machine_type"] == selected_machine].sort_values("timestamp")

    if not machine_df.empty:
        fig = px.line(
            machine_df,
            x="timestamp",
            y="production_efficiency",
            title=f"{selected_machine} Production Efficiency Over Time",
            labels={"production_efficiency": "Efficiency (%)", "timestamp": "Time"},
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            title_font=dict(size=16),
            margin=dict(t=30, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available yet for this machine.")

st.markdown(f"### Recent Data for {selected_machine}")
selected_machine_df = history_df[history_df["machine_type"] == selected_machine]
recent_data = selected_machine_df.tail(10).copy()
rounded_columns = {
    "oil_temp": 1,
    "hydraulic_temp": 1,
    "bearing_temp": 1,
    "vibration": 2,
    "production_efficiency": 2,
    "defect_rate": 2,
    "energy_usage": 2,
}
recent_data = recent_data.round(rounded_columns)
recent_data["energy_cost"] = recent_data["energy_cost"].map("${:.2f}".format)
st.dataframe(recent_data, use_container_width=True)

st.markdown(f"Refreshing in {REFRESH_INTERVAL_SECONDS} seconds...")
time.sleep(REFRESH_INTERVAL_SECONDS)
st.session_state.refresh_live_data = True
st.rerun()
