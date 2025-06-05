
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from data_simulator import get_live_data
from fault_predictor import predict_fault
import time

# --- Page Config ---
st.set_page_config("Manufacturing Dashboard", layout="wide")

# --- Custom Glassmorphic Styling ---
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
        border-radius: 16px;
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
    .stPlotlyChart {
        border-radius: 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    }
    table {
        background-color: rgba(255,255,255,0.05) !important;
        color: white;
    }
    .block-container { padding: 2rem; }
    </style>
""", unsafe_allow_html=True)

# --- Gauge Function ---
def create_gauge(value, green_range, orange_range, red_range, unit='°C'):
    if value <= green_range[1]:
        bar_color = "#d0aaff"
    elif value <= orange_range[1]:
        bar_color = "#b07bfa"
    else:
        bar_color = "#9446ff"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={'suffix': f" {unit}", 'font': {'size': 24, 'color': "white"}},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [None, red_range[1]], 'tickcolor': "#888", 'tickwidth': 1.5},
            'bar': {'color': bar_color, 'thickness': 0.25},
            'bgcolor': "rgba(0,0,0,0.1)",
            'borderwidth': 0,
            'steps': [],
            'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': red_range[0]}
        }
    ))
    fig.update_traces(gauge_shape="angular")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"), height=250, margin=dict(t=20, b=10, l=0, r=0))
    return fig

# --- Initialize History ---
if "history_df" not in st.session_state:
    st.session_state.history_df = pd.DataFrame()

# --- Get and Append Live Data ---
live_df = get_live_data()
st.session_state.history_df = pd.concat([st.session_state.history_df, live_df], ignore_index=True)
history_df = st.session_state.history_df.copy()

# --- Aggregate Overall Metrics from History ---
total_units = history_df["units_produced"].sum()
avg_eff = round(history_df["production_efficiency"].mean(), 2)
avg_defect = round(history_df["defect_rate"].mean(), 2)
total_energy = round(history_df["energy_usage"].sum(), 2)
total_cost = round(history_df["energy_cost"].sum(), 2)

# --- Title ---
st.title("🏭 Manufacturing Operations Dashboard")

st.markdown("### 📊 Overall Manufacturing KPIs")

# --- Global KPI Cards ---
kpi_cols = st.columns(5)
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

# --- Machine Selection ---
machine_options = live_df["machine_type"].unique().tolist()
selected_machine = st.selectbox("🛠️ Select Machine for Gauges & Alerts", machine_options)
selected_row = live_df[live_df["machine_type"] == selected_machine].iloc[0]

# --- Selected Machine KPI Cards ---
selected_machine_df = history_df[history_df["machine_type"] == selected_machine]
st.markdown("### 📌 Selected Machine Stats (Total & Avg)")

machine_kpi_cols = st.columns(5)
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

# --- AI Fault Prediction ---
probability, downtime_hours, explanation = predict_fault(selected_row, history_df)
st.markdown("### 🧠 AI Fault Prediction")
ai_cols = st.columns(2)
with ai_cols[0]:
    st.markdown(f"<div class='metric-card'><div class='metric-value'>{probability*100:.1f}%</div><div class='metric-label'>Failure Probability</div></div>", unsafe_allow_html=True)
with ai_cols[1]:
    st.markdown(f"<div class='metric-card'><div class='metric-value'>{downtime_hours:.1f} hrs</div><div class='metric-label'>Est. Downtime</div></div>", unsafe_allow_html=True)
st.markdown(f"**Explanation:** {explanation}")

# --- Alerts ---
alert_conditions = []
if selected_row["oil_temp"] > 110:
    alert_conditions.append(f"⚠️ {selected_machine} - Oil temperature is critically high!")
if selected_row["vibration"] > 10:
    alert_conditions.append(f"⚠️ {selected_machine} - Abnormal vibration levels detected!")
if selected_row["bearing_temp"] > 115:
    alert_conditions.append(f"⚠️ {selected_machine} - Bearing temperature is too high!")

sound_enabled = st.checkbox("🔔 Enable Sound Alerts", value=False, help="Play audio when alerts are triggered")

if alert_conditions:
    if sound_enabled:
        st.audio("alert.mp3", format="audio/mp3", autoplay=True)
    for msg in alert_conditions:
        st.markdown(f"<div class='alert-box'>{msg}</div>", unsafe_allow_html=True)

# --- Gauges ---
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

# --- Charts ---
chart_row = st.columns(2)
with chart_row[0]:
    today = pd.Timestamp.now().normalize()
    today_df = history_df[history_df["timestamp"] >= today]
    daily_prod = today_df.groupby("machine_type")["units_produced"].sum().reset_index()
    fig_bar = px.bar(daily_prod, x="machine_type", y="units_produced", title="Daily Production by Machine Type")
    fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="white", title_font=dict(size=16), margin=dict(t=40, b=20))
    st.plotly_chart(fig_bar, use_container_width=True)

with chart_row[1]:
    st.markdown(f"### 📈 Efficiency Trend for {selected_machine}")
    history_df["timestamp"] = pd.to_datetime(history_df["timestamp"])
    machine_df = history_df[history_df["machine_type"] == selected_machine].sort_values("timestamp")

    if not machine_df.empty:
        fig = px.line(
            machine_df,
            x="timestamp",
            y="production_efficiency",
            title=f"{selected_machine} Production Efficiency Over Time",
            labels={"production_efficiency": "Efficiency (%)", "timestamp": "Time"}
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            title_font=dict(size=16),
            margin=dict(t=30, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available yet for this machine.")

# --- Table ---
st.markdown(f"### 📋 Recent Data for {selected_machine}")
selected_machine_df = history_df[history_df["machine_type"] == selected_machine]
st.dataframe(
    selected_machine_df.tail(10).style.format({
        "oil_temp": "{:.1f}",
        "hydraulic_temp": "{:.1f}",
        "bearing_temp": "{:.1f}",
        "vibration": "{:.2f}",
        "production_efficiency": "{:.2f}",
        "defect_rate": "{:.2f}",
        "energy_usage": "{:.2f}",
        "energy_cost": "${:.2f}"
    })
)


# --- Auto-refresh ---
st.markdown("Refreshing in 10 seconds...")
time.sleep(10)
st.rerun()
