import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from data_simulator import get_live_data, generate_fake_history

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

    .metric-value {
        font-size: 28px;
        font-weight: 600;
        color: #ffffff;
    }

    .metric-label {
        font-size: 14px;
        color: #bbbbbb;
    }

    .gauge-title {
        text-align: center;
        font-size: 16px;
        color: white;
        margin-top: -12px;
    }

    .stPlotlyChart {
        border-radius: 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    }

    table {
        background-color: rgba(255,255,255,0.05) !important;
        color: white;
    }

    .block-container {
        padding: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- Gauge Function ---
def create_gauge(value, green_range, orange_range, red_range, unit='°C'):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={'suffix': f" {unit}", 'font': {'size': 24, 'color': "white"}},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [None, red_range[1]], 'tickcolor': "white"},
            'bar': {'color': "#d291ff"},
            'steps': [
                {'range': green_range, 'color': "#3a0ca3"},
                {'range': orange_range, 'color': "#ff5d8f"},
                {'range': red_range, 'color': "#ff0040"},
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        height=220,
        margin=dict(t=20, b=10, l=0, r=0)
    )
    return fig

# --- Get Data ---
live_df = get_live_data()
live = live_df.iloc[0]
history_df = generate_fake_history(n=50, freq='H')

# --- Title ---
st.title("🏭 Manufacturing Operations Dashboard")

# --- KPI Section ---
st.markdown("### ")
kpi_cols = st.columns(5)

with kpi_cols[0]:
    st.markdown(f"<div class='metric-card'><div class='metric-value'>{live['units_produced']}</div><div class='metric-label'>Units Produced</div></div>", unsafe_allow_html=True)
with kpi_cols[1]:
    st.markdown(f"<div class='metric-card'><div class='metric-value'>{live['production_efficiency']:.1f}%</div><div class='metric-label'>Efficiency</div></div>", unsafe_allow_html=True)
with kpi_cols[2]:
    st.markdown(f"<div class='metric-card'><div class='metric-value'>{live['defect_rate']:.2f}%</div><div class='metric-label'>Defect Rate</div></div>", unsafe_allow_html=True)
with kpi_cols[3]:
    st.markdown(f"<div class='metric-card'><div class='metric-value'>{live['energy_usage']:.1f} kWh</div><div class='metric-label'>Energy Use</div></div>", unsafe_allow_html=True)
with kpi_cols[4]:
    st.markdown(f"<div class='metric-card'><div class='metric-value'>${live['energy_cost']:.2f}</div><div class='metric-label'>Energy Cost</div></div>", unsafe_allow_html=True)

# --- Gauges ---
gauge_cols_1 = st.columns(2)
gauge_cols_2 = st.columns(2)

with gauge_cols_1[0]:
    st.plotly_chart(create_gauge(live["oil_temp"], (40, 70), (70, 85), (85, 120)), use_container_width=True)
    st.markdown("<div class='gauge-title'>Oil Temp</div>", unsafe_allow_html=True)

with gauge_cols_1[1]:
    st.plotly_chart(create_gauge(live["hydraulic_temp"], (30, 60), (60, 75), (75, 120)), use_container_width=True)
    st.markdown("<div class='gauge-title'>Hydraulic Temp</div>", unsafe_allow_html=True)

with gauge_cols_2[0]:
    st.plotly_chart(create_gauge(live["vibration"], (0.0, 4.0), (4.0, 7.0), (7.0, 15.0), unit="mm/s"), use_container_width=True)
    st.markdown("<div class='gauge-title'>Vibration</div>", unsafe_allow_html=True)

with gauge_cols_2[1]:
    st.plotly_chart(create_gauge(live["bearing_temp"], (40, 80), (80, 95), (95, 120)), use_container_width=True)
    st.markdown("<div class='gauge-title'>Bearing Temp</div>", unsafe_allow_html=True)

# --- Charts ---
st.markdown("### ")
chart_row = st.columns(2)

with chart_row[0]:
    fig_bar = px.bar(history_df, x="machine_type", y="units_produced", title="Daily Production by Machine Type")
    fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="white", title_font=dict(size=16), margin=dict(t=40, b=20))
    st.plotly_chart(fig_bar, use_container_width=True)

with chart_row[1]:
    fig_line = px.line(history_df, x="timestamp", y="production_efficiency", title="Production Efficiency Trend")
    fig_line.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="white", title_font=dict(size=16), margin=dict(t=40, b=20))
    st.plotly_chart(fig_line, use_container_width=True)

# --- Table ---
st.markdown("### 📋 Summary Table")
st.dataframe(
    history_df.tail(8).style.format({
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
