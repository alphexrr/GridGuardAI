"""
GRID GUARD AI
Early Detection of Hidden Electrical Infrastructure Faults
Streamlit Interactive Dashboard & Hybrid AI Diagnostic Platform
Original Cyber-Physical Edition with Integrated 3D Digital Twin
"""

import os
import sys
import time
import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components

# Setup system path to import local modules
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.generator import generate_synthetic_sensor_data, CLASSES
from src.detection import GridGuardDetector, analyze_sensor_rules
from src.train_model import train_and_evaluate, FEATURE_COLS

# ---------------------------------------------------------
# PAGE CONFIGURATION & STYLING (ORIGINAL INITIAL THEME)
# ---------------------------------------------------------
st.set_page_config(
    page_title="GRID GUARD AI - Infrastructure Diagnostic System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Cyber-Physical Grid Styling (Dark Navy & Electric Blue Palette)
CUSTOM_CSS = """
<style>
    /* Global Background and Fonts */
    .stApp {
        background: radial-gradient(circle at 15% 15%, #0B132B 0%, #070D1F 100%);
        color: #E2E8F0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Top Header Bar */
    .grid-header {
        background: linear-gradient(135deg, rgba(28, 37, 65, 0.85) 0%, rgba(11, 19, 43, 0.95) 100%);
        border: 1px solid rgba(0, 240, 255, 0.25);
        border-radius: 12px;
        padding: 24px 30px;
        margin-bottom: 25px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(8px);
    }
    
    .grid-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #FFFFFF 0%, #00F0FF 50%, #48E5C2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    
    .grid-subtitle {
        font-size: 1.05rem;
        color: #94A3B8;
        font-weight: 400;
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(28, 37, 65, 0.7);
        border: 1px solid rgba(58, 80, 107, 0.6);
        border-radius: 10px;
        padding: 16px 20px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: #00F0FF;
        transform: translateY(-2px);
    }
    .metric-label {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #94A3B8;
        font-weight: 600;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #F8FAFC;
        margin-top: 4px;
    }

    /* Status Badges */
    .badge-normal {
        background-color: rgba(0, 230, 118, 0.15);
        color: #00E676;
        border: 1px solid #00E676;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-degradation {
        background-color: rgba(255, 164, 0, 0.15);
        color: #FFA400;
        border: 1px solid #FFA400;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-fault {
        background-color: rgba(255, 82, 82, 0.18);
        color: #FF5252;
        border: 1px solid #FF5252;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-manipulation {
        background-color: rgba(255, 23, 68, 0.2);
        color: #FF1744;
        border: 1px solid #FF1744;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }

    /* Educational Disclaimer Banner */
    .edu-banner {
        background: rgba(15, 23, 42, 0.9);
        border-left: 4px solid #00F0FF;
        border-radius: 6px;
        padding: 12px 18px;
        font-size: 0.88rem;
        color: #CBD5E1;
        margin-bottom: 20px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------
# COLOR MAP & CONSTANTS
# ---------------------------------------------------------
COLOR_MAP = {
    "normal": "#00E676",            # Green
    "degradation": "#FFA400",       # Amber
    "physical_fault": "#FF5252",    # Crimson Red
    "data_manipulation": "#E040FB"  # Electric Magenta / Cyber Red
}

MODEL_PATH = os.path.join(CURRENT_DIR, "models", "grid_guard_rf.joblib")
DATA_PATH = os.path.join(CURRENT_DIR, "data", "grid_guard_dataset.csv")
TWIN_HTML_PATH = os.path.join(CURRENT_DIR, "digital_twin_3d.html")

# ---------------------------------------------------------
# CACHED INITIALIZATION & RESOURCE LOADING
# ---------------------------------------------------------
@st.cache_resource
def load_detector() -> GridGuardDetector:
    """Loads or initializes the hybrid detector and trained model."""
    if not os.path.exists(MODEL_PATH):
        train_and_evaluate(
            data_path=DATA_PATH if os.path.exists(DATA_PATH) else None,
            save_model_path=MODEL_PATH
        )
    return GridGuardDetector(model_path=MODEL_PATH)

@st.cache_data
def get_default_dataset() -> pd.DataFrame:
    """Loads the pre-generated sample dataset."""
    if not os.path.exists(DATA_PATH):
        df = generate_synthetic_sensor_data(n_samples=650, random_state=42)
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
        return df
    return pd.read_csv(DATA_PATH)

detector = load_detector()

# ---------------------------------------------------------
# SIDEBAR NAVIGATION & CONTROLS (WITH INTEGRATED 3D MODEL)
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 10px 0;'>
        <div style='font-size: 2.5rem;'>⚡</div>
        <div style='font-size: 1.3rem; font-weight: 800; color: #00F0FF; letter-spacing: 1px;'>GRID GUARD AI</div>
        <div style='font-size: 0.8rem; color: #94A3B8;'>Early Fault & Tamper Detection</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    selected_page = st.radio(
        "Navigation",
        [
            "⚡ Grid Overview",
            "📡 3D Cyber-Physical Twin",
            "🔍 Analyze Sensor Data",
            "🔴 Live Grid Simulation",
            "📊 Model Evaluation",
            "ℹ️ Architecture & Transparency"
        ],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 🎛️ Active Telemetry")
    use_uploaded = st.checkbox("Use Uploaded Data (if available)", value=False)
    
    st.markdown("---")
    st.markdown("""
    <div style='font-size: 0.78rem; color: #64748B;'>
        <strong>Framework:</strong> Streamlit + scikit-learn<br>
        <strong>Model:</strong> Random Forest Classifier<br>
        <strong>3D Engine:</strong> Three.js WebGL (10 Hz DAQ)<br>
        <strong>Target Classes:</strong> 4 Condition States<br>
        <strong>Version:</strong> v1.0 Prototype
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# SESSION STATE MANAGEMENT
# ---------------------------------------------------------
if "uploaded_data" not in st.session_state:
    st.session_state["uploaded_data"] = None
if "analyzed_data" not in st.session_state:
    st.session_state["analyzed_data"] = None
if "sim_stream" not in st.session_state:
    st.session_state["sim_stream"] = []

# Select active DataFrame
if use_uploaded and st.session_state["uploaded_data"] is not None:
    active_df = st.session_state["uploaded_data"].copy()
    data_source_label = "Custom Uploaded Telemetry"
else:
    active_df = get_default_dataset().copy()
    data_source_label = "Pre-loaded Synthetic Baseline Dataset (650 records)"

# ---------------------------------------------------------
# HEADER COMPONENT & TRANSPARENCY BANNER
# ---------------------------------------------------------
st.markdown(f"""
<div class='grid-header'>
    <div class='grid-title'>GRID GUARD AI</div>
    <div class='grid-subtitle'>Early Detection of Hidden Electrical Infrastructure Faults & False Data Injections</div>
</div>
""", unsafe_allow_html=True)

# Educational Disclaimer Banner
st.markdown("""
<div class='edu-banner'>
    <strong>🎓 Project Transparency Note:</strong> This is an educational prototype using synthetic sensor data. It demonstrates the concept of detecting normal operation, physical faults, degradation, and potentially manipulated readings. It has not been validated for operational use on real electrical infrastructure.
</div>
""", unsafe_allow_html=True)


# =========================================================
# SECTION 1: OVERVIEW DASHBOARD
# =========================================================
if selected_page == "⚡ Grid Overview":
    st.subheader(f"📊 Infrastructure Telemetry Overview ({data_source_label})")
    
    # Run detector on active data
    try:
        analyzed_overview = detector.analyze_dataframe(active_df)
    except Exception as e:
        st.error(f"Error processing telemetry: {e}")
        analyzed_overview = active_df.copy()
        if "condition" in analyzed_overview.columns:
            analyzed_overview["predicted_condition"] = analyzed_overview["condition"]
        else:
            analyzed_overview["predicted_condition"] = "normal"
            
    # Calculate counts
    total_records = len(analyzed_overview)
    counts = analyzed_overview["predicted_condition"].value_counts().to_dict()
    n_normal = counts.get("normal", 0)
    n_fault = counts.get("physical_fault", 0)
    n_degrad = counts.get("degradation", 0)
    n_manip = counts.get("data_manipulation", 0)
    
    # Determine Grid Overall Health Status
    if n_fault > 0 or n_manip > 0:
        status_text = "CRITICAL ALERTS DETECTED"
        status_color = "#FF5252"
        status_desc = f"{n_fault} physical fault(s) and {n_manip} potential cyber-spoofing reading(s) flagged."
    elif n_degrad > 0:
        status_text = "MAINTENANCE ADVISORY"
        status_color = "#FFA400"
        status_desc = f"{n_degrad} component degradation event(s) detected."
    else:
        status_text = "ALL FEEDERS NOMINAL"
        status_color = "#00E676"
        status_desc = "Telemetry operating strictly within IEEE/IEC safety thresholds."
        
    # Top KPI Row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Total Readings</div>
            <div class='metric-value'>{total_records}</div>
            <div style='font-size: 0.8rem; color: #64748B; margin-top: 4px;'>Telemetry Records</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='metric-card' style='border-top: 3px solid #00E676;'>
            <div class='metric-label'>Normal State</div>
            <div class='metric-value' style='color: #00E676;'>{n_normal}</div>
            <div style='font-size: 0.8rem; color: #00E676; margin-top: 4px;'>{(n_normal/max(1, total_records))*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class='metric-card' style='border-top: 3px solid #FF5252;'>
            <div class='metric-label'>Physical Faults</div>
            <div class='metric-value' style='color: #FF5252;'>{n_fault}</div>
            <div style='font-size: 0.8rem; color: #FF5252; margin-top: 4px;'>{(n_fault/max(1, total_records))*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class='metric-card' style='border-top: 3px solid #FFA400;'>
            <div class='metric-label'>Degradation</div>
            <div class='metric-value' style='color: #FFA400;'>{n_degrad}</div>
            <div style='font-size: 0.8rem; color: #FFA400; margin-top: 4px;'>{(n_degrad/max(1, total_records))*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
        <div class='metric-card' style='border-top: 3px solid #E040FB;'>
            <div class='metric-label'>Data Tampering</div>
            <div class='metric-value' style='color: #E040FB;'>{n_manip}</div>
            <div style='font-size: 0.8rem; color: #E040FB; margin-top: 4px;'>{(n_manip/max(1, total_records))*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    # Grid System Status Banner
    st.markdown(f"""
    <div style='background: rgba(15, 23, 42, 0.8); border: 1px solid {status_color}; border-radius: 8px; padding: 14px 20px; margin: 20px 0;'>
        <div style='display: flex; align-items: center; justify-content: space-between;'>
            <div>
                <span style='font-size: 0.85rem; text-transform: uppercase; color: #94A3B8; font-weight: 600;'>System Status:</span>
                <span style='font-size: 1.15rem; font-weight: 800; color: {status_color}; margin-left: 10px;'>● {status_text}</span>
            </div>
            <div style='font-size: 0.9rem; color: #CBD5E1;'>{status_desc}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Charts Row: Distribution & Time Series
    chart_col1, chart_col2 = st.columns([1, 2])
    
    with chart_col1:
        st.markdown("#### 🧭 Condition Distribution")
        dist_df = pd.DataFrame({
            "Condition": list(counts.keys()),
            "Readings": list(counts.values())
        })
        fig_pie = px.pie(
            dist_df,
            values="Readings",
            names="Condition",
            color="Condition",
            color_discrete_map=COLOR_MAP,
            hole=0.55
        )
        fig_pie.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=20, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with chart_col2:
        st.markdown("#### 📈 Multi-Sensor Telemetry Trends")
        time_x = analyzed_overview["timestamp"] if "timestamp" in analyzed_overview.columns else np.arange(len(analyzed_overview))
        
        fig_time = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Voltage (V) [Nominal 230V]", "Current (A) [Normal < 40A]", "Temperature (°C) [Safe < 65°C]", "Vibration (mm/s) [Safe < 3.0]"),
            vertical_spacing=0.15,
            horizontal_spacing=0.08
        )
        
        # Voltage trace
        fig_time.add_trace(go.Scatter(
            x=time_x, y=analyzed_overview["voltage"],
            mode="lines", name="Voltage", line=dict(color="#00F0FF", width=1.5)
        ), row=1, col=1)
        fig_time.add_hline(y=185.0, line_dash="dot", line_color="#FF5252", annotation_text="Sag Limit (185V)", row=1, col=1)
        
        # Current trace
        fig_time.add_trace(go.Scatter(
            x=time_x, y=analyzed_overview["current"],
            mode="lines", name="Current", line=dict(color="#FFD166", width=1.5)
        ), row=1, col=2)
        fig_time.add_hline(y=65.0, line_dash="dot", line_color="#FF5252", annotation_text="Overcurrent (65A)", row=1, col=2)
        
        # Temperature trace
        fig_time.add_trace(go.Scatter(
            x=time_x, y=analyzed_overview["temperature"],
            mode="lines", name="Temperature", line=dict(color="#FF5252", width=1.5)
        ), row=2, col=1)
        fig_time.add_hline(y=78.0, line_dash="dot", line_color="#FF1744", annotation_text="Thermal Trip (78°C)", row=2, col=1)
        
        # Vibration trace
        fig_time.add_trace(go.Scatter(
            x=time_x, y=analyzed_overview["vibration"],
            mode="lines", name="Vibration", line=dict(color="#48E5C2", width=1.5)
        ), row=2, col=2)
        fig_time.add_hline(y=5.5, line_dash="dot", line_color="#FF1744", annotation_text="Vib Alarm (5.5mm/s)", row=2, col=2)
        
        fig_time.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(11,19,43,0.5)",
            height=380,
            showlegend=False,
            margin=dict(l=30, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_time, use_container_width=True)


# =========================================================
# SECTION 2: INTEGRATED 3D CYBER-PHYSICAL DIGITAL TWIN
# =========================================================
elif selected_page == "📡 3D Cyber-Physical Twin":
    st.subheader("📡 3D Cyber-Physical Substation Twin & Live Telemetry Collection")
    st.markdown("""
    Interactive 3D model of a distribution power transformer. Inspect live physical sensor probes 
    (Voltage PT, Current CT, RTD Thermal Probe, Accelerometer), observe real-time 50 Hz AC waveforms, 
    and trigger fault scenarios directly in 3D.
    """)

    # Controls Bar
    t_col1, t_col2 = st.columns([1, 1])
    with t_col1:
        expand_3d = st.checkbox("Expand Viewport Height (Full Screen Mode)", value=False)
    with t_col2:
        st.markdown("""
        <div style='text-align: right;'>
            <a href='file:///C:/Users/nandd/OneDrive/Desktop/GridGuardAI/digital_twin_3d.html' target='_blank' style='background: rgba(0, 240, 255, 0.15); border: 1px solid #00F0FF; color: #00F0FF; padding: 6px 14px; border-radius: 6px; text-decoration: none; font-size: 0.82rem; font-weight: 700;'>
                ⛶ Open Standalone Fullscreen Tab
            </a>
        </div>
        """, unsafe_allow_html=True)

    twin_height = 950 if expand_3d else 800

    if os.path.exists(TWIN_HTML_PATH):
        try:
            with open(TWIN_HTML_PATH, "r", encoding="utf-8") as f:
                twin_code = f.read()
            components.html(twin_code, height=twin_height, scrolling=False)
            st.caption("⚡ Powered by Three.js WebGL & Web Audio API (50 Hz magnetostriction hum). Controls: Left drag to rotate, right drag to pan, scroll to zoom.")
        except Exception as e:
            st.error(f"Error loading 3D Digital Twin: {e}")
    else:
        st.warning("3D Model file not found.")


# =========================================================
# SECTION 3: ANALYZE SENSOR DATA
# =========================================================
elif selected_page == "🔍 Analyze Sensor Data":
    st.subheader("🔍 Automated Sensor Telemetry Ingestion & Diagnostic Engine")
    st.markdown("""
    Upload a sensor telemetry CSV file to automatically detect hidden physical faults, component wear degradation, 
    and false data injection attacks. You can also download our calibrated baseline sample dataset below.
    """)
    
    # Download sample button
    sample_df = get_default_dataset()
    csv_buffer = io.StringIO()
    sample_df.to_csv(csv_buffer, index=False)
    
    btn_col1, btn_col2 = st.columns([1, 3])
    with btn_col1:
        st.download_button(
            label="📥 Download Sample CSV",
            data=csv_buffer.getvalue(),
            file_name="grid_guard_sample_telemetry.csv",
            mime="text/csv",
            help="Download a validated 650-row synthetic electrical grid sensor dataset."
        )
        
    uploaded_file = st.file_uploader(
        "Upload Grid Sensor Telemetry (CSV format)",
        type=["csv"],
        help="Expected columns: timestamp, voltage, current, temperature, vibration (optional: condition)"
    )
    
    target_df = None
    if uploaded_file is not None:
        try:
            raw_upload = pd.read_csv(uploaded_file)
            required_cols = ["voltage", "current", "temperature", "vibration"]
            cols_lower = {col.lower(): col for col in raw_upload.columns}
            missing_cols = [rc for rc in required_cols if rc not in cols_lower]
            
            if missing_cols:
                st.error(f"❌ Upload Failed: Required column(s) missing: {missing_cols}. Please ensure file contains: voltage, current, temperature, vibration.")
            elif raw_upload.empty:
                st.error("❌ Upload Failed: The uploaded CSV file is empty.")
            else:
                rename_map = {cols_lower[rc]: rc for rc in required_cols}
                if "timestamp" in cols_lower: rename_map[cols_lower["timestamp"]] = "timestamp"
                if "condition" in cols_lower: rename_map[cols_lower["condition"]] = "condition"
                    
                target_df = raw_upload.rename(columns=rename_map)
                st.session_state["uploaded_data"] = target_df
                st.success(f"✅ Successfully validated {len(target_df)} sensor records from '{uploaded_file.name}'!")
        except Exception as e:
            st.error(f"❌ Could not parse uploaded CSV: {e}")
    else:
        if st.checkbox("Analyze the loaded baseline dataset", value=True):
            target_df = sample_df.copy()
            st.info("ℹ️ Using pre-loaded synthetic grid dataset.")

    if target_df is not None:
        st.markdown("### 📋 Telemetry Preview")
        st.dataframe(target_df.head(6), use_container_width=True)
        
        if st.button("🚀 Run Grid Diagnostic & Prediction Engine", type="primary"):
            with st.spinner("Executing hybrid ML classifier and physics invariant rule analysis..."):
                analyzed_res = detector.analyze_dataframe(target_df)
                st.session_state["analyzed_data"] = analyzed_res
                st.success("Diagnostic analysis completed!")

        if st.session_state["analyzed_data"] is not None:
            res_df = st.session_state["analyzed_data"]
            st.markdown("---")
            st.markdown("### 📊 Diagnostic Analysis Findings")
            
            res_counts = res_df["predicted_condition"].value_counts()
            total_n = len(res_df)
            
            m_cols = st.columns(4)
            for idx, cls_name in enumerate(CLASSES):
                cnt = res_counts.get(cls_name, 0)
                pct = (cnt / total_n) * 100
                with m_cols[idx]:
                    st.markdown(f"""
                    <div class='metric-card' style='border-left: 4px solid {COLOR_MAP[cls_name]};'>
                        <div class='metric-label'>{cls_name.replace('_', ' ').title()}</div>
                        <div class='metric-value' style='color: {COLOR_MAP[cls_name]};'>{cnt}</div>
                        <div style='font-size: 0.85rem; color: #94A3B8;'>{pct:.1f}% of total</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
            if "condition" in res_df.columns:
                match_count = (res_df["predicted_condition"] == res_df["condition"]).sum()
                acc_rate = (match_count / total_n) * 100.0
                st.info(f"🎯 **Ground Truth Concordance:** {acc_rate:.2f}% ({match_count}/{total_n} predictions matched the provided condition label).")

            st.markdown("#### 🚨 Flagged Suspicious & Anomaly Telemetry")
            flagged_rows = res_df[res_df["is_abnormal"]].copy()
            
            if flagged_rows.empty:
                st.success("🎉 No suspicious or abnormal telemetry readings detected! All systems nominal.")
            else:
                display_cols = ["timestamp", "voltage", "current", "temperature", "vibration", "predicted_condition", "confidence_pct", "diagnostic_explanation"]
                display_cols = [c for c in display_cols if c in flagged_rows.columns]
                
                st.markdown(f"**Showing {len(flagged_rows)} flagged readings requiring operator attention:**")
                st.dataframe(
                    flagged_rows[display_cols].style.applymap(
                        lambda val: f"color: {COLOR_MAP.get(val, '#E2E8F0')}; font-weight: bold;" if val in COLOR_MAP else ""
                    ),
                    use_container_width=True
                )

            st.markdown("#### 📈 Sensor Telemetry with Anomaly Markers")
            fig_scatter = go.Figure()
            
            x_vals = res_df["timestamp"] if "timestamp" in res_df.columns else np.arange(len(res_df))
            fig_scatter.add_trace(go.Scatter(
                x=x_vals, y=res_df["voltage"],
                mode="lines", name="Voltage (V)", line=dict(color="#3A506B", width=1.2)
            ))
            
            for cls_name in ["physical_fault", "degradation", "data_manipulation"]:
                subset = res_df[res_df["predicted_condition"] == cls_name]
                if not subset.empty:
                    sub_x = subset["timestamp"] if "timestamp" in subset.columns else subset.index
                    fig_scatter.add_trace(go.Scatter(
                        x=sub_x, y=subset["voltage"],
                        mode="markers",
                        name=f"Detected: {cls_name}",
                        marker=dict(color=COLOR_MAP[cls_name], size=7, symbol="diamond"),
                        hovertext=subset["diagnostic_explanation"]
                    ))
                    
            fig_scatter.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(11,19,43,0.6)",
                height=350,
                title="Voltage Telemetry with Flagged Condition Classifications",
                margin=dict(l=30, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

            out_buffer = io.StringIO()
            res_df.to_csv(out_buffer, index=False)
            st.download_button(
                label="💾 Download Complete Analyzed Diagnostic Report (CSV)",
                data=out_buffer.getvalue(),
                file_name="grid_guard_analyzed_report.csv",
                mime="text/csv",
                type="primary"
            )


# =========================================================
# SECTION 4: LIVE GRID SIMULATION
# =========================================================
elif selected_page == "🔴 Live Grid Simulation":
    st.subheader("🔴 Interactive Cyber-Physical Telemetry Simulator")
    st.markdown("""
    Generate synthetic electrical telemetry in real time to simulate grid transients, equipment wear, 
    and cyber-physical sensor tampering. Observe how the detection system diagnoses anomalies on the fly.
    """)
    
    sim_col1, sim_col2, sim_col3 = st.columns([1, 1, 1])
    with sim_col1:
        num_readings = st.slider("Readings to Generate", min_value=5, max_value=50, value=15, step=5)
    with sim_col2:
        scenario_type = st.selectbox(
            "Telemetry Scenario",
            [
                "Mixed Operational Stream (Realistic)",
                "Standard Nominal Baseline",
                "Feeder Line Short-Circuit (Physical Fault)",
                "Bushing / Contact Thermal Degradation",
                "False Data Injection (Cyber Spoofing)"
            ]
        )
    with sim_col3:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        btn_c1, btn_c2 = st.columns(2)
        with btn_c1:
            run_sim = st.button("▶ Run Simulation", type="primary")
        with btn_c2:
            reset_sim = st.button("🔄 Reset Buffer")
            
    if reset_sim:
        st.session_state["sim_stream"] = []
        st.success("Simulation buffer cleared.")

    if run_sim:
        new_records = []
        current_time = pd.Timestamp.now()
        
        for k in range(num_readings):
            ts = (current_time + pd.Timedelta(seconds=k * 10)).strftime("%H:%M:%S")
            
            if scenario_type == "Standard Nominal Baseline":
                v = float(np.random.normal(230.0, 1.0))
                i = float(np.random.normal(22.0, 2.0))
                t = float(42.0 + (i**1.2) * 0.15 + np.random.normal(0, 0.5))
                vib = float(np.random.normal(1.2, 0.2))
                true_cond = "normal"
            elif scenario_type == "Feeder Line Short-Circuit (Physical Fault)":
                v = float(np.random.uniform(135.0, 170.0))
                i = float(np.random.uniform(85.0, 130.0))
                t = float(np.random.uniform(82.0, 105.0))
                vib = float(np.random.uniform(6.5, 11.0))
                true_cond = "physical_fault"
            elif scenario_type == "Bushing / Contact Thermal Degradation":
                prog = k / max(1, num_readings)
                v = float(224.0 - prog * 8.0 + np.random.normal(0, 1.0))
                i = float(26.0 + prog * 8.0 + np.random.normal(0, 1.2))
                t = float(62.0 + prog * 12.0 + np.random.normal(0, 0.8))
                vib = float(3.6 + prog * 1.5 + np.random.normal(0, 0.3))
                true_cond = "degradation"
            elif scenario_type == "False Data Injection (Cyber Spoofing)":
                if k % 2 == 0:
                    v, i, t, vib = 230.00, 20.00, 42.00, 1.20
                else:
                    v, i, t, vib = 12.0, 45.0, 70.0, 1.5
                true_cond = "data_manipulation"
            else:
                rand_val = np.random.rand()
                if rand_val < 0.60:
                    v, i, t, vib = float(np.random.normal(230.0, 1.1)), float(np.random.normal(23.0, 2.2)), float(41.0 + (23**1.2)*0.16), float(np.random.normal(1.3, 0.2))
                    true_cond = "normal"
                elif rand_val < 0.75:
                    v, i, t, vib = float(np.random.uniform(140.0, 175.0)), float(np.random.uniform(70.0, 115.0)), float(np.random.uniform(78.0, 95.0)), float(np.random.uniform(5.8, 9.2))
                    true_cond = "physical_fault"
                elif rand_val < 0.88:
                    v, i, t, vib = float(np.random.uniform(214.0, 222.0)), float(np.random.uniform(28.0, 38.0)), float(np.random.uniform(62.0, 74.0)), float(np.random.uniform(3.5, 5.0))
                    true_cond = "degradation"
                else:
                    v, i, t, vib = float(np.random.uniform(8.0, 18.0)), float(np.random.uniform(32.0, 50.0)), float(np.random.uniform(65.0, 80.0)), 1.5
                    true_cond = "data_manipulation"

            new_records.append({
                "timestamp": ts,
                "voltage": round(v, 2),
                "current": round(i, 2),
                "temperature": round(t, 2),
                "vibration": round(vib, 2),
                "condition": true_cond
            })
            
        sim_df = pd.DataFrame(new_records)
        evaluated_sim = detector.analyze_dataframe(sim_df)
        
        existing = st.session_state["sim_stream"]
        combined = existing + evaluated_sim.to_dict(orient="records")
        st.session_state["sim_stream"] = combined[-60:]
        st.toast(f"Generated {num_readings} new telemetry readings!", icon="⚡")

    if st.session_state["sim_stream"]:
        stream_df = pd.DataFrame(st.session_state["sim_stream"])
        latest = stream_df.iloc[-1]
        
        st.markdown("### 🔔 Latest Telemetry Diagnostic Assessment")
        lat_cond = latest["predicted_condition"]
        lat_expl = latest["diagnostic_explanation"]
        lat_conf = latest.get("confidence_pct", 95.0)
        
        if lat_cond == "normal":
            st.markdown(f"""
            <div style='background: rgba(0, 230, 118, 0.12); border-left: 5px solid #00E676; padding: 14px 20px; border-radius: 6px;'>
                <div style='font-size: 1.1rem; font-weight: 700; color: #00E676;'>✅ SYSTEM NOMINAL (Confidence: {lat_conf:.1f}%)</div>
                <div style='color: #CBD5E1; margin-top: 4px;'>{lat_expl}</div>
                <div style='font-size: 0.82rem; color: #94A3B8; margin-top: 6px;'>Parameters: V={latest['voltage']}V | I={latest['current']}A | T={latest['temperature']}°C | Vib={latest['vibration']}mm/s</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            alert_color = COLOR_MAP.get(lat_cond, "#FF5252")
            st.markdown(f"""
            <div style='background: rgba(255, 82, 82, 0.15); border-left: 5px solid {alert_color}; padding: 14px 20px; border-radius: 6px;'>
                <div style='font-size: 1.1rem; font-weight: 800; color: {alert_color};'>🚨 ALERT: {lat_cond.upper()} DETECTED (Confidence: {lat_conf:.1f}%)</div>
                <div style='color: #F8FAFC; font-weight: 500; margin-top: 4px;'>{lat_expl}</div>
                <div style='font-size: 0.82rem; color: #CBD5E1; margin-top: 6px;'>Observed: V={latest['voltage']}V | I={latest['current']}A | T={latest['temperature']}°C | Vib={latest['vibration']}mm/s</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### ♊ Simplified Digital-Twin Inspired Sensor Comparison")
        st.caption("Demonstrates expected theoretical nominal baseline versus currently observed telemetry.")
        
        dt_col1, dt_col2, dt_col3, dt_col4 = st.columns(4)
        v_exp, i_exp, t_exp, vib_exp = 230.0, 24.0, 42.0, 1.4
        
        with dt_col1:
            v_obs = latest["voltage"]
            st.metric("Voltage (vs 230V Nominal)", f"{v_obs:.1f} V", f"{v_obs - v_exp:+.1f} V")
        with dt_col2:
            i_obs = latest["current"]
            st.metric("Current (vs 24A Nominal)", f"{i_obs:.1f} A", f"{i_obs - i_exp:+.1f} A")
        with dt_col3:
            t_obs = latest["temperature"]
            st.metric("Temperature (vs 42°C Nominal)", f"{t_obs:.1f} °C", f"{t_obs - t_exp:+.1f} °C")
        with dt_col4:
            vib_obs = latest["vibration"]
            st.metric("Vibration (vs 1.4 mm/s Nominal)", f"{vib_obs:.2f} mm/s", f"{vib_obs - vib_exp:+.2f} mm/s")

        st.markdown("#### 📡 Real-Time Telemetry Stream Strip Chart")
        fig_sim = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Live Voltage Stream (V)", "Live Current Stream (A)", "Live Temperature (°C)", "Live Vibration (mm/s)"),
            vertical_spacing=0.18
        )
        x_ticks = stream_df["timestamp"]
        fig_sim.add_trace(go.Scatter(x=x_ticks, y=stream_df["voltage"], mode="lines+markers", line=dict(color="#00F0FF", width=2), marker=dict(size=4)), row=1, col=1)
        fig_sim.add_trace(go.Scatter(x=x_ticks, y=stream_df["current"], mode="lines+markers", line=dict(color="#FFD166", width=2), marker=dict(size=4)), row=1, col=2)
        fig_sim.add_trace(go.Scatter(x=x_ticks, y=stream_df["temperature"], mode="lines+markers", line=dict(color="#FF5252", width=2), marker=dict(size=4)), row=2, col=1)
        fig_sim.add_trace(go.Scatter(x=x_ticks, y=stream_df["vibration"], mode="lines+markers", line=dict(color="#48E5C2", width=2), marker=dict(size=4)), row=2, col=2)
        fig_sim.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(11,19,43,0.5)", height=400, showlegend=False, margin=dict(l=30, r=20, t=30, b=20))
        st.plotly_chart(fig_sim, use_container_width=True)
        
        st.markdown("#### 📜 Live Telemetry Log")
        st.dataframe(stream_df[["timestamp", "voltage", "current", "temperature", "vibration", "predicted_condition", "confidence_pct", "diagnostic_explanation"]].tail(10), use_container_width=True)


# =========================================================
# SECTION 5: MODEL EVALUATION
# =========================================================
elif selected_page == "📊 Model Evaluation":
    st.subheader("📊 scikit-learn Baseline Classifier Evaluation")
    st.markdown("""
    Evaluation of the baseline **Random Forest Classifier** trained on synthetic electrical telemetry.
    All metrics and performance reports below are calculated strictly from actual predictions on held-out test data (20% stratified test split).
    """)
    
    with st.expander("⚙️ Retrain Classifier / Adjust Hyperparameters", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            n_trees = st.slider("Number of Estimators", min_value=50, max_value=200, value=120, step=10)
        with c2:
            max_d = st.slider("Max Tree Depth", min_value=5, max_value=20, value=12, step=1)
        with c3:
            retrain_btn = st.button("🔄 Retrain Baseline Model")
            
        if retrain_btn:
            with st.spinner("Retraining Random Forest model on telemetry..."):
                train_and_evaluate(data_path=DATA_PATH, save_model_path=MODEL_PATH, random_state=42)
                detector.load_model(MODEL_PATH)
                st.success("Model successfully retrained and reloaded!")

    try:
        import joblib
        model_bundle = joblib.load(MODEL_PATH)
        metrics = model_bundle.get("metrics", {})
    except Exception:
        metrics = train_and_evaluate(data_path=DATA_PATH, save_model_path=MODEL_PATH)
        
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class='metric-card' style='border-left: 4px solid #00F0FF;'>
            <div class='metric-label'>Held-Out Accuracy</div>
            <div class='metric-value' style='color: #00F0FF;'>{metrics['accuracy'] * 100:.2f}%</div>
            <div style='font-size: 0.8rem; color: #94A3B8;'>N_test = {metrics.get('n_test', 130)} samples</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class='metric-card' style='border-left: 4px solid #48E5C2;'>
            <div class='metric-label'>Precision (Macro)</div>
            <div class='metric-value' style='color: #48E5C2;'>{metrics['precision_macro']:.4f}</div>
            <div style='font-size: 0.8rem; color: #94A3B8;'>Weighted: {metrics['precision_weighted']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class='metric-card' style='border-left: 4px solid #FFD166;'>
            <div class='metric-label'>Recall (Macro)</div>
            <div class='metric-value' style='color: #FFD166;'>{metrics['recall_macro']:.4f}</div>
            <div style='font-size: 0.8rem; color: #94A3B8;'>Weighted: {metrics['recall_weighted']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class='metric-card' style='border-left: 4px solid #FFA400;'>
            <div class='metric-label'>F1-Score (Macro)</div>
            <div class='metric-value' style='color: #FFA400;'>{metrics['f1_macro']:.4f}</div>
            <div style='font-size: 0.8rem; color: #94A3B8;'>Weighted: {metrics['f1_weighted']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    
    ev_col1, ev_col2 = st.columns(2)
    with ev_col1:
        st.markdown("#### 🎯 Confusion Matrix (Test Split)")
        cm_data = np.array(metrics["confusion_matrix"])
        fig_cm = px.imshow(
            cm_data,
            labels=dict(x="Predicted Condition", y="Actual Ground Truth", color="Sample Count"),
            x=CLASSES, y=CLASSES, text_auto=True, color_continuous_scale="Tealgrn"
        )
        fig_cm.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=40, r=20, t=30, b=30), height=360)
        st.plotly_chart(fig_cm, use_container_width=True)
        
    with ev_col2:
        st.markdown("#### 🌲 Random Forest Feature Importances")
        feat_dict = metrics.get("feature_importances", {})
        feat_df = pd.DataFrame({"Feature": list(feat_dict.keys()), "Importance": list(feat_dict.values())}).sort_values(by="Importance", ascending=True)
        fig_feat = px.bar(feat_df, x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale="Purp")
        fig_feat.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=40, r=20, t=30, b=30), height=360, coloraxis_showscale=False)
        st.plotly_chart(fig_feat, use_container_width=True)

    st.markdown("#### 📋 Detailed Classification Report (Per-Class Metrics)")
    report_dict = metrics.get("classification_report", {})
    rep_rows = []
    for cls in CLASSES:
        if cls in report_dict:
            rep_rows.append({
                "Condition Class": cls.replace('_', ' ').title(),
                "Precision": f"{report_dict[cls]['precision']:.4f}",
                "Recall": f"{report_dict[cls]['recall']:.4f}",
                "F1-Score": f"{report_dict[cls]['f1-score']:.4f}",
                "Test Support": int(report_dict[cls]['support'])
            })
    st.table(pd.DataFrame(rep_rows))


# =========================================================
# SECTION 6: ARCHITECTURE & TRANSPARENCY
# =========================================================
elif selected_page == "ℹ️ Architecture & Transparency":
    st.subheader("ℹ️ System Architecture & Engineering Specifications")
    
    st.markdown("""
    **GRID GUARD AI** employs a two-tier hybrid diagnostic architecture that unifies machine-learning statistical pattern recognition with deterministic electrical engineering physical invariant checks.
    """)
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div style='background: rgba(28, 37, 65, 0.6); border: 1px solid rgba(0, 240, 255, 0.2); border-radius: 8px; padding: 18px;'>
            <h4 style='color: #00F0FF; margin-top: 0;'>⚡ Tier 1: Physics Invariant & Anomaly Engine</h4>
            <p style='font-size: 0.88rem; color: #CBD5E1;'>Checks raw and derived electrical properties against deterministic physical laws:</p>
            <ul style='font-size: 0.85rem; color: #94A3B8;'>
                <li><strong>Ohm's & Joule's Invariants:</strong> Flags zero-voltage with high-current telemetry.</li>
                <li><strong>Thermal Dissipation:</strong> Identifies high operating temperatures occurring with zero load.</li>
                <li><strong>Entropy Analysis:</strong> Detects telemetry flatlines where sensor noise is unnaturally zeroed.</li>
                <li><strong>IEEE Std 1159 Voltage Limits:</strong> Flags sags below 185V and surges above 260V.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col_b:
        st.markdown("""
        <div style='background: rgba(28, 37, 65, 0.6); border: 1px solid rgba(72, 229, 194, 0.2); border-radius: 8px; padding: 18px;'>
            <h4 style='color: #48E5C2; margin-top: 0;'>🌲 Tier 2: Ensemble Random Forest Classifier</h4>
            <p style='font-size: 0.88rem; color: #CBD5E1;'>A 120-tree Random Forest model trained on cross-feature multi-modal telemetry:</p>
            <ul style='font-size: 0.85rem; color: #94A3B8;'>
                <li><strong>Apparent Power (kVA):</strong> Computed via V &times; I / 1000 for load tracking.</li>
                <li><strong>Apparent Impedance Proxy:</strong> Tracks line impedance degradation V / (I + &epsilon;).</li>
                <li><strong>Thermal-Load Ratio:</strong> Pinpoints contact resistance escalation before failure.</li>
                <li><strong>Vibration-Power Index:</strong> Detects mechanical looseness in transformer cores.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    st.markdown("#### 📐 Telemetry Thresholds Reference Matrix")
    threshold_data = [
        {"Parameter": "Voltage (V)", "Nominal Range": "225.0 - 235.0 V", "Degradation Band": "195.0 - 220.0 V", "Critical Fault Limit": "< 185.0 V or > 260.0 V", "Physical Meaning": "Distribution feeder voltage regulation"},
        {"Parameter": "Current (A)", "Nominal Range": "15.0 - 35.0 A", "Degradation Band": "35.0 - 50.0 A (elevated)", "Critical Fault Limit": ">= 65.0 A", "Physical Meaning": "Load draw and short-circuit fault current"},
        {"Parameter": "Temperature (°C)", "Nominal Range": "35.0 - 52.0 °C", "Degradation Band": "58.0 - 75.0 °C", "Critical Fault Limit": ">= 78.0 °C", "Physical Meaning": "Winding and bushing thermal dissipation"},
        {"Parameter": "Vibration (mm/s)", "Nominal Range": "0.8 - 2.5 mm/s", "Degradation Band": "3.2 - 5.4 mm/s", "Critical Fault Limit": ">= 5.5 mm/s", "Physical Meaning": "Transformer core hum and bearing looseness"}
    ]
    st.table(pd.DataFrame(threshold_data))

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; font-size: 0.8rem; color: #64748B; padding-bottom: 20px;'>
    GRID GUARD AI Prototype &bull; Student Engineering Project &bull; Local Streamlit Demonstration Environment
</div>
""", unsafe_allow_html=True)
