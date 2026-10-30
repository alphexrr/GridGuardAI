# GRID GUARD AI
### Early Detection of Hidden Electrical Infrastructure Faults & False Data Injections

---

## ⚡ Project Overview

**GRID GUARD AI** is an intelligent cyber-physical electrical grid diagnostic prototype built with Python, Streamlit, Plotly, and scikit-learn. It is engineered to detect early-stage hidden electrical infrastructure faults, gradual thermo-mechanical equipment degradation, and malicious sensor telemetry tampering (False Data Injection / FDI) before catastrophic grid blackout or equipment destruction occurs.

The system features a **two-tier hybrid diagnostic architecture**:
1. **Tier 1 (Physics Invariant & Deterministic Rules):** Evaluates physical laws (Ohm's Law, Joule heating, telemetry entropy, and IEEE Std 1159 voltage tolerances).
2. **Tier 2 (Ensemble Machine Learning):** A 120-tree `RandomForestClassifier` trained on multi-modal cross-feature engineering (apparent power, impedance proxy, thermal-load ratio, and vibration-power index).

---

## 🏗️ Project Architecture

```
GridGuardAI/
├── app.py                      # Main Streamlit web application & interactive dashboard
├── requirements.txt            # Python package dependencies
├── README.md                   # System documentation & Windows instructions
├── data/
│   └── grid_guard_dataset.csv  # 650 synthetic telemetry records across 4 condition states
├── src/
│   ├── __init__.py
│   ├── generator.py            # Reproducible synthetic telemetry generation engine
│   ├── detection.py            # Hybrid detector (ML + physics-invariant diagnostic rules)
│   └── train_model.py          # scikit-learn model training & evaluation pipeline
├── models/
│   └── grid_guard_rf.joblib    # Serialized trained Random Forest model & metrics bundle
└── outputs/                    # Output reports and CSV exports
```

---

## 🌟 Key Application Sections

### 1. ⚡ Grid Overview
- Real-time KPI cards: Total Telemetry Records, Normal Operation, Physical Faults, Component Degradation, and Data Tampering counts and percentages.
- Overall Grid Health Status banner (ALL FEEDERS NOMINAL, MAINTENANCE ADVISORY, or CRITICAL ALERTS DETECTED).
- Condition distribution donut chart.
- Synchronized multi-trace time-series charts displaying Voltage, Current, Temperature, and Vibration with IEEE/IEC safety threshold limit indicators.

### 2. 🔍 Analyze Sensor Data
- Ingest sensor telemetry CSV files with comprehensive column validation (`voltage`, `current`, `temperature`, `vibration`).
- Download calibrated sample CSV telemetry directly from the dashboard.
- Predict conditions for every reading with associated ML confidence score.
- Dedicated **Flagged Suspicious Telemetry Table** with specific root-cause diagnostic explanations (e.g., "Severe overcurrent (92.4A) & critical voltage sag (155.2V)", "Physics Invariant Violation: High current with near-zero voltage").
- Time-series scatter visualization with diamond anomaly overlays.
- One-click export of analyzed diagnostic reports to CSV.

### 3. 🔴 Live Grid Simulation
- Interactive cyber-physical telemetry simulator generating readings in real-time.
- Scenario modes: Mixed Operational Stream, Standard Nominal Baseline, Feeder Line Short-Circuit, Bushing Degradation, and False Data Injection.
- **Simplified Digital-Twin Inspired Sensor Comparison:** Displays theoretical expected values vs observed telemetry with dynamic delta metrics.
- Multi-trace streaming strip charts with upper/lower safety bounds.
- Real-time diagnostic alert banner.

### 4. 📊 Model Evaluation
- Real scikit-learn metrics computed on held-out test data (20% stratified test split).
- Actual test accuracy, macro and weighted Precision, Recall, and F1-scores.
- Interactive Plotly Confusion Matrix heatmap.
- Feature Importance ranking chart demonstrating cross-feature dependencies.
- Full classification report table.
- Academic limitations & assumptions analysis.

---

## 💻 Exact Windows Installation & Launch Commands

### Prerequisites
- Python 3.10+ (Python 3.10, 3.11, 3.12, 3.13, 3.14 supported).

### Step 1: Open PowerShell or Command Prompt
Navigate to the project directory:
```powershell
cd C:\Users\nandd\OneDrive\Desktop\GridGuardAI
```
*(or within VERIDEX: `cd C:\Users\nandd\OneDrive\Desktop\VERIDEX\GridGuardAI`)*

### Step 2: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3: (Optional) Re-generate Dataset or Re-train Model
```powershell
python src/generator.py
python src/train_model.py
```

### Step 4: Launch the Streamlit Dashboard
```powershell
streamlit run app.py
```
The application will automatically open in your default browser at:
`http://localhost:8501`

---

## 🧪 Verification & Automated Tests

To run the verification test suite:
```powershell
python -m unittest tests/test_grid_guard.py
```
Or run the diagnostic pipeline validation directly:
```powershell
python -c "from src.detection import GridGuardDetector; import pandas as pd; d = GridGuardDetector('models/grid_guard_rf.joblib'); df = pd.read_csv('data/grid_guard_dataset.csv'); res = d.analyze_dataframe(df); print('Analysis passed:', len(res), 'rows processed.')"
```

---

## 🎤 Demonstration Script for Faculty Presentation

When presenting this prototype to faculty or examiners, follow this 3-minute structured walkthrough:

1. **Introduction (30s):**
   - *"Good morning/afternoon. Today I present **Grid Guard AI**, an early detection prototype for hidden electrical infrastructure faults and sensor tampering in distribution grids."*
   - *"Modern grids face two simultaneous challenges: physical aging of equipment and cyber-physical tampering. Grid Guard AI combines machine learning with deterministic electrical physics invariants."*

2. **Overview Dashboard (45s):**
   - Open **⚡ Grid Overview**.
   - Point to the KPI cards: *"Notice our baseline distribution of 650 telemetry records. The system flags the overall health status and visualizes multi-metric sensor trends with IEEE 1159 safety thresholds."*

3. **Sensor Telemetry Analysis & Root-Cause Explanation (45s):**
   - Switch to **🔍 Analyze Sensor Data**.
   - Click **Download Sample CSV** or show the uploaded data preview.
   - Click **Run Grid Diagnostic Engine**.
   - Highlight the **Flagged Suspicious Telemetry** table: *"Unlike black-box models, Grid Guard AI provides explainable diagnostic notes: distinguishing whether a reading is an overcurrent physical fault, gradual thermal contact wear, or an impossible physics state caused by False Data Injection."*

4. **Live Grid Simulation & Digital-Twin Comparison (40s):**
   - Switch to **🔴 Live Grid Simulation**.
   - Select scenario *"Feeder Line Short-Circuit"* or *"False Data Injection"* and click **Run Simulation**.
   - Point to the **Simplified Digital-Twin Inspired Comparison**: *"Here we contrast theoretical nominal values against observed streaming sensor readings, immediately alerting operators to voltage sag and thermal surges."*

5. **Model Evaluation & Academic Limitations (20s):**
   - Switch to **📊 Model Evaluation**.
   - Highlight the Confusion Matrix and Feature Importance chart: *"All metrics shown here are strictly computed from actual held-out test predictions, not hardcoded values."*
   - Conclude by noting the educational synthetic-data boundaries stated in the transparency banner.

---

## 🎓 Academic Transparency Notice
*This is an educational prototype using synthetic sensor data. It demonstrates the concept of detecting normal operation, physical faults, degradation, and potentially manipulated readings. It has not been validated for operational use on real electrical infrastructure.*
