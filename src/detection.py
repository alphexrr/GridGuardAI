"""
GRID GUARD AI - Detection Engine
Combines a trained machine-learning model with deterministic physics-informed rules
and explainable diagnostic alerts.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List

CLASSES = ["normal", "physical_fault", "degradation", "data_manipulation"]

# Engineering feature extraction
def extract_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes derived electrical & thermo-mechanical physics features.
    """
    df = df.copy()
    v = df["voltage"].astype(float)
    i = df["current"].astype(float)
    t = df["temperature"].astype(float)
    vib = df["vibration"].astype(float)
    
    # Apparent power (kVA proxy): V * I / 1000
    df["apparent_power"] = (v * i) / 1000.0
    
    # Apparent impedance proxy: V / (I + epsilon)
    df["impedance_proxy"] = v / (np.abs(i) + 1e-3)
    
    # Thermal load intensity: T / (I + epsilon)
    df["temp_load_ratio"] = t / (np.abs(i) + 1e-3)
    
    # Mechanical vibration intensity relative to electrical power
    df["vib_power_ratio"] = vib / (df["apparent_power"] + 1e-3)
    
    return df

def analyze_sensor_rules(row: pd.Series, prev_row: pd.Series = None) -> Tuple[bool, str, str]:
    """
    Applies deterministic electrical grid rules and physics invariant checks.
    
    Returns:
        (is_flagged, rule_category, explanation_string)
    """
    v = float(row.get("voltage", 230.0))
    i = float(row.get("current", 20.0))
    t = float(row.get("temperature", 40.0))
    vib = float(row.get("vibration", 1.5))
    
    reasons = []
    
    # 1. Physics Invariant & False Data Injection Checks
    if v < 0 or i < 0 or vib < -0.1 or t < -30.0:
        return True, "data_manipulation", f"Out-of-bounds reading detected (V={v}V, I={i}A, T={t}°C, Vib={vib}mm/s) violating sensor calibration boundaries."
        
    if i > 25.0 and v < 50.0:
        return True, "data_manipulation", f"Physics Invariant Violation: Substantial load current ({i:.1f}A) with near-zero voltage ({v:.1f}V), indicating telemetry spoofing."
        
    if i <= 0.05 and t > 75.0:
        return True, "data_manipulation", f"Physics Invariant Violation: Thermal extreme ({t:.1f}°C) with zero electrical load current ({i:.1f}A)."

    # Flatline detection against previous row
    if prev_row is not None:
        pv = float(prev_row.get("voltage", -999))
        pi = float(prev_row.get("current", -999))
        pt = float(prev_row.get("temperature", -999))
        pvib = float(prev_row.get("vibration", -999))
        if abs(v - pv) < 1e-4 and abs(i - pi) < 1e-4 and abs(t - pt) < 1e-4 and abs(vib - pvib) < 1e-4:
            return True, "data_manipulation", "Telemetry Flatline: Sensor outputs remained exactly static with zero measurement noise."

    # 2. Critical Physical Fault Checks
    fault_flags = []
    if i >= 65.0:
        fault_flags.append(f"Severe overcurrent ({i:.1f}A >= 65.0A)")
    if v < 185.0:
        fault_flags.append(f"Critical voltage sag ({v:.1f}V < 185.0V)")
    elif v > 260.0:
        fault_flags.append(f"Dangerous overvoltage surge ({v:.1f}V > 260.0V)")
    if t >= 78.0:
        fault_flags.append(f"Thermal emergency ({t:.1f}°C >= 78.0°C)")
    if vib >= 5.5:
        fault_flags.append(f"Violent vibration spike ({vib:.2f}mm/s >= 5.5mm/s)")
        
    if fault_flags:
        return True, "physical_fault", " & ".join(fault_flags) + " [Physical Fault threshold breached]"

    # 3. Gradual Degradation Checks
    degrad_flags = []
    if 58.0 <= t < 78.0:
        degrad_flags.append(f"Elevated operating temperature ({t:.1f}°C)")
    if 3.2 <= vib < 5.5:
        degrad_flags.append(f"Increased mechanical looseness ({vib:.2f}mm/s)")
    if 195.0 <= v < 220.0:
        degrad_flags.append(f"Sub-nominal distribution voltage ({v:.1f}V)")
        
    if len(degrad_flags) >= 2 or (len(degrad_flags) >= 1 and (t >= 65.0 or vib >= 4.0)):
        return True, "degradation", " & ".join(degrad_flags) + " [Component wear / thermal decay pattern]"

    # 4. Normal Nominal
    return False, "normal", "All parameters within standard IEEE/IEC electrical distribution tolerances."


class GridGuardDetector:
    """
    Hybrid inference engine that orchestrates trained Machine Learning models
    with deterministic diagnostic rules and confidence scoring.
    """
    def __init__(self, model_path: str = None):
        self.model = None
        self.scaler = None
        self.feature_names = [
            "voltage", "current", "temperature", "vibration",
            "apparent_power", "impedance_proxy", "temp_load_ratio", "vib_power_ratio"
        ]
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            
    def load_model(self, model_path: str):
        """Loads serialized model dictionary."""
        data = joblib.load(model_path)
        self.model = data["model"]
        self.scaler = data.get("scaler", None)
        self.feature_names = data.get("features", self.feature_names)
        
    def analyze_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyzes a telemetry DataFrame, returning ML predictions, rule alerts,
        confidence percentages, and specific diagnostic explanations.
        """
        # Validate columns
        required = ["voltage", "current", "temperature", "vibration"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Required telemetry column '{col}' is missing.")
                
        processed = extract_physics_features(df)
        
        # If model is loaded, run ML prediction
        if self.model is not None:
            X = processed[self.feature_names]
            if self.scaler is not None:
                X_scaled = self.scaler.transform(X)
            else:
                X_scaled = X
                
            predictions = self.model.predict(X_scaled)
            if hasattr(self.model, "predict_proba"):
                probas = self.model.predict_proba(X_scaled)
                confidences = np.max(probas, axis=1) * 100.0
            else:
                confidences = np.ones(len(predictions)) * 95.0
        else:
            # Fallback if model not trained yet
            predictions = []
            confidences = []
            for _, r in df.iterrows():
                _, r_cat, _ = analyze_sensor_rules(r)
                predictions.append(r_cat)
                confidences.append(90.0)
            predictions = np.array(predictions)
            confidences = np.array(confidences)
            
        results = df.copy()
        results["predicted_condition"] = predictions
        results["confidence_pct"] = np.round(confidences, 1)
        
        # Apply rule-based diagnostics and explanations
        rule_flags = []
        rule_categories = []
        explanations = []
        
        prev_row = None
        for idx, row in df.iterrows():
            flagged, cat, expl = analyze_sensor_rules(row, prev_row)
            rule_flags.append(flagged)
            rule_categories.append(cat)
            
            # Formulate unified comprehensive explanation
            ml_pred = predictions[idx] if len(predictions) > idx else cat
            if ml_pred != "normal":
                combined_expl = f"[{ml_pred.upper()}] {expl}"
            else:
                combined_expl = expl
                
            explanations.append(combined_expl)
            prev_row = row
            
        results["rule_flagged"] = rule_flags
        results["rule_condition"] = rule_categories
        results["diagnostic_explanation"] = explanations
        results["is_abnormal"] = (results["predicted_condition"] != "normal") | (results["rule_flagged"])
        
        return results
