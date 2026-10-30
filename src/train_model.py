"""
GRID GUARD AI - Model Training & Evaluation Pipeline
Trains a baseline scikit-learn RandomForestClassifier on synthetic electrical telemetry,
evaluating strictly on held-out test data.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

# Relative / local imports
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from generator import generate_synthetic_sensor_data, CLASSES
from detection import extract_physics_features

FEATURE_COLS = [
    "voltage", "current", "temperature", "vibration",
    "apparent_power", "impedance_proxy", "temp_load_ratio", "vib_power_ratio"
]

def train_and_evaluate(
    data_path: str = None,
    n_samples: int = 750,
    random_state: int = 42,
    save_model_path: str = None
) -> dict:
    """
    Trains a reproducible baseline model and computes honest evaluation metrics on held-out test data.
    """
    # 1. Load or generate training dataset
    if data_path and os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        df = generate_synthetic_sensor_data(n_samples=n_samples, random_state=random_state)
        
    df_feat = extract_physics_features(df)
    
    X = df_feat[FEATURE_COLS]
    y = df_feat["condition"]
    
    # 2. Stratified train/test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=random_state,
        stratify=y
    )
    
    # 3. Fit Scaler & RandomForest
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    clf = RandomForestClassifier(
        n_estimators=120,
        max_depth=12,
        min_samples_split=3,
        random_state=random_state,
        class_weight="balanced"
    )
    clf.fit(X_train_scaled, y_train)
    
    # 4. Strictly evaluate on held-out test data
    y_pred = clf.predict(X_test_scaled)
    y_proba = clf.predict_proba(X_test_scaled)
    
    acc = float(accuracy_score(y_test, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )
    
    clf_report_dict = classification_report(
        y_test, y_pred, labels=CLASSES, target_names=CLASSES, output_dict=True, zero_division=0
    )
    clf_report_text = classification_report(
        y_test, y_pred, labels=CLASSES, target_names=CLASSES, zero_division=0
    )
    
    cm = confusion_matrix(y_test, y_pred, labels=CLASSES)
    
    feature_importances = dict(zip(FEATURE_COLS, [float(x) for x in clf.feature_importances_]))
    
    results = {
        "accuracy": acc,
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(p_weighted),
        "recall_weighted": float(r_weighted),
        "f1_weighted": float(f1_weighted),
        "classification_report": clf_report_dict,
        "classification_report_text": clf_report_text,
        "confusion_matrix": cm.tolist(),
        "classes": CLASSES,
        "feature_importances": feature_importances,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "test_labels": y_test.tolist(),
        "test_preds": y_pred.tolist(),
    }
    
    # 5. Save model bundle
    if save_model_path:
        os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
        joblib.dump({
            "model": clf,
            "scaler": scaler,
            "features": FEATURE_COLS,
            "classes": CLASSES,
            "metrics": results
        }, save_model_path)
        print(f"Model and evaluation metrics successfully saved to: {save_model_path}")
        
    return results

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file = os.path.join(base_dir, "data", "grid_guard_dataset.csv")
    model_file = os.path.join(base_dir, "models", "grid_guard_rf.joblib")
    
    print("Training Grid Guard AI baseline classifier...")
    eval_res = train_and_evaluate(
        data_path=data_file if os.path.exists(data_file) else None,
        n_samples=750,
        random_state=42,
        save_model_path=model_file
    )
    print("\n--- Model Evaluation Results (Held-Out Test Set) ---")
    print(f"Accuracy:  {eval_res['accuracy'] * 100:.2f}%")
    print(f"F1-Macro:  {eval_res['f1_macro']:.4f}")
    print(f"Precision: {eval_res['precision_macro']:.4f}")
    print(f"Recall:    {eval_res['recall_macro']:.4f}")
    print("\nClassification Report:\n", eval_res['classification_report_text'])
