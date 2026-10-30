"""
GRID GUARD AI - Synthetic Sensor Data Generator
Generates realistic electrical grid distribution telemetry:
1. normal (stable voltage, load-correlated current, safe temp, low vibration)
2. physical_fault (voltage sag/surge, fault overcurrent, thermal surge, violent vibration)
3. degradation (creeping contact resistance, insulation decay, bearing looseness)
4. data_manipulation (frozen flatlines, physics invariant violations, synthetic step injections)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

CLASSES = ["normal", "physical_fault", "degradation", "data_manipulation"]

def generate_synthetic_sensor_data(n_samples: int = 650, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a reproducible, structured electrical telemetry dataset.
    
    Parameters:
        n_samples (int): Total number of sensor readings to generate.
        random_state (int): Seed for reproducibility.
        
    Returns:
        pd.DataFrame: Telemetry with timestamp, voltage, current, temperature, vibration, condition.
    """
    np.random.seed(random_state)
    start_time = datetime(2026, 10, 23, 8, 0, 0)
    timestamps = [start_time + timedelta(minutes=i) for i in range(n_samples)]
    
    # Class allocations: ~55% normal, ~15% physical_fault, ~15% degradation, ~15% data_manipulation
    n_normal = int(n_samples * 0.55)
    n_fault = int(n_samples * 0.15)
    n_degrad = int(n_samples * 0.15)
    n_manip = n_samples - (n_normal + n_fault + n_degrad)
    
    records = []
    
    # --- 1. NORMAL OPERATION ---
    for i in range(n_normal):
        # Nominal 230V LV distribution, standard diurnal load variation
        hour_factor = np.sin((i / 60.0) * np.pi) * 2.0
        v = float(np.random.normal(230.0 + hour_factor, 1.2))
        i_curr = float(np.random.normal(24.0 + hour_factor * 1.5, 2.5))
        i_curr = max(8.0, i_curr)
        # Joule heating: T proportional to current squared with ambient baseline ~38C
        temp = float(38.0 + (i_curr ** 1.3) * 0.18 + np.random.normal(0, 0.8))
        # Nominal 50Hz hum mechanical vibration: 0.8 - 2.2 mm/s
        vib = float(np.random.normal(1.4, 0.25))
        vib = max(0.4, vib)
        
        records.append({
            "voltage": round(v, 2),
            "current": round(i_curr, 2),
            "temperature": round(temp, 2),
            "vibration": round(vib, 2),
            "condition": "normal"
        })
        
    # --- 2. PHYSICAL FAULT ---
    for i in range(n_fault):
        fault_type = i % 3
        if fault_type == 0:
            # Symmetrical short-circuit / feeder ground fault: heavy voltage sag, massive fault current
            v = float(np.random.uniform(130.0, 175.0))
            i_curr = float(np.random.uniform(75.0, 135.0))
            temp = float(np.random.uniform(76.0, 105.0))
            vib = float(np.random.uniform(5.5, 11.0))
        elif fault_type == 1:
            # Overvoltage surge + dielectric breakdown flashover
            v = float(np.random.uniform(265.0, 295.0))
            i_curr = float(np.random.uniform(55.0, 95.0))
            temp = float(np.random.uniform(72.0, 98.0))
            vib = float(np.random.uniform(4.8, 9.5))
        else:
            # Severe cooling pump failure / transformer core shudder
            v = float(np.random.normal(224.0, 3.0))
            i_curr = float(np.random.uniform(45.0, 65.0))
            temp = float(np.random.uniform(85.0, 115.0))
            vib = float(np.random.uniform(6.5, 12.5))
            
        records.append({
            "voltage": round(v, 2),
            "current": round(i_curr, 2),
            "temperature": round(temp, 2),
            "vibration": round(vib, 2),
            "condition": "physical_fault"
        })
        
    # --- 3. DEGRADATION ---
    for i in range(n_degrad):
        # Progressively worsening contact resistance, loose bushings, thermal creep
        progress = i / max(1, n_degrad)
        v = float(228.0 - (progress * 12.0) + np.random.normal(0, 1.5))
        i_curr = float(25.0 + (progress * 10.0) + np.random.normal(0, 1.8))
        temp = float(58.0 + (progress * 16.0) + np.random.normal(0, 1.2)) # 58C -> 74C
        vib = float(3.2 + (progress * 2.2) + np.random.normal(0, 0.35))   # 3.2 -> 5.4 mm/s
        
        records.append({
            "voltage": round(v, 2),
            "current": round(i_curr, 2),
            "temperature": round(temp, 2),
            "vibration": round(vib, 2),
            "condition": "degradation"
        })
        
    # --- 4. DATA MANIPULATION ---
    for i in range(n_manip):
        manip_type = i % 3
        if manip_type == 0:
            # Frozen telemetry / bypass flatline: constant invariant float values
            v = 230.00
            i_curr = 20.00
            temp = 42.00
            vib = 1.20
        elif manip_type == 1:
            # Physics invariant violation: high current while voltage is dead, or zero current while burning hot
            if i % 2 == 0:
                v = float(np.random.uniform(5.0, 18.0))   # Almost zero voltage
                i_curr = float(np.random.uniform(35.0, 60.0)) # Impossible high power consumption
                temp = float(np.random.uniform(60.0, 75.0))
                vib = float(np.random.uniform(1.2, 2.0))
            else:
                v = float(np.random.uniform(228.0, 234.0))
                i_curr = 0.00 # Zero load current
                temp = float(np.random.uniform(88.0, 110.0)) # Yet boiling temperature!
                vib = float(np.random.uniform(0.1, 0.4))
        else:
            # Unphysical negative or out-of-scale sensor artifacts
            v = float(np.random.uniform(220.0, 240.0))
            i_curr = float(np.random.uniform(15.0, 30.0))
            temp = float(np.random.choice([-15.0, 180.0, 25.0])) # Unnatural sensor jump
            vib = float(np.random.choice([-1.5, 0.0, 18.5]))     # Negative or uncalibrated reading
            
        records.append({
            "voltage": round(v, 2),
            "current": round(i_curr, 2),
            "temperature": round(temp, 2),
            "vibration": round(vib, 2),
            "condition": "data_manipulation"
        })
        
    # Shuffle into realistic operational order while keeping timestamps chronological
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    shuffled_records = [records[idx] for idx in indices]
    
    df = pd.DataFrame(shuffled_records)
    df["timestamp"] = [t.strftime("%Y-%m-%d %H:%M:%S") for t in timestamps]
    
    # Reorder columns: timestamp first
    cols = ["timestamp", "voltage", "current", "temperature", "vibration", "condition"]
    return df[cols]

if __name__ == "__main__":
    import os
    dataset = generate_synthetic_sensor_data(650, random_state=42)
    os.makedirs("data", exist_ok=True)
    dataset.to_csv("data/grid_guard_dataset.csv", index=False)
    print(f"Generated {len(dataset)} records in data/grid_guard_dataset.csv")
    print(dataset["condition"].value_counts())
