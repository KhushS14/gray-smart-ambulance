"""
Generate test patient with realistic deterioration for evaluation
"""
import numpy as np
import pandas as pd
import sys
sys.path.append('.')

from src.artifact_handler import clean_vitals
from src.features import extract_window_features
from src.anomaly_model import predict_anomalies
from src.risk_score import compute_risk_score
import joblib

np.random.seed(42)

def generate_deteriorating_patient(duration=1800, patient_id=99):
    """
    Generate 30-minute transport with progressive deterioration
    
    Timeline:
    - 0-600s (0-10 min): Normal, stable transport
    - 600-900s (10-15 min): Gradual deterioration begins
    - 900-1200s (15-20 min): Critical deterioration
    - 1200-1800s (20-30 min): Critical, stabilizing with intervention
    """
    times = np.arange(duration)
    
    # Heart Rate
    hr_normal = np.random.normal(80, 5, 600)
    hr_rising = np.linspace(80, 145, 300) + np.random.normal(0, 3, 300)
    hr_critical = np.random.normal(148, 5, 300)
    hr_stabilizing = np.linspace(148, 120, 600) + np.random.normal(0, 4, 600)
    heart_rate = np.concatenate([hr_normal, hr_rising, hr_critical, hr_stabilizing])
    
    # SpO2
    spo2_normal = np.random.normal(97, 1, 600)
    spo2_declining = np.linspace(97, 87, 300) + np.random.normal(0, 1, 300)
    spo2_critical = np.random.normal(85, 2, 300)
    spo2_recovering = np.linspace(85, 93, 600) + np.random.normal(0, 1.5, 600)
    spo2 = np.concatenate([spo2_normal, spo2_declining, spo2_critical, spo2_recovering])
    
    # Blood Pressure (declines during deterioration)
    sbp_normal = np.random.normal(120, 8, 600)
    sbp_declining = np.linspace(120, 85, 300) + np.random.normal(0, 5, 300)
    sbp_critical = np.random.normal(82, 4, 300)
    sbp_recovering = np.linspace(82, 105, 600) + np.random.normal(0, 5, 600)
    sbp = np.concatenate([sbp_normal, sbp_declining, sbp_critical, sbp_recovering])
    
    dbp_normal = np.random.normal(80, 5, 600)
    dbp_declining = np.linspace(80, 55, 300) + np.random.normal(0, 3, 300)
    dbp_critical = np.random.normal(52, 3, 300)
    dbp_recovering = np.linspace(52, 68, 600) + np.random.normal(0, 3, 600)
    dbp = np.concatenate([dbp_normal, dbp_declining, dbp_critical, dbp_recovering])
    
    # Motion (higher during critical period due to paramedic intervention)
    motion_normal = np.random.uniform(0.05, 0.25, 600)
    motion_intervention = np.random.uniform(0.15, 0.4, 600)  # Increased activity
    motion_stable = np.random.uniform(0.05, 0.2, 600)
    motion = np.concatenate([motion_normal, motion_intervention, motion_stable])
    
    # Ensure no negative values
    spo2 = np.clip(spo2, 70, 100)
    heart_rate = np.clip(heart_rate, 60, 180)
    sbp = np.clip(sbp, 70, 180)
    dbp = np.clip(dbp, 40, 120)
    
    df = pd.DataFrame({
        'time_sec': times,
        'patient_id': patient_id,
        'heart_rate': heart_rate,
        'spo2': spo2,
        'sbp': sbp,
        'dbp': dbp,
        'motion': motion
    })
    
    return df


def create_ground_truth_labels(df):
    """
    Label true anomalies based on critical thresholds
    """
    labels = []
    for _, row in df.iterrows():
        # Critical if any of these conditions met
        is_critical = (
            row['spo2'] < 90 or
            row['heart_rate'] > 140 or
            row['sbp'] < 90 or
            (row['spo2'] < 92 and row['heart_rate'] > 120)  # Combined risk
        )
        labels.append(1 if is_critical else 0)
    
    df['true_label'] = labels
    return df


if __name__ == "__main__":
    print("=" * 70)
    print("GENERATING TEST PATIENT WITH DETERIORATION")
    print("=" * 70)
    
    # Generate patient
    print("\n1. Generating 30-min deteriorating patient...")
    df_raw = generate_deteriorating_patient(duration=1800, patient_id=99)
    
    # Add ground truth labels
    df_raw = create_ground_truth_labels(df_raw)
    
    print(f"   ✅ Generated {len(df_raw)} seconds of vitals")
    print(f"\n   Vital Ranges:")
    print(f"   SpO2: {df_raw['spo2'].min():.1f} - {df_raw['spo2'].max():.1f}%")
    print(f"   HR:   {df_raw['heart_rate'].min():.1f} - {df_raw['heart_rate'].max():.1f} bpm")
    print(f"   SBP:  {df_raw['sbp'].min():.1f} - {df_raw['sbp'].max():.1f} mmHg")
    print(f"\n   Critical Periods:")
    print(f"   SpO2 < 90%: {(df_raw['spo2'] < 90).sum()} seconds ({(df_raw['spo2'] < 90).sum()/len(df_raw)*100:.1f}%)")
    print(f"   HR > 140:   {(df_raw['heart_rate'] > 140).sum()} seconds ({(df_raw['heart_rate'] > 140).sum()/len(df_raw)*100:.1f}%)")
    print(f"   SBP < 90:   {(df_raw['sbp'] < 90).sum()} seconds ({(df_raw['sbp'] < 90).sum()/len(df_raw)*100:.1f}%)")
    print(f"   True anomalies: {df_raw['true_label'].sum()} seconds ({df_raw['true_label'].sum()/len(df_raw)*100:.1f}%)")
    
    # Save raw data
    df_raw.to_csv("data/test/patient_99_deteriorating_raw.csv", index=False)
    print("\n2. ✅ Saved raw data to data/test/patient_99_deteriorating_raw.csv")
    
    # Clean vitals
    print("\n3. Cleaning vitals...")
    df_clean = clean_vitals(df_raw)
    df_clean.to_csv("data/test/patient_99_deteriorating_clean.csv", index=False)
    print(f"   ✅ Cleaned {len(df_clean)} records")
    print(f"   ✅ Saved to data/test/patient_99_deteriorating_clean.csv")
    
    # Extract features
    print("\n4. Extracting features...")
    feat_df = extract_window_features(df_clean, window_size=30, step_size=5)
    feat_df.to_csv("data/test/patient_99_deteriorating_features.csv", index=False)
    print(f"   ✅ Extracted {len(feat_df)} feature windows")
    print(f"   ✅ Saved to data/test/patient_99_deteriorating_features.csv")
    
    # Run anomaly detection
    print("\n5. Running anomaly detection...")
    model = joblib.load("models/anomaly_model.joblib")
    scaler = joblib.load("models/scaler.joblib")
    feature_cols = joblib.load("models/feature_cols.joblib")
    
    feat_out = predict_anomalies(model, scaler, feat_df, feature_cols)
    print(f"   ✅ Anomaly flags: {feat_out['anomaly_flag'].sum()} / {len(feat_out)}")
    
    # Compute risk scores
    print("\n6. Computing risk scores...")
    risk_df = compute_risk_score(feat_out)
    risk_df.to_csv("data/test/patient_99_deteriorating_risk.csv", index=False)
    print(f"   ✅ Risk scores computed")
    print(f"   ✅ Saved to data/test/patient_99_deteriorating_risk.csv")
    
    print(f"\n   Risk Score Stats:")
    print(f"   Min: {risk_df['risk_score'].min():.1f}")
    print(f"   Max: {risk_df['risk_score'].max():.1f}")
    print(f"   Mean: {risk_df['risk_score'].mean():.1f}")
    print(f"   Alerts: {risk_df['alert'].sum()} / {len(risk_df)} windows")
    
    print("\n" + "=" * 70)
    print("✅ TEST PATIENT GENERATED SUCCESSFULLY")
    print("=" * 70)
    print("\nNext step: Run evaluation on this deteriorating patient:")
    print("  python metrics_enhanced.py")
    print("\nMake sure to update metrics_enhanced.py to use:")
    print("  df_clean = pd.read_csv('data/test/patient_99_deteriorating_clean.csv')")
    print("  df_risk = pd.read_csv('data/test/patient_99_deteriorating_risk.csv')")
    print("=" * 70)