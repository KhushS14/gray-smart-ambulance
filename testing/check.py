import pandas as pd

# Load data
df_clean = pd.read_csv("data/processed/patient_01_clean.csv")
df_risk = pd.read_csv("data/processed/patient_01_risk.csv")

print("=" * 60)
print("DATA DIAGNOSTIC")
print("=" * 60)

# Check predictions
print(f"\nPredictions (df_risk['alert']):")
print(f"  Total windows: {len(df_risk)}")
print(f"  Alerts (1): {df_risk['alert'].sum()}")
print(f"  Normal (0): {(df_risk['alert'] == 0).sum()}")
print(f"  Alert rate: {df_risk['alert'].mean():.1%}")

# Check vitals ranges
print(f"\nVital Signs Ranges (df_clean):")
print(f"  SpO2: {df_clean['spo2_clean'].min():.1f} - {df_clean['spo2_clean'].max():.1f}")
print(f"  HR:   {df_clean['heart_rate_clean'].min():.1f} - {df_clean['heart_rate_clean'].max():.1f}")
print(f"  SBP:  {df_clean['sbp'].min():.1f} - {df_clean['sbp'].max():.1f}")

# Check if there SHOULD be anomalies
critical_spo2 = (df_clean['spo2_clean'] < 90).sum()
critical_hr = (df_clean['heart_rate_clean'] > 140).sum()
critical_sbp = (df_clean['sbp'] < 90).sum()

print(f"\nExpected Anomalies in Raw Data:")
print(f"  SpO2 < 90%: {critical_spo2} records ({critical_spo2/len(df_clean)*100:.1f}%)")
print(f"  HR > 140:   {critical_hr} records ({critical_hr/len(df_clean)*100:.1f}%)")
print(f"  SBP < 90:   {critical_sbp} records ({critical_sbp/len(df_clean)*100:.1f}%)")

# Check risk scores
if 'risk_score' in df_risk.columns:
    print(f"\nRisk Scores:")
    print(f"  Min: {df_risk['risk_score'].min():.1f}")
    print(f"  Max: {df_risk['risk_score'].max():.1f}")
    print(f"  Mean: {df_risk['risk_score'].mean():.1f}")
    print(f"  >50: {(df_risk['risk_score'] > 50).sum()}")
    print(f"  >70: {(df_risk['risk_score'] > 70).sum()}")

print("\n" + "=" * 60)