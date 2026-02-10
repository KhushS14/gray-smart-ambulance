from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import logging

from src.artifact_handler import clean_vitals
from src.features import extract_window_features
from src.anomaly_model import predict_anomalies
from src.risk_score import compute_risk_score

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Smart Ambulance Early Warning API",
    description="Detects patient deterioration risk from streaming vitals",
    version="1.0"
)


# -----------------------------
# Input schema
# -----------------------------
class VitalsInput(BaseModel):
    time_sec: list[int]
    heart_rate: list[float]
    spo2: list[float]
    sbp: list[float]
    dbp: list[float]
    motion: list[float]


# -----------------------------
# Load trained model artifacts
# -----------------------------
MODEL_PATH = "models/anomaly_model.joblib"
SCALER_PATH = "models/scaler.joblib"
FEATURES_PATH = "models/feature_cols.joblib"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_cols = joblib.load(FEATURES_PATH)


# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict(vitals: VitalsInput):
    logger.info("🔥 Received prediction request")
    
    df = pd.DataFrame(vitals.dict())
    logger.info(f"Input shape: {df.shape}, HR range: [{df['heart_rate'].min():.1f}, {df['heart_rate'].max():.1f}], "
                f"SpO2 range: [{df['spo2'].min():.1f}, {df['spo2'].max():.1f}]")
    
    df_clean = clean_vitals(df)
    logger.info(f"After cleaning: {len(df_clean)} rows (removed {len(df) - len(df_clean)})")
    
    # 🚑 SAFETY OVERRIDE - CHECK FIRST (before feature extraction)
    min_spo2 = df_clean["spo2"].min() if not df_clean.empty else 100
    max_hr = df_clean["heart_rate"].max() if not df_clean.empty else 60
    min_sbp = df_clean["sbp"].min() if not df_clean.empty else 120
    max_sbp = df_clean["sbp"].max() if not df_clean.empty else 120
    
    safety_triggered = False
    safety_score = 0.0
    
    # Critical thresholds
    if min_spo2 < 90:
        safety_score = max(safety_score, 95.0)
        safety_triggered = True
        logger.warning(f"⚠️ CRITICAL: SpO2 = {min_spo2:.1f} < 90")
    elif min_spo2 < 92:
        safety_score = max(safety_score, 80.0)
        safety_triggered = True
        logger.warning(f"⚠️ WARNING: SpO2 = {min_spo2:.1f} < 92")
    
    if max_hr > 140:
        safety_score = max(safety_score, 95.0)
        safety_triggered = True
        logger.warning(f"⚠️ CRITICAL: HR = {max_hr:.1f} > 140")
    elif max_hr > 120:
        safety_score = max(safety_score, 75.0)
        safety_triggered = True
        logger.warning(f"⚠️ WARNING: HR = {max_hr:.1f} > 120")
    
    if min_sbp < 90:
        safety_score = max(safety_score, 95.0)
        safety_triggered = True
        logger.warning(f"⚠️ CRITICAL: SBP = {min_sbp:.1f} < 90")
    elif min_sbp < 100:
        safety_score = max(safety_score, 75.0)
        safety_triggered = True
        logger.warning(f"⚠️ WARNING: SBP = {min_sbp:.1f} < 100")
    
    if max_sbp > 180:
        safety_score = max(safety_score, 90.0)
        safety_triggered = True
        logger.warning(f"⚠️ CRITICAL: SBP = {max_sbp:.1f} > 180")
    
    if min_spo2 < 94 and max_hr > 110:
        safety_score = max(safety_score, 85.0)
        safety_triggered = True
        logger.warning(f"⚠️ COMBINED RISK: Low SpO2 ({min_spo2:.1f}) + High HR ({max_hr:.1f})")
    
    # If safety triggered, return immediately
    if safety_triggered:
        logger.info(f"✅ Safety override triggered with score: {safety_score:.1f}")
        return {
            "anomaly": True,
            "risk_score": float(safety_score),
            "confidence": float(df_clean["confidence"].mean()) if not df_clean.empty else 0.0,
            "safety_override": True
        }
    
    # Otherwise, try model-based prediction
    feat_df = extract_window_features(df_clean)
    logger.info(f"Extracted {len(feat_df)} feature windows")

    if feat_df.empty:
        logger.warning("No features extracted - returning safe default")
        return {
            "anomaly": False,
            "risk_score": 0.0,
            "confidence": float(df_clean["confidence"].mean()) if not df_clean.empty else 0.0,
            "safety_override": False
        }

    feat_out = predict_anomalies(model, scaler, feat_df, feature_cols)
    risk_out = compute_risk_score(feat_out)
    latest = risk_out.iloc[-1]
    
    logger.info(f"Model output - Anomaly: {latest['anomaly_flag']}, Risk: {latest['risk_score']:.1f}")

    return {
        "anomaly": bool(latest["anomaly_flag"]),
        "risk_score": float(latest["risk_score"]),
        "confidence": float(latest["confidence_mean"]),
        "safety_override": False
    }