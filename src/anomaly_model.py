import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib


def train_anomaly_model(
    feature_df: pd.DataFrame,
    contamination: float = 0.05,
    random_state: int = 42
):
    """
    Train Isolation Forest for anomaly detection.
    """

    feature_cols = [
        "hr_mean",
        "hr_slope",
        "spo2_mean",
        "spo2_slope",
        "sbp_mean",
        "dbp_mean",
        "motion_mean",
        "confidence_mean",
    ]

    X = feature_df[feature_cols]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state
    )

    model.fit(X_scaled)

    return model, scaler, feature_cols


def predict_anomalies(
    model,
    scaler,
    feature_df: pd.DataFrame,
    feature_cols: list
) -> pd.DataFrame:
    """
    Predict anomaly scores and flags.
    """

    X = feature_df[feature_cols]
    X_scaled = scaler.transform(X)

    feature_df = feature_df.copy()
    feature_df["anomaly_score"] = -model.decision_function(X_scaled)
    feature_df["anomaly_flag"] = model.predict(X_scaled)

    # Convert IsolationForest output: -1 = anomaly, 1 = normal
    feature_df["anomaly_flag"] = feature_df["anomaly_flag"].map(
        {1: 0, -1: 1}
    )

    return feature_df


if __name__ == "__main__":
    import joblib
    import os

    os.makedirs("models", exist_ok=True)

    # Load features
    df_feat = pd.read_csv("data/processed/patient_01_features.csv")

    # Train model
    model, scaler, feature_cols = train_anomaly_model(df_feat)

    # Predict anomalies (optional, but keeps pipeline consistent)
    df_out = predict_anomalies(
        model,
        scaler,
        df_feat,
        feature_cols
    )

    df_out.to_csv(
        "data/processed/patient_01_anomalies.csv",
        index=False
    )

    # ✅ Save artifacts
    joblib.dump(model, "models/anomaly_model.joblib")
    joblib.dump(scaler, "models/scaler.joblib")
    joblib.dump(feature_cols, "models/feature_cols.joblib")