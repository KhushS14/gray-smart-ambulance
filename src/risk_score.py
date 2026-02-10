import numpy as np
import pandas as pd


def normalize(series: pd.Series) -> pd.Series:
    """
    Normalize safely. If only one value, return it scaled.
    """
    if len(series) == 1:
        return pd.Series([1.0], index=series.index)

    return (series - series.min()) / (series.max() - series.min() + 1e-6)


def compute_risk_score(
    df: pd.DataFrame,
    anomaly_weight: float = 0.5,
    trend_weight: float = 0.3,
    spo2_weight: float = 0.2,
    confidence_threshold: float = 0.5,
    min_persistence: int = 2
) -> pd.DataFrame:
    """
    Compute final risk score and alert flag.

    Logic:
    - Combine anomaly score, trends, and absolute SpO₂
    - Suppress alerts during low confidence
    - Require persistence to avoid noise
    """

    df = df.copy()

    # -----------------------------
    # 1. Normalize components
    # -----------------------------
    df["anomaly_norm"] = normalize(df["anomaly_score"])
    df["spo2_risk"] = normalize(100 - df["spo2_mean"])
    df["trend_risk"] = normalize(
        np.abs(df["spo2_slope"]) + np.abs(df["hr_slope"])
    )

    # -----------------------------
    # 2. Composite risk score
    # -----------------------------
    df["risk_score"] = (
        anomaly_weight * df["anomaly_norm"] +
        trend_weight * df["trend_risk"] +
        spo2_weight * df["spo2_risk"]
    )

    # Scale to 0–100
    df["risk_score"] = (df["risk_score"] * 100).clip(0, 100)

    # -----------------------------
    # 3. Alert suppression logic
    # -----------------------------
    df["alert_raw"] = (
        (df["risk_score"] > 50) &
        (df["confidence_mean"] >= confidence_threshold)
    ).astype(int)

    # Require alert persistence
    df["alert"] = (
        df["alert_raw"]
        .rolling(window=min_persistence, min_periods=1)
        .sum() >= min_persistence
    ).astype(int)
    print("DEBUG: normalize len =", len(df))

    return df


if __name__ == "__main__":
    df = pd.read_csv("data/processed/patient_01_anomalies.csv")

    df_risk = compute_risk_score(df)

    df_risk.to_csv(
        "data/processed/patient_01_risk.csv",
        index=False
    )
