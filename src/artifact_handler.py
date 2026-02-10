import numpy as np
import pandas as pd


def compute_confidence(motion: pd.Series) -> pd.Series:
    """
    Compute continuous signal confidence based on motion intensity.
    Higher motion -> lower confidence.
    """
    confidence = np.exp(-motion)
    confidence = np.clip(confidence, 0.3, 1.0)
    return confidence


def clean_vitals(df: pd.DataFrame,
                 window: int = 5) -> pd.DataFrame:
    """
    Clean vital signals by handling artifacts and missing data.

    Steps:
    - Rolling median filter for HR & SpO2
    - Interpolate short missing segments
    - Compute confidence score
    """

    df = df.copy()

    # -----------------------------
    # 1. Rolling median filtering
    # -----------------------------
    df["heart_rate_clean"] = (
        df["heart_rate"]
        .rolling(window=window, center=True, min_periods=1)
        .median()
    )

    df["spo2_clean"] = (
        df["spo2"]
        .rolling(window=window, center=True, min_periods=1)
        .median()
    )

    # -----------------------------
    # 2. Interpolate missing values
    # -----------------------------
    df["heart_rate_clean"] = df["heart_rate_clean"].interpolate(
        method="linear", limit=5
    )

    df["spo2_clean"] = df["spo2_clean"].interpolate(
        method="linear", limit=5
    )

    # -----------------------------
    # 3. Confidence score
    # -----------------------------
    df["confidence"] = compute_confidence(df["motion"])

    return df


if __name__ == "__main__":
    # Manual test
    df = pd.read_csv("data/raw/patient_01.csv")
    df_clean = clean_vitals(df)

    df_clean.to_csv("data/processed/patient_01_clean.csv", index=False)