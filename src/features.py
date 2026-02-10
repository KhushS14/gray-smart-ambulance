import numpy as np
import pandas as pd


def compute_slope(series: pd.Series) -> float:
    """
    Compute linear trend (slope) of a time series.
    """
    y = series.values
    x = np.arange(len(y))
    if len(y) < 2 or np.all(np.isnan(y)):
        return 0.0
    return np.polyfit(x, y, 1)[0]


def extract_window_features(
    df: pd.DataFrame,
    window_size: int = 30,
    step_size: int = 5
) -> pd.DataFrame:
    """
    Extract sliding-window features for anomaly detection.
    
    Now handles small datasets by using adaptive window sizes.
    Minimum window size is 3 seconds.

    Features:
    - Mean HR
    - HR trend (slope)
    - Mean SpO2
    - SpO2 trend
    - Mean BP
    - Motion intensity
    - Mean confidence
    """
    
    if df.empty:
        return pd.DataFrame()
    
    features = []
    
    # Adaptive window size: use what we have, minimum 3 seconds
    actual_window_size = min(window_size, len(df))
    
    if actual_window_size < 3:
        # Not enough data for meaningful features
        return pd.DataFrame()
    
    # If we have less data than the window, extract one feature from all data
    if len(df) <= window_size:
        window = df
        
        feat = {
            "time_sec": window["time_sec"].iloc[-1],
            "hr_mean": window["heart_rate_clean"].mean(),
            "hr_slope": compute_slope(window["heart_rate_clean"]),
            "spo2_mean": window["spo2_clean"].mean(),
            "spo2_slope": compute_slope(window["spo2_clean"]),
            "sbp_mean": window["sbp"].mean(),
            "dbp_mean": window["dbp"].mean(),
            "motion_mean": window["motion"].mean(),
            "confidence_mean": window["confidence"].mean(),
        }
        
        features.append(feat)
    else:
        # Standard sliding window approach
        for start in range(0, len(df) - window_size + 1, step_size):
            window = df.iloc[start:start + window_size]

            feat = {
                "time_sec": window["time_sec"].iloc[-1],
                "hr_mean": window["heart_rate_clean"].mean(),
                "hr_slope": compute_slope(window["heart_rate_clean"]),
                "spo2_mean": window["spo2_clean"].mean(),
                "spo2_slope": compute_slope(window["spo2_clean"]),
                "sbp_mean": window["sbp"].mean(),
                "dbp_mean": window["dbp"].mean(),
                "motion_mean": window["motion"].mean(),
                "confidence_mean": window["confidence"].mean(),
            }

            features.append(feat)

    return pd.DataFrame(features)


if __name__ == "__main__":
    df = pd.read_csv("data/processed/patient_01_clean.csv")
    feature_df = extract_window_features(df)

    feature_df.to_csv(
        "data/processed/patient_01_features.csv",
        index=False
    )