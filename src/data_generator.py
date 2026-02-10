import numpy as np
import pandas as pd


def generate_patient_data(
    duration_sec: int = 1800,
    patient_id: int = 1,
    random_seed: int = None
) -> pd.DataFrame:
    """
    Generate synthetic patient vitals for a smart ambulance scenario.

    Parameters
    ----------
    duration_sec : int
        Total duration of simulation in seconds (default: 30 minutes)
    patient_id : int
        Identifier for the patient
    random_seed : int
        Optional seed for reproducibility

    Returns
    -------
    pd.DataFrame
        Time-series vitals sampled at 1 Hz
    """

    if random_seed is not None:
        np.random.seed(random_seed)

    t = np.arange(duration_sec)

    # -----------------------------
    # 1. Baseline physiological signals
    # -----------------------------
    hr = 80 + np.random.normal(0, 2, duration_sec)           # bpm
    spo2 = 98 + np.random.normal(0, 0.3, duration_sec)       # %
    sbp = 120 + np.random.normal(0, 3, duration_sec)         # mmHg
    dbp = 80 + np.random.normal(0, 2, duration_sec)          # mmHg

    # -----------------------------
    # 2. Motion / vibration signal
    # -----------------------------
    motion = np.random.normal(0.3, 0.15, duration_sec)

    # Random road bump events
    bump_indices = np.random.choice(
        duration_sec, size=int(0.03 * duration_sec), replace=False
    )
    motion[bump_indices] += np.random.uniform(1.5, 3.0, len(bump_indices))
    motion = np.clip(motion, 0, None)

    # -----------------------------
    # 3. Gradual deterioration event
    # -----------------------------
    if duration_sec > 600:
        start = np.random.randint(300, duration_sec - 300)
        length = np.random.randint(120, 300)

        spo2[start:start + length] -= np.linspace(0, 6, length)
        hr[start:start + length] += np.linspace(0, 20, length)

    # -----------------------------
    # 4. Sudden distress event
    # -----------------------------
    if duration_sec > 900:
        start = np.random.randint(600, duration_sec - 60)
        length = np.random.randint(30, 60)

        sbp[start:start + length] -= np.linspace(0, 25, length)
        dbp[start:start + length] -= np.linspace(0, 15, length)
        hr[start:start + length] += np.linspace(0, 30, length)

    # -----------------------------
    # 5. Motion-induced sensor artifacts
    # -----------------------------
    high_motion = motion > 1.5

    # SpO2 false drops during motion
    spo2[high_motion] -= np.random.uniform(2, 6, high_motion.sum())

    # HR false spikes during bumps
    hr[high_motion] += np.random.uniform(5, 15, high_motion.sum())

    # -----------------------------
    # 6. Missing data segments (sensor dropout)
    # -----------------------------
    for _ in range(3):
        start = np.random.randint(0, duration_sec - 10)
        length = np.random.randint(3, 8)

        hr[start:start + length] = np.nan
        spo2[start:start + length] = np.nan

    # -----------------------------
    # 7. Clip to physiological limits
    # -----------------------------
    hr = np.clip(hr, 30, 200)
    spo2 = np.clip(spo2, 70, 100)
    sbp = np.clip(sbp, 60, 200)
    dbp = np.clip(dbp, 40, 130)

    # -----------------------------
    # 8. Assemble DataFrame
    # -----------------------------
    df = pd.DataFrame({
        "time_sec": t,
        "patient_id": patient_id,
        "heart_rate": hr,
        "spo2": spo2,
        "sbp": sbp,
        "dbp": dbp,
        "motion": motion
    })

    return df


if __name__ == "__main__":
    # Example usage
    df = generate_patient_data(duration_sec=1800, patient_id=1, random_seed=42)
    df.to_csv("data/raw/patient_01.csv", index=False)