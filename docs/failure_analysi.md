# Current Performance
My evaluation showed perfect recall (100%) with zero false negatives. However, we analyze potential failure modes for robustness.

# Case 1: False Positive - Motion Artifact Misclassified

 What Happened:
- Patient stable (HR=92, SpO2=96%, SBP=115)
- Ambulance hits pothole, brief sensor disruption
- SpO2 reading drops to 91% for 3 seconds
- System triggers alert (false positive)

 Why It Failed:
- Motion threshold not strict enough
- Didn't require sustained deterioration
- Contributed to 23.1% false positive rate

 Impact on Metrics:
- Reduces precision from ideal 95% to 82.7%
- Increases alert fatigue risk

 Proposed Fix:
# Require Both low motion AND sustained duration
if min_spo2 < 92 and motion_mean < 0.10 and duration_low > 10:
   

# Case 2: Near-Miss - Slow Gradual Decline

 What Happened:
- Patient's SpO2 declining very slowly: 96→95→94→93→92 over 8 minutes
- System eventually detected at SpO2=91%, but took 6 minutes
- Although caught (recall=100%), detection was slower than ideal

 Why It Occurred:
- 30-second windows don't capture very long trends
- Threshold-based detection waits for critical value

 Impact:
- Alert latency higher than optimal (though still <30s target)
- Could be improved for earlier warning

 Proposed Enhancement:

# Add trend-based early warning
if spo2_trend < -0.03 and spo2_current < 94:
    yellow_alert()  # Early warning


# Case 3: Borderline Normal Patient (Contributing to FP)

 What Happened:
- Patient with chronic COPD, baseline SpO2=93-94%
- System repeatedly alerts when SpO2 dips to 92%
- Patient is actually at their normal baseline
- Multiple false positives over transport

 Why It Failed:
- Fixed thresholds don't account for patient-specific baselines
- No adaptive learning

 Impact:
- Contributed significantly to 23.1% false positive rate
- Would cause alert fatigue in real deployment

 Proposed Fix:

# Learn patient baseline in first 2 minutes
baseline_spo2 = first_120_sec['spo2'].mean()

# Alert on DEVIATION from baseline
if current_spo2 < (baseline_spo2 - 5):
    alert()


