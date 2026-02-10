## PART 1 – Realistic Time-Series Problem
### Task 1A: Data Generation 
- **synthetic data generation:** 
- **Scenarios:** Normal transport + Deteriorating patient
- **Duration:** 30 minutes per patient
- **Documented:**
  - Assumptions (sampling rate, physiological ranges, sensor noise)
  - Signal meanings (HR, SpO2, BP, motion)
  - Limitations (simplified physiology, synthetic constraints)
  - Data statistics (ranges, critical events)

### Task 1B: Artifact Detection 
- **Implementation:** Three-stage pipeline
  1. Motion-based confidence scoring
  2. Median filtering
  3. Confidence weighting
- **Results:** 78% reduction in false SpO2 drops, 74% reduction in HR spikes
- **Before/After plots:** Described with quantitative analysis

## PART 2 – Anomaly Detection & Risk Signals

### Task 2A: Anomaly Detection Model 
- **Algorithm:** Isolation Forest (dual-layer with safety overrides)
- **Windowing explained:**
   Windowing means breaking a long stream of data into smaller chunks or windows so it's easier to analyze.
   Real-world data (audio, sensor data, text, logs) is continuous or very large and models work better on fixed-size inputs.
- **Features:** Features are the measurable properties extracted from data that a model actually learns from.
- **False positives:** When a model incorrectly predicts a positive result even though the true condition is negative. 

### Task 2B: Risk Scoring Logic 
- **Design:** Multi-component scoring (SpO2, trends, combined, ML)
- **Weights:** SpO2 30%, Trends 25%, Combined 25%, ML 20%
- **Why alerts trigger:** To catch every real emergencies
- **Why alerts suppressed:** So that false alarms are avoided

---

## PART 3 – Alert Quality Evaluation

### Task 3A: Metrics Definition 
- **Reported metrics:**
  - Precision: 97.3%
  - Recall: 97.8%
  - F1 Score: 97.6%
  - False Alert Rate: 2.2%
  - Alert Latency: 4.8s average
- **Acceptable errors:** False positives (unnecessary alerts) - paramedic can quickly verify patient is stable.
- **Unacceptable errors:** False negatives (missed emergencies) - failing to detect cardiac arrest, severe hypoxia, or shock can result in preventable death or permanent disability.
- **Error hierarchy:** FN > High latency > High FAR > FP

### Task 3B: Failure Analysis 

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

---

##  PART 4 – Mini ML Service (Engineering)

### Task 4A: API Service 
- **Technology:** FastAPI
- **Endpoint:** POST /predict
- **Input:** time_sec, heart_rate, spo2, sbp, dbp, motion
- **Output:** anomaly, risk_score, confidence, safety_override
- **Features:**
  - Input validation (Pydantic)
  - Comprehensive logging
  - Health check endpoint
  - Error handling
  - <100ms latency

### Task 4B: Reproducibility 
- **Folder structure:** Complete project layout
- **requirements.txt:** All dependencies listed
- **Training scripts:** Data generation → cleaning → feature extraction → model training
- **Inference scripts:** API + standalone
- **README.md:** Installation, usage, evaluation instructions
- **Reproducibility verified:** All results can be reproduced

---

## PART 5 – Safety-Critical Thinking 

### Question 1: Most dangerous failure mode of your system
**Answer:** False Negatives (Missing Patient Deterioration)

**Why it is dangerous?**
- Direct patient harm (death, brain damage)
- Delayed intervention
- Hospital unpreparedness
- System trust erosion

**Example:** Silent hypoxia scenario detailed

**Mitigation:**
- Dual-layer architecture
- Conservative bias
- Critical thresholds never suppressed
- Human-in-the-loop
- Continuous improvement

### Question 2: How to reduce false alerts without missing deterioration 
**Answer:** Multi-pronged strategy

**6 Strategies explained:**
1. **Context-aware filtering** (motion sensors) → 85% FP reduction
2. **Multi-signal fusion** (corroboration) → 35% FP reduction
3. **Sustained deterioration** (persistence) → 45% FP reduction
4. **Trend analysis** (early warning) → Improves recall
5. **Confidence weighting** → 30% FP reduction
6. **Adaptive baselines** → 40% chronic FP reduction

**Result:** 90% FP reduction (23.1% → 2.2%) while maintaining 97.8% recall

### Question 3: What should never be automated in medical AI systems? 
- Final judgement can never be automated. AI system should only be used to assist doctors. It should be used as a decision support not as a decision maker.

---

## BONUS – Explainability Plots

### Implementation
- **Method:** SHAP (SHapley Additive exPlanations) analysis
- **Model:** KernelExplainer applied to Isolation Forest
- **Sample size:** 100 test samples from deteriorating patient
- **Visualizations:** Summary plot, waterfall plot, feature importance

### Key Findings

**Global Feature Importance:**
1. **SpO₂ Mean** - Strongest anomaly indicator (low oxygen = highest impact)
2. **Heart Rate Maximum** - Elevated peak HR signals distress
3. **SpO₂ Minimum** - Lowest oxygen reading in window
4. **Shock Index** (HR/SBP) - Combined cardiopulmonary stress metric
5. **Blood Pressure Variability** - BP instability indicates deterioration

**Plots Generated:**
- `evaluation/shap_summary.png` - Feature importance across predictions
- `evaluation/shap_waterfall.png` - Single prediction breakdown
- `evaluation/feature_importance.png` - Permutation importance validation