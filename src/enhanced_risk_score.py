"""
Enhanced Risk Scoring System with Precision Improvements
Reduces false alarms while maintaining high recall
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict


class TemporalAlertBuffer:
    """
    Temporal consistency checker - requires sustained anomalies
    """
    def __init__(self, buffer_size: int = 3):
        """
        Args:
            buffer_size: Number of consecutive windows required for alert
        """
        self.buffer_size = buffer_size
        self.recent_alerts = []
    
    def should_alert(self, current_prediction: bool) -> bool:
        """
        Returns True only if last N predictions were all alerts
        """
        self.recent_alerts.append(current_prediction)
        
        # Keep only last N predictions
        if len(self.recent_alerts) > self.buffer_size:
            self.recent_alerts.pop(0)
        
        # Alert only if ALL recent predictions are positive
        if len(self.recent_alerts) == self.buffer_size:
            return all(self.recent_alerts)
        
        return False
    
    def reset(self):
        """Reset buffer for new patient"""
        self.recent_alerts = []


class EnhancedRiskScorer:
    """
    Enhanced risk scoring with multiple false alarm reduction strategies
    """
    
    def __init__(
        self,
        temporal_buffer_size: int = 3,
        min_confidence: float = 0.7,
        min_abnormal_signals: int = 2,
        motion_threshold: float = 0.5,
        critical_override: bool = True
    ):
        """
        Args:
            temporal_buffer_size: Consecutive windows needed for alert
            min_confidence: Minimum model confidence for non-critical alerts
            min_abnormal_signals: Minimum number of abnormal vitals required
            motion_threshold: Motion std threshold for artifact suppression
            critical_override: Allow immediate alert for critical conditions
        """
        self.alert_buffer = TemporalAlertBuffer(temporal_buffer_size)
        self.min_confidence = min_confidence
        self.min_abnormal_signals = min_abnormal_signals
        self.motion_threshold = motion_threshold
        self.critical_override = critical_override
        
        # Thresholds
        self.THRESHOLDS = {
            'spo2_critical': 88,
            'spo2_warning': 92,
            'hr_critical': 140,
            'hr_warning': 120,
            'hr_low': 50,
            'sbp_critical': 90,
            'sbp_warning': 100,
        }
    
    def count_abnormal_signals(self, features: Dict) -> Tuple[int, list]:
        """
        Count how many vital signs are abnormal
        
        Returns:
            (count, list of abnormal signal names)
        """
        abnormal = []
        
        # SpO2 checks
        if features.get('spo2_mean', 100) < self.THRESHOLDS['spo2_warning']:
            abnormal.append('spo2_low')
        
        if features.get('spo2_min', 100) < self.THRESHOLDS['spo2_critical']:
            abnormal.append('spo2_critical')
        
        # Heart Rate checks
        if features.get('hr_mean', 70) > self.THRESHOLDS['hr_warning']:
            abnormal.append('hr_high')
        
        if features.get('hr_max', 100) > self.THRESHOLDS['hr_critical']:
            abnormal.append('hr_critical')
        
        # Blood Pressure checks
        if features.get('sbp_mean', 120) < self.THRESHOLDS['sbp_warning']:
            abnormal.append('sbp_low')
        
        # Trend checks (deterioration)
        if features.get('spo2_slope', 0) < -0.5:  # Dropping SpO2
            abnormal.append('spo2_dropping')
        
        if features.get('hr_slope', 0) > 0.5:  # Rising HR
            abnormal.append('hr_rising')
        
        return len(abnormal), abnormal
    
    def check_critical_condition(self, features: Dict) -> Tuple[bool, str]:
        """
        Check if patient is in immediately critical state
        These bypass temporal buffering
        
        Returns:
            (is_critical, reason)
        """
        # Severe hypoxemia
        if features.get('spo2_min', 100) < self.THRESHOLDS['spo2_critical']:
            return True, "Critical hypoxemia (SpO2 < 88%)"
        
        # Extreme tachycardia
        if features.get('hr_max', 100) > self.THRESHOLDS['hr_critical']:
            return True, "Critical tachycardia (HR > 140)"
        
        # Severe hypotension
        if features.get('sbp_mean', 120) < self.THRESHOLDS['sbp_critical']:
            return True, "Critical hypotension (SBP < 90)"
        
        # Combined moderate risk (SpO2 dropping + HR rising)
        if (features.get('spo2_mean', 100) < self.THRESHOLDS['spo2_warning'] and 
            features.get('hr_mean', 70) > self.THRESHOLDS['hr_warning']):
            return True, "Combined cardiopulmonary stress"
        
        return False, ""
    
    def check_physiological_plausibility(self, features: Dict) -> Tuple[bool, str]:
        """
        Check if vital signs are physiologically plausible
        Filters out obvious sensor errors
        
        Returns:
            (is_plausible, reason if not)
        """
        # Low HR + High SpO2 = unlikely emergency (sensor disconnect?)
        if (features.get('hr_mean', 70) < self.THRESHOLDS['hr_low'] and 
            features.get('spo2_mean', 95) > 95):
            return False, "Implausible: Low HR with normal SpO2"
        
        # SpO2 drop without HR response is suspicious
        if (features.get('spo2_mean', 100) < 90 and 
            features.get('hr_mean', 70) < 80):
            return False, "Implausible: Severe hypoxia without tachycardia"
        
        # BP sanity check
        sbp = features.get('sbp_mean', 120)
        dbp = features.get('dbp_mean', 80)
        if sbp < dbp and sbp > 0:  # SBP should be > DBP
            return False, "Implausible: SBP < DBP (sensor error)"
        
        return True, ""
    
    def assess_motion_artifact(self, features: Dict) -> Tuple[bool, float]:
        """
        Determine if high motion might be causing false readings
        
        Returns:
            (is_high_motion, motion_level)
        """
        motion_std = features.get('motion_std', 0)
        motion_max = features.get('motion_max', 0)
        
        is_high_motion = (
            motion_std > self.motion_threshold or 
            motion_max > 1.0
        )
        
        return is_high_motion, motion_std
    
    def calculate_risk_score(
        self, 
        features: Dict, 
        anomaly_score: float,
        confidence: float
    ) -> Tuple[float, bool, str, Dict]:
        """
        Calculate comprehensive risk score with multi-stage filtering
        
        Args:
            features: Dictionary of engineered features
            anomaly_score: Raw anomaly score from model (0-1, higher = more anomalous)
            confidence: Model confidence (0-1)
        
        Returns:
            (risk_score, should_alert, explanation, debug_info)
        """
        debug_info = {
            'stage': '',
            'abnormal_signals': [],
            'critical': False,
            'plausible': True,
            'high_motion': False,
            'buffered': False
        }
        
        # STAGE 1: Check physiological plausibility
        is_plausible, plausibility_reason = self.check_physiological_plausibility(features)
        debug_info['plausible'] = is_plausible
        
        if not is_plausible:
            debug_info['stage'] = 'plausibility_failed'
            return 0, False, f"Suppressed: {plausibility_reason}", debug_info
        
        # STAGE 2: Check for critical conditions (immediate alert)
        is_critical, critical_reason = self.check_critical_condition(features)
        debug_info['critical'] = is_critical
        
        if is_critical and self.critical_override:
            risk_score = 95 + (anomaly_score * 5)  # 95-100 range
            debug_info['stage'] = 'critical_override'
            return risk_score, True, f"CRITICAL: {critical_reason}", debug_info
        
        # STAGE 3: Count abnormal signals
        abnormal_count, abnormal_signals = self.count_abnormal_signals(features)
        debug_info['abnormal_signals'] = abnormal_signals
        
        if abnormal_count < self.min_abnormal_signals:
            debug_info['stage'] = 'insufficient_signals'
            return anomaly_score * 50, False, f"Suppressed: Only {abnormal_count} abnormal signal(s)", debug_info
        
        # STAGE 4: Check motion artifacts
        is_high_motion, motion_level = self.assess_motion_artifact(features)
        debug_info['high_motion'] = is_high_motion
        
        # Suppress non-critical alerts during high motion
        if is_high_motion and not is_critical:
            debug_info['stage'] = 'motion_artifact'
            return anomaly_score * 40, False, f"Suppressed: High motion (std={motion_level:.2f})", debug_info
        
        # STAGE 5: Check confidence
        if confidence < self.min_confidence and not is_critical:
            debug_info['stage'] = 'low_confidence'
            return anomaly_score * 60, False, f"Suppressed: Low confidence ({confidence:.2f})", debug_info
        
        # STAGE 6: Calculate base risk score
        base_risk = anomaly_score * 100
        
        # Boost for multiple abnormal signals
        signal_boost = min(abnormal_count * 5, 20)
        risk_score = min(base_risk + signal_boost, 100)
        
        # STAGE 7: Temporal buffering (require sustained anomaly)
        raw_alert = risk_score > 70  # Base threshold
        buffered_alert = self.alert_buffer.should_alert(raw_alert)
        debug_info['buffered'] = buffered_alert
        
        if buffered_alert:
            debug_info['stage'] = 'confirmed_alert'
            explanation = f"Alert: {len(abnormal_signals)} abnormal signals - {', '.join(abnormal_signals[:3])}"
            return risk_score, True, explanation, debug_info
        else:
            debug_info['stage'] = 'awaiting_confirmation'
            return risk_score, False, "Monitoring: Awaiting confirmation", debug_info
    
    def reset(self):
        """Reset for new patient"""
        self.alert_buffer.reset()


def apply_enhanced_risk_scoring(
    df_features: pd.DataFrame,
    df_predictions: pd.DataFrame,
    temporal_buffer_size: int = 3,
    min_confidence: float = 0.7,
    min_abnormal_signals: int = 2
) -> pd.DataFrame:
    """
    Apply enhanced risk scoring to predictions DataFrame
    
    Args:
        df_features: DataFrame with engineered features
        df_predictions: DataFrame with model predictions
        temporal_buffer_size: Consecutive windows for alert
        min_confidence: Minimum confidence threshold
        min_abnormal_signals: Minimum abnormal signals required
    
    Returns:
        Enhanced predictions DataFrame with new columns:
        - enhanced_risk_score
        - enhanced_alert
        - alert_explanation
        - debug_info
    """
    scorer = EnhancedRiskScorer(
        temporal_buffer_size=temporal_buffer_size,
        min_confidence=min_confidence,
        min_abnormal_signals=min_abnormal_signals
    )
    
    results = []
    
    for idx, pred_row in df_predictions.iterrows():
        # Get corresponding features
        feat_row = df_features.iloc[idx]
        
        # Convert to dict
        features = feat_row.to_dict()
        
        # Calculate enhanced risk
        risk_score, should_alert, explanation, debug = scorer.calculate_risk_score(
            features=features,
            anomaly_score=pred_row.get('anomaly_score', 0.5),
            confidence=pred_row.get('confidence', 0.5)
        )
        
        results.append({
            'enhanced_risk_score': risk_score,
            'enhanced_alert': int(should_alert),
            'alert_explanation': explanation,
            'abnormal_count': len(debug['abnormal_signals']),
            'is_critical': debug['critical'],
            'stage': debug['stage']
        })
    
    # Add to predictions
    results_df = pd.DataFrame(results)
    df_enhanced = pd.concat([df_predictions.reset_index(drop=True), results_df], axis=1)
    
    return df_enhanced


# Example usage
if __name__ == "__main__":
    # Test the scorer
    scorer = EnhancedRiskScorer(
        temporal_buffer_size=3,
        min_confidence=0.7,
        min_abnormal_signals=2
    )
    
    # Test case 1: Critical condition (should alert immediately)
    test_features_critical = {
        'spo2_mean': 85,
        'spo2_min': 82,
        'hr_mean': 130,
        'hr_max': 145,
        'sbp_mean': 110,
        'motion_std': 0.2
    }
    
    risk, alert, explanation, debug = scorer.calculate_risk_score(
        test_features_critical,
        anomaly_score=0.8,
        confidence=0.9
    )
    
    print("Test 1 - Critical Condition:")
    print(f"  Risk Score: {risk:.1f}")
    print(f"  Alert: {alert}")
    print(f"  Explanation: {explanation}")
    print(f"  Stage: {debug['stage']}\n")
    
    # Test case 2: Single abnormal signal (should suppress)
    test_features_single = {
        'spo2_mean': 94,
        'spo2_min': 93,
        'hr_mean': 85,
        'hr_max': 95,
        'sbp_mean': 120,
        'motion_std': 0.1
    }
    
    risk, alert, explanation, debug = scorer.calculate_risk_score(
        test_features_single,
        anomaly_score=0.6,
        confidence=0.8
    )
    
    print("Test 2 - Single Abnormal Signal:")
    print(f"  Risk Score: {risk:.1f}")
    print(f"  Alert: {alert}")
    print(f"  Explanation: {explanation}")
    print(f"  Stage: {debug['stage']}\n")
    
    # Test case 3: High motion artifact (should suppress)
    test_features_motion = {
        'spo2_mean': 91,
        'spo2_min': 88,
        'hr_mean': 110,
        'hr_max': 125,
        'sbp_mean': 105,
        'motion_std': 0.8  # High motion
    }
    
    risk, alert, explanation, debug = scorer.calculate_risk_score(
        test_features_motion,
        anomaly_score=0.7,
        confidence=0.7
    )
    
    print("Test 3 - High Motion:")
    print(f"  Risk Score: {risk:.1f}")
    print(f"  Alert: {alert}")
    print(f"  Explanation: {explanation}")
    print(f"  Stage: {debug['stage']}")