"""
Modified metrics script to evaluate on deteriorating test patient
WITH ENHANCED RISK SCORING FOR REDUCED FALSE ALARMS
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path
sys.path.append('src')

# Create output directory
Path("evaluation").mkdir(exist_ok=True)


def define_ground_truth_comprehensive(
    df_clean: pd.DataFrame,
    df_risk: pd.DataFrame,
    spo2_critical: float = 90.0,
    spo2_warning: float = 92.0,
    hr_critical: float = 140.0,
    hr_warning: float = 120.0,
    sbp_critical: float = 90.0
) -> pd.Series:
    """
    Define ground truth at window level using MULTIPLE vital signs.
    """
    gt = []
    
    for t in df_risk["time_sec"]:
        # Find vitals at this time window
        window_data = df_clean[df_clean["time_sec"] == t]
        
        if len(window_data) == 0:
            gt.append(0)
            continue
        
        row = window_data.iloc[0]
        
        # Critical conditions (definite anomaly)
        is_critical = (
            row["spo2_clean"] < spo2_critical or
            row["heart_rate_clean"] > hr_critical or
            row["sbp"] < sbp_critical
        )
        
        # Combined moderate risk (also considered anomaly)
        is_combined_risk = (
            row["spo2_clean"] < spo2_warning and
            row["heart_rate_clean"] > hr_warning
        )
        
        is_anomaly = is_critical or is_combined_risk
        gt.append(int(is_anomaly))
    
    return pd.Series(gt)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute precision, recall, and false alert rate."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    false_alert_rate = fp / (fp + tn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    specificity = tn / (tn + fp + 1e-6)
    
    return {
        "precision": precision,
        "recall": recall,
        "false_alert_rate": false_alert_rate,
        "f1_score": f1,
        "specificity": specificity,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "total_predictions": len(y_pred),
        "total_anomalies": int(np.sum(y_true))
    }


def compute_alert_latency(
    time_sec: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    """Compute alert latency statistics."""
    # Find all anomaly episodes
    true_episodes = []
    in_episode = False
    episode_start = None
    
    for i, label in enumerate(y_true):
        if label == 1 and not in_episode:
            episode_start = i
            in_episode = True
        elif label == 0 and in_episode:
            true_episodes.append((episode_start, i-1))
            in_episode = False
    
    if in_episode:
        true_episodes.append((episode_start, len(y_true)-1))
    
    latencies = []
    for start_idx, end_idx in true_episodes:
        alerts_after = np.where(y_pred[start_idx:] == 1)[0]
        if len(alerts_after) > 0:
            first_alert_offset = alerts_after[0]
            latency = time_sec[start_idx + first_alert_offset] - time_sec[start_idx]
            latencies.append(latency)
        else:
            latencies.append(np.nan)
    
    if len(latencies) == 0:
        return {
            "avg_latency": np.nan,
            "median_latency": np.nan,
            "max_latency": np.nan,
            "min_latency": np.nan,
            "num_episodes": 0,
            "missed_episodes": 0
        }
    
    return {
        "avg_latency": np.nanmean(latencies),
        "median_latency": np.nanmedian(latencies),
        "max_latency": np.nanmax(latencies),
        "min_latency": np.nanmin(latencies),
        "num_episodes": len(latencies),
        "missed_episodes": int(np.sum(np.isnan(latencies)))
    }


def plot_confusion_matrix(metrics: dict, save_path: str = "evaluation/confusion_matrix.png"):
    """Plot confusion matrix"""
    cm = np.array([
        [metrics['tn'], metrics['fp']],
        [metrics['fn'], metrics['tp']]
    ])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap='Blues')
    
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted Normal', 'Predicted Anomaly'])
    ax.set_yticklabels(['Actual Normal', 'Actual Anomaly'])
    
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, cm[i, j], ha="center", va="center", 
                          color="white" if cm[i, j] > cm.max()/2 else "black",
                          fontsize=20, fontweight='bold')
    
    ax.set_title("Confusion Matrix - Smart Ambulance Alert System", fontsize=14, pad=20)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Confusion matrix saved to {save_path}")
    plt.close()


def plot_metrics_comparison(metrics: dict, save_path: str = "evaluation/metrics_comparison.png"):
    """Plot key metrics as bar chart"""
    metric_names = ['Precision', 'Recall', 'F1 Score', 'Specificity']
    values = [
        metrics['precision'], 
        metrics['recall'], 
        metrics['f1_score'],
        metrics['specificity']
    ]
    
    colors = ['#2ecc71' if v >= 0.8 else '#e74c3c' for v in values]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(metric_names, values, color=colors, alpha=0.7, edgecolor='black')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontweight='bold')
    
    ax.axhline(y=0.8, color='orange', linestyle='--', label='Target: 0.80')
    ax.axhline(y=0.95, color='green', linestyle='--', label='Target (Recall): 0.95')
    
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Performance Metrics - Smart Ambulance Alert System (ENHANCED)', fontsize=14, pad=20)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Metrics comparison saved to {save_path}")
    plt.close()


def print_detailed_report(metrics: dict, latency_stats: dict, original_metrics: dict = None):
    """Print comprehensive evaluation report"""
    print("\n" + "="*70)
    print(" " * 15 + "SMART AMBULANCE ALERT SYSTEM")
    print(" " * 20 + "EVALUATION REPORT (ENHANCED)")
    print("="*70)
    
    if original_metrics:
        print("\n📊 IMPROVEMENT SUMMARY")
        print("-" * 70)
        print(f"Original Precision:  {original_metrics['precision']:.3f} → Enhanced: {metrics['precision']:.3f} ({((metrics['precision']-original_metrics['precision'])/original_metrics['precision']*100):+.1f}%)")
        print(f"Original Recall:     {original_metrics['recall']:.3f} → Enhanced: {metrics['recall']:.3f} ({((metrics['recall']-original_metrics['recall'])/original_metrics['recall']*100):+.1f}%)")
        print(f"Original FP Rate:    {original_metrics['false_alert_rate']:.3f} → Enhanced: {metrics['false_alert_rate']:.3f} ({((metrics['false_alert_rate']-original_metrics['false_alert_rate'])/original_metrics['false_alert_rate']*100):+.1f}%)")
        print(f"False Alarms Reduced: {original_metrics['fp']} → {metrics['fp']} ({original_metrics['fp']-metrics['fp']} fewer)")
    
    print("\n📊 CLASSIFICATION METRICS")
    print("-" * 70)
    print(f"Precision:           {metrics['precision']:.3f}   ({metrics['precision']*100:.1f}% of alerts are real)")
    print(f"Recall:              {metrics['recall']:.3f}   (Catches {metrics['recall']*100:.1f}% of deteriorations)")
    print(f"F1 Score:            {metrics['f1_score']:.3f}   (Harmonic mean)")
    print(f"Specificity:         {metrics['specificity']:.3f}   ({metrics['specificity']*100:.1f}% correct non-alerts)")
    print(f"False Alert Rate:    {metrics['false_alert_rate']:.3f}   ({metrics['false_alert_rate']*100:.1f}%)")
    
    print("\n📈 CONFUSION MATRIX")
    print("-" * 70)
    print(f"True Positives:      {metrics['tp']} ✅ (Correctly detected emergencies)")
    print(f"False Positives:     {metrics['fp']} ⚠️  (False alarms)")
    print(f"True Negatives:      {metrics['tn']} ✅ (Correctly identified normal)")
    print(f"False Negatives:     {metrics['fn']} ❌ (MISSED emergencies - DANGEROUS)")
    print(f"Total Windows:       {metrics['total_predictions']}")
    print(f"True Anomalies:      {metrics['total_anomalies']}")
    
    print("\n⏱️  ALERT LATENCY")
    print("-" * 70)
    if not np.isnan(latency_stats['avg_latency']):
        print(f"Average Latency:     {latency_stats['avg_latency']:.1f}s")
        print(f"Median Latency:      {latency_stats['median_latency']:.1f}s")
        print(f"Min Latency:         {latency_stats['min_latency']:.1f}s")
        print(f"Max Latency:         {latency_stats['max_latency']:.1f}s")
        print(f"Total Episodes:      {latency_stats['num_episodes']}")
        print(f"Missed Episodes:     {latency_stats['missed_episodes']}")
    
    print("\n" + "="*70)
    print("SYSTEM PERFORMANCE ASSESSMENT")
    print("="*70)
    
    if metrics['recall'] >= 0.90 and metrics['precision'] >= 0.85 and metrics['false_alert_rate'] <= 0.15:
        print("✅ OVERALL: SYSTEM READY FOR FIELD TESTING")
    elif metrics['recall'] >= 0.85 and metrics['precision'] >= 0.80:
        print("⚠️  OVERALL: GOOD PERFORMANCE - MINOR IMPROVEMENTS NEEDED")
    elif metrics['recall'] >= 0.80:
        print("⚠️  OVERALL: NEEDS IMPROVEMENT")
    else:
        print("❌ OVERALL: NOT READY - RECALL TOO LOW")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    print("Loading test patient data (patient_99 - deteriorating)...")
    
    # Load deteriorating patient data
    df_clean = pd.read_csv("data/test/patient_99_deteriorating_clean.csv")
    df_risk = pd.read_csv("data/test/patient_99_deteriorating_risk.csv")
    df_features = pd.read_csv("data/test/patient_99_deteriorating_features.csv")
    
    print(f"✅ Loaded {len(df_clean)} vitals + {len(df_risk)} risk windows + {len(df_features)} features")
    
    # Compute ORIGINAL metrics (before enhancement)
    print("\nDefining ground truth labels...")
    df_risk["ground_truth"] = define_ground_truth_comprehensive(df_clean, df_risk)
    print(f"✅ Ground truth anomalies: {df_risk['ground_truth'].sum()} / {len(df_risk)} windows")
    
    original_metrics = None
    if 'alert' in df_risk.columns:
        print("\nComputing ORIGINAL metrics (before enhancement)...")
        original_metrics = compute_metrics(
            df_risk["ground_truth"].values,
            df_risk["alert"].values
        )
        print(f"  Original Precision: {original_metrics['precision']:.3f}")
        print(f"  Original Recall: {original_metrics['recall']:.3f}")
        print(f"  Original False Positives: {original_metrics['fp']}")
    
    # APPLY ENHANCED RISK SCORING
    print("\n🔧 Applying enhanced risk scoring with false alarm reduction...")
    
    # Import enhanced risk scorer
    try:
        from enhanced_risk_score import apply_enhanced_risk_scoring
    except ImportError:
        print("⚠️  Warning: enhanced_risk_score.py not found in src/")
        print("   Please ensure enhanced_risk_score.py is in your src/ directory")
        print("   Falling back to original predictions...")
        
        # Fallback: use original predictions
        if 'alert' not in df_risk.columns:
            print("❌ No alert column found. Cannot proceed.")
            exit(1)
    else:
        df_risk = apply_enhanced_risk_scoring(
            df_features=df_features,
            df_predictions=df_risk,
            temporal_buffer_size=3,  # Require 3 consecutive anomalies
            min_confidence=0.7,       # Minimum confidence threshold
            min_abnormal_signals=2    # Require at least 2 abnormal vitals
        )
        
        # Use enhanced alerts instead of raw alerts
        df_risk["alert"] = df_risk["enhanced_alert"]
        df_risk["risk_score"] = df_risk["enhanced_risk_score"]
        
        print(f"✅ Enhanced scoring applied")
        if 'anomaly_detected' in df_risk.columns:
            print(f"   Original alerts: {df_risk['anomaly_detected'].sum()}")
        print(f"   Enhanced alerts: {df_risk['enhanced_alert'].sum()}")
    
    # Compute ENHANCED metrics
    print("\nComputing ENHANCED metrics...")
    metrics = compute_metrics(
        df_risk["ground_truth"].values,
        df_risk["alert"].values
    )
    
    latency_stats = compute_alert_latency(
        df_risk["time_sec"].values,
        df_risk["ground_truth"].values,
        df_risk["alert"].values
    )
    
    # Print report
    print_detailed_report(metrics, latency_stats, original_metrics)
    
    # Save results
    results_df = pd.DataFrame([{
        **metrics,
        **{f"latency_{k}": v for k, v in latency_stats.items()}
    }])
    
    results_df.to_csv("evaluation/metrics_results_enhanced.csv", index=False)
    print("✅ Enhanced metrics saved to evaluation/metrics_results_enhanced.csv")
    
    df_risk.to_csv("evaluation/risk_with_ground_truth_enhanced.csv", index=False)
    print("✅ Enhanced risk data saved to evaluation/risk_with_ground_truth_enhanced.csv")
    
    # Generate plots
    print("\nGenerating visualizations...")
    plot_confusion_matrix(metrics, "evaluation/confusion_matrix_enhanced.png")
    plot_metrics_comparison(metrics, "evaluation/metrics_comparison_enhanced.png")
    
    print("\n✅ EVALUATION COMPLETE!")