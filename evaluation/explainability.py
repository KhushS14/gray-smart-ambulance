import shap
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import numpy as np

# Load model and test data
model = joblib.load("models/anomaly_model.joblib")
scaler = joblib.load("models/scaler.joblib")
feature_cols = joblib.load("models/feature_cols.joblib")

feat_df = pd.read_csv("data/test/patient_99_deteriorating_features.csv")
X = feat_df[feature_cols]
X_scaled = scaler.transform(X)

# SHAP explainability for Isolation Forest
try:
    # Sample a background dataset (use fewer samples for faster computation)
    background = shap.sample(X_scaled, 50)
    
    # Create explainer
    explainer = shap.KernelExplainer(model.predict, background)
    
    # Calculate SHAP values for first 100 samples (this may take time)
    shap_values = explainer.shap_values(X_scaled[:100])
    
    # Plot 1: Feature importance
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_scaled[:100], 
                      feature_names=feature_cols, show=False)
    plt.tight_layout()
    plt.savefig("evaluation/shap_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ SHAP summary plot saved")
    
    # Plot 2: Single prediction explanation
    plt.figure(figsize=(10, 4))
    shap.waterfall_plot(shap.Explanation(values=shap_values[0], 
                                         base_values=explainer.expected_value,
                                         data=X_scaled[0],
                                         feature_names=feature_cols), 
                       show=False)
    plt.tight_layout()
    plt.savefig("evaluation/shap_waterfall.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ SHAP waterfall plot saved")
    
except Exception as e:
    print(f"⚠️ SHAP analysis failed: {e}")

# Alternative: Calculate feature importance via permutation
# For anomaly detection, we use the anomaly scores as the target
from sklearn.inspection import permutation_importance

print("\n🔄 Calculating permutation importance...")

# Get anomaly scores (decision_function returns anomaly scores)
# More negative = more anomalous
y_scores = model.decision_function(X_scaled)

# Use permutation importance with anomaly scores
result = permutation_importance(
    model, 
    X_scaled, 
    y_scores,  # Use anomaly scores as target
    scoring='neg_mean_squared_error',
    n_repeats=10, 
    random_state=42,
    n_jobs=-1  # Use all CPU cores
)

importances = result.importances_mean
std = result.importances_std
indices = np.argsort(importances)[::-1]

# Create bar plot with error bars
plt.figure(figsize=(12, 6))
plt.bar(range(len(importances)), 
        importances[indices],
        yerr=std[indices],
        capsize=3,
        alpha=0.7,
        edgecolor='black')
plt.xticks(range(len(importances)), 
           [feature_cols[i] for i in indices], 
           rotation=45, 
           ha='right')
plt.xlabel('Features', fontsize=12)
plt.ylabel('Permutation Importance', fontsize=12)
plt.title("Feature Importance - Isolation Forest (Permutation Method)", fontsize=14)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("evaluation/feature_importance.png", dpi=150, bbox_inches='tight')
plt.close()
print("✅ Feature importance plot saved")

# Print top features
print("\n📊 Top 10 Most Important Features:")
for i, idx in enumerate(indices[:10], 1):
    print(f"{i}. {feature_cols[idx]}: {importances[idx]:.4f} ± {std[idx]:.4f}")