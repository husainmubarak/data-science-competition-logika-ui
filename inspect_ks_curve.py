"""
inspect_ks_curve.py
Analyze the TPR vs FPR curve to see where the KS peak is and why it stops at ~0.504
"""
import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_curve, roc_auc_score

b = joblib.load('artifacts/features_v5.joblib')
y = b['y_train']
X = b['X_train']

ens = joblib.load('artifacts/selected_ensemble.joblib')
print("Loaded ensemble bundle:", ens.keys())

# Let's inspect the models and oof predictions
# Let's load the oof predictions from a quick run or from saved files if any
# Let's check single high-leverage features:
for col in ['composite_risk_score', 'rep_max_dpd_mean', 'rep_late_ratio', 'bureau_curr_score_mean', 'ec_groceries_item_mean']:
    val = X[col].fillna(X[col].median()).values
    auc = roc_auc_score(y, val)
    if auc < 0.5:
        auc = 1 - auc
        val = -val
    fpr, tpr, thresholds = roc_curve(y, val)
    ks = np.max(tpr - fpr)
    idx = np.argmax(tpr - fpr)
    print(f"Feature {col:30s} AUC: {auc:.4f} | KS: {ks:.4f} (at TPR={tpr[idx]:.3f}, FPR={fpr[idx]:.3f})")
