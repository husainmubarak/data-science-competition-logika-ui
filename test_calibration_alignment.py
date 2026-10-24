"""
test_calibration_alignment.py
Test if shifting or scaling probabilities of new customers vs existing customers boosts pooled KS
"""
import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.isotonic import IsotonicRegression

b = joblib.load('artifacts/features_v5.joblib')
y = b['y_train']
X = b['X_train']
is_ex = (X['has_repayment_history'] == 1).values

# Let's run 5-fold CV to get OOF
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_raw = np.zeros(len(y))
for tr, val in skf.split(X, y):
    m = LGBMClassifier(n_estimators=500, learning_rate=0.03, max_depth=7, num_leaves=63, random_state=42, n_jobs=-1, verbosity=-1)
    m.fit(X.iloc[tr], y[tr])
    oof_raw[val] = m.predict_proba(X.iloc[val])[:, 1]

print("Raw Pooled KS:", ks_2samp(oof_raw[y==1], oof_raw[y==0]).statistic)
print("Raw Existing KS:", ks_2samp(oof_raw[(y==1)&is_ex], oof_raw[(y==0)&is_ex]).statistic)
print("Raw New KS:", ks_2samp(oof_raw[(y==1)&(~is_ex)], oof_raw[(y==0)&(~is_ex)]).statistic)

# Experiment A: Isotonic calibration on each segment out-of-fold
oof_iso = np.zeros(len(y))
for tr, val in skf.split(X, y):
    # fit isotonic separately on existing and new in train
    for mask in [is_ex, ~is_ex]:
        tr_m = tr[mask[tr]]
        val_m = val[mask[val]]
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(oof_raw[tr_m], y[tr_m])
        oof_iso[val_m] = iso.predict(oof_raw[val_m])

print("Isotonic Calibrated Pooled KS:", ks_2samp(oof_iso[y==1], oof_iso[y==0]).statistic)

# Experiment B: Grid search multiplier / shift for new customers
best_ks = ks_2samp(oof_raw[y==1], oof_raw[y==0]).statistic
best_shift = 0
best_mult = 1.0

for mult in np.linspace(0.5, 2.0, 31):
    for shift in np.linspace(-0.1, 0.1, 21):
        adj = oof_raw.copy()
        adj[~is_ex] = np.clip(adj[~is_ex] * mult + shift, 0, 1)
        score = ks_2samp(adj[y==1], adj[y==0]).statistic
        if score > best_ks:
            best_ks = score
            best_shift = shift
            best_mult = mult

print(f"Optimal Adjustment Pooled KS: {best_ks:.5f} (mult={best_mult:.2f}, shift={best_shift:.4f})")
