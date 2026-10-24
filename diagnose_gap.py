"""
diagnose_gap.py
Investigate the performance gap to reach KS >= 0.57
"""
import numpy as np
import pandas as pd
import joblib
from scipy.stats import ks_2samp

bundle = joblib.load('artifacts/features_v5.joblib')
X = bundle['X_train']
y = bundle['y_train']

ensemble = joblib.load('artifacts/selected_ensemble.joblib')

# Let's inspect OOF KS by subset
is_existing = (X['has_repayment_history'] == 1).values
print(f"Overall target rate: {y.mean():.4f}")
print(f"Existing customers target rate: {y[is_existing].mean():.4f} (n={is_existing.sum()})")
print(f"New customers target rate:      {y[~is_existing].mean():.4f} (n={(~is_existing).sum()})")

# Let's check feature correlations with target
corrs = X.apply(lambda col: col.corr(pd.Series(y)))
top_pos = corrs.sort_values(ascending=False).head(25)
top_neg = corrs.sort_values().head(25)

print("\n--- TOP 25 POSITIVE CORRELATIONS WITH TARGET ---")
print(top_pos.to_string())

print("\n--- TOP 25 NEGATIVE CORRELATIONS WITH TARGET ---")
print(top_neg.to_string())

# Check KS for single features directly
print("\n--- TOP SINGLE FEATURES BY DIRECT KS ---")
single_ks = []
for c in X.columns:
    col_vals = X[c].fillna(-999).values
    # evaluate direct KS
    bads = col_vals[y == 1]
    goods = col_vals[y == 0]
    ks = ks_2samp(bads, goods).statistic
    single_ks.append((c, ks))

single_ks = sorted(single_ks, key=lambda x: x[1], reverse=True)
for c, ks in single_ks[:30]:
    print(f"{c:40s}: {ks:.4f}")
