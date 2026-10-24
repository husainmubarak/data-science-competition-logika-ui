"""
check_subsets.py
Check KS breakdown by segment
"""
import numpy as np
import pandas as pd
import joblib
from scipy.stats import ks_2samp

bundle = joblib.load('artifacts/features_v5.joblib')
X = bundle['X_train']
y = bundle['y_train']

# Let's inspect the models saved in experiment_models_v5 or retrain a quick model
is_existing = (X['has_repayment_history'] == 1).values

# Let's check single feature KS on existing vs new:
print("--- KS ON EXISTING CUSTOMERS (n=46312) ---")
for c in ['composite_risk_score', 'rep_late_ratio', 'rep_max_dpd_mean', 'bureau_curr_score_mean', 'ec_90d_spend_sum']:
    vals = X.loc[is_existing, c].dropna()
    y_sub = y[is_existing][X.loc[is_existing, c].notna()]
    bads = vals[y_sub == 1]
    goods = vals[y_sub == 0]
    print(f"  {c:30s}: KS = {ks_2samp(bads, goods).statistic:.4f}")

print("\n--- KS ON NEW CUSTOMERS (n=23688) ---")
for c in ['bureau_curr_score_mean', 'annual_interest_rate', 'ec_90d_spend_sum', 'rate_to_bureau_score', 'loan_amount', 'dti']:
    vals = X.loc[~is_existing, c].dropna()
    y_sub = y[~is_existing][X.loc[~is_existing, c].notna()]
    bads = vals[y_sub == 1]
    goods = vals[y_sub == 0]
    print(f"  {c:30s}: KS = {ks_2samp(bads, goods).statistic:.4f}")
