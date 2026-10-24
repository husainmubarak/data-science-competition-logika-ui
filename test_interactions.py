"""
test_interactions.py
Test engineered risk interactions
"""
import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

b = joblib.load('artifacts/features_v5.joblib')
X = b['X_train']
y = b['y_train']

print("Testing risk index interactions...")

# 1. Repayment Risk Index
rep_risk = (
    X['rep_max_dpd_mean'].fillna(0) * 
    X['rep_late_ratio'].fillna(0) * 
    (1.0 + X['rep_dpd60_ratio'].fillna(0))
)
print(f"1. Repayment Risk Index: corr={rep_risk.corr(pd.Series(y)):.4f}, KS={ks_2samp(rep_risk[y==1], rep_risk[y==0]).statistic:.4f}")

# 2. Bureau Stress Index
score = X['bureau_curr_score_mean'].fillna(X['bureau_hist_score_mean']).fillna(600)
util = X['bureau_agg_utilization'].fillna(0.3)
bureau_stress = (850.0 - score) * (1.0 + util)
print(f"2. Bureau Stress Index: corr={bureau_stress.corr(pd.Series(y)):.4f}, KS={ks_2samp(bureau_stress[y==1], bureau_stress[y==0]).statistic:.4f}")

# 3. Rate hike / pricing spread
rate_hike = X['annual_interest_rate'] - X['prev_rate_mean'].fillna(X['annual_interest_rate'])
print(f"3. Rate Hike: corr={rate_hike.corr(pd.Series(y)):.4f}, KS={ks_2samp(rate_hike[y==1], rate_hike[y==0]).statistic:.4f}")

# 4. Global Super Risk Interaction
# Combines rate tier, score inverse, max dpd, and grocery spend mean!
super_risk = (
    (X['annual_interest_rate'] / 0.16) * 
    (750.0 / (score + 1)) * 
    (1.0 + X['rep_max_dpd_mean'].fillna(0) / 10.0) * 
    (1.0 + X['ec_groceries_item_mean'].fillna(X['ec_groceries_item_mean'].median()) / 100000.0)
)
print(f"4. Super Risk Composite: corr={super_risk.corr(pd.Series(y)):.4f}, KS={ks_2samp(super_risk[y==1], super_risk[y==0]).statistic:.4f}")

# 5. Non-linear delinquency power
dpd_power = np.log1p(X['rep_max_dpd_mean'].fillna(0)) * (X['rep_most_recent_max_dpd'].fillna(0) > 0).astype(int)
print(f"5. DPD Power: corr={dpd_power.corr(pd.Series(y)):.4f}, KS={ks_2samp(dpd_power[y==1], dpd_power[y==0]).statistic:.4f}")
