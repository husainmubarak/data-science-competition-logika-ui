"""
test_features_v7_signals.py
Test high-signal features from:
1. Repayment early/timely payment & delinquency slope
2. Account cash buffer & free cash flow
3. Bureau credit hunger & score trend
4. E-commerce order velocity & COD ratio
5. Combined DTI with bureau debt
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import ks_2samp

DATA_DIR = Path('D:/LOGIKA UI/code/data-science-competition-logika-ui-2026')

train = pd.read_csv(DATA_DIR / 'train_applications.csv', parse_dates=['application_datetime', 'decision_datetime'])
y = train['target'].values
profiles = pd.read_csv(DATA_DIR / 'customer_profiles.csv')
train = train.merge(profiles, on='customer_id', how='left')

print("Evaluating new candidate features on 70,000 train applications...")

# 1. Repayment Timeliness & Slopes
repay = pd.read_csv(DATA_DIR / 'repayment_history.csv', parse_dates=['due_date', 'last_payment_datetime'])
cust_cutoff = dict(zip(train['customer_id'], train['decision_datetime']))
repay['cutoff'] = repay['customer_id'].map(cust_cutoff)
repay = repay[repay['cutoff'].notna()]
repay = repay[repay['due_date'] <= repay['cutoff']]

# Early payment days: positive = paid before due date, negative = paid after due date
paid_mask = repay['last_payment_datetime'].notna() & (repay['last_payment_datetime'] <= repay['cutoff'])
repay['payment_days_diff'] = np.where(paid_mask, (repay['due_date'] - repay['last_payment_datetime']).dt.total_seconds() / 86400.0, -999.0)

# Unpaid ratio
repay['paid_amt'] = np.where(paid_mask, repay['amount_paid'], 0)
repay['unpaid_amt'] = np.maximum(0, repay['amount_due'] - repay['paid_amt'])

rep_signals = repay.groupby('customer_id').agg(
    early_days_mean=('payment_days_diff', lambda x: x[x > -500].mean() if (x > -500).any() else np.nan),
    early_days_min=('payment_days_diff', lambda x: x[x > -500].min() if (x > -500).any() else np.nan),
    total_unpaid_sum=('unpaid_amt', 'sum'),
    total_due_sum=('amount_due', 'sum'),
).reset_index()
rep_signals['overall_unpaid_ratio'] = rep_signals['total_unpaid_sum'] / (rep_signals['total_due_sum'] + 1)

train = train.merge(rep_signals[['customer_id', 'early_days_mean', 'early_days_min', 'overall_unpaid_ratio']], on='customer_id', how='left')

# 2. Account Cash Buffer & Free Cash Flow
acct = pd.read_csv(DATA_DIR / 'account_behavior.csv', parse_dates=['month'])
acct['cutoff'] = acct['customer_id'].map(cust_cutoff)
acct = acct[acct['cutoff'].notna()]
acct = acct[acct['month'] <= acct['cutoff']]

acct_latest = acct.sort_values('month').groupby('customer_id').last().reset_index()
acct_latest['acct_latest_fcf'] = acct_latest['inflows'] - acct_latest['outflows']
train = train.merge(acct_latest[['customer_id', 'closing_balance', 'inflows', 'outflows', 'acct_latest_fcf']], on='customer_id', how='left')

train['cash_buffer_installments'] = train['closing_balance'] / (train['proposed_monthly_installment'] + 1)
train['post_loan_fcf'] = train['acct_latest_fcf'] - train['proposed_monthly_installment']
train['fcf_to_income'] = train['post_loan_fcf'] / (train['declared_monthly_income'] + 1)

# 3. Bureau Total DTI
bureau = pd.read_csv(DATA_DIR / 'bureau_observations.csv', parse_dates=['available_datetime'])
bureau_m = bureau.merge(train[['application_id', 'decision_datetime']], on='application_id')
bureau_m = bureau_m[bureau_m['available_datetime'] <= bureau_m['decision_datetime']]

b_agg = bureau_m.groupby('application_id').agg(
    b_outst_sum=('outstanding_balance', 'sum'),
    b_rev_outst_sum=('revolving_outstanding_balance', 'sum'),
    b_rev_limit_sum=('revolving_credit_limit', 'sum'),
    b_inq_sum=('inquiries_last_6m', 'sum'),
    b_score_mean=('score', 'mean'),
    b_delinq_sum=('delinquent_accounts_last_12m', 'sum'),
).reset_index()
b_agg['b_total_utilization'] = b_agg['b_rev_outst_sum'] / (b_agg['b_rev_limit_sum'] + 1)

train = train.merge(b_agg, on='application_id', how='left')
train['total_monthly_debt_burden'] = train['proposed_monthly_installment'] + (train['b_outst_sum'].fillna(0) / 12.0)
train['total_dti'] = train['total_monthly_debt_burden'] / (train['declared_monthly_income'] + 1)
train['credit_hunger_index'] = train['b_inq_sum'] * (850.0 - train['b_score_mean'].fillna(650.0)) / 100.0

print("\nCandidate features KS & Correlations:")
cols_to_check = [
    'early_days_mean', 'early_days_min', 'overall_unpaid_ratio',
    'cash_buffer_installments', 'post_loan_fcf', 'fcf_to_income',
    'total_dti', 'b_total_utilization', 'credit_hunger_index'
]

for col in cols_to_check:
    corr = train[col].corr(pd.Series(y))
    vals = train[col].fillna(train[col].median()).values
    bads = vals[y == 1]
    goods = vals[y == 0]
    ks = ks_2samp(bads, goods).statistic
    print(f"  {col:30s} | corr: {corr:+.4f} | KS: {ks:.4f}")
