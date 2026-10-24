"""
test_new_signals.py
Directly test new signal hypotheses on raw data to find features that give major KS boost
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import ks_2samp

DATA_DIR = Path('D:/LOGIKA UI/code/data-science-competition-logika-ui-2026')

train = pd.read_csv(DATA_DIR / 'train_applications.csv', parse_dates=['application_datetime', 'decision_datetime'])
y = train['target'].values
n = len(train)
print(f"Train size: {n}, Positive rate: {y.mean():.4f}")

profiles = pd.read_csv(DATA_DIR / 'customer_profiles.csv')
train = train.merge(profiles, on='customer_id', how='left')

# 1. Income inflation test
acct = pd.read_csv(DATA_DIR / 'account_behavior.csv', parse_dates=['month'])
acct_m = acct.merge(train[['customer_id', 'decision_datetime']], on='customer_id')
acct_m = acct_m[acct_m['month'] <= acct_m['decision_datetime']]
acct_inc = acct_m.groupby('customer_id')['income_like_credits'].mean().reset_index()
train = train.merge(acct_inc, on='customer_id', how='left')

train['income_inflation_ratio'] = train['declared_monthly_income'] / (train['income_like_credits'] + 1000)
corr1 = train['income_inflation_ratio'].corr(pd.Series(y))
print(f"1. Income inflation ratio corr: {corr1:.4f}")

# 2. Account balance volatility & low balance trajectory
acct_sorted = acct_m.sort_values(['customer_id', 'month'])
acct_last2 = acct_sorted.groupby('customer_id').tail(2)
acct_trend = acct_last2.groupby('customer_id').agg(
    low_bal_last=('low_balance_days', 'last'),
    low_bal_prev=('low_balance_days', 'first'),
    bal_last=('closing_balance', 'last'),
    bal_prev=('closing_balance', 'first'),
    open_last=('opening_balance', 'last'),
).reset_index()
acct_trend['low_bal_acceleration'] = acct_trend['low_bal_last'] - acct_trend['low_bal_prev']
acct_trend['savings_rate_last'] = (acct_trend['bal_last'] - acct_trend['open_last']) / (acct_trend['bal_last'] + 1000)
acct_trend['bal_growth'] = acct_trend['bal_last'] / (acct_trend['bal_prev'] + 1000)

train = train.merge(acct_trend, on='customer_id', how='left')
for col in ['low_bal_acceleration', 'savings_rate_last', 'bal_growth']:
    corr = train[col].corr(pd.Series(y))
    print(f"2. {col:25s} corr: {corr:.4f}")

# 3. Previous application rejection & haircut
prev = pd.read_csv(DATA_DIR / 'previous_applications.csv', parse_dates=['decision_datetime'])
prev_m = prev.merge(train[['customer_id', 'decision_datetime']], on='customer_id', suffixes=('', '_curr'))
prev_m = prev_m[prev_m['decision_datetime'] <= prev_m['decision_datetime_curr']]

prev_last = prev_m.sort_values('decision_datetime').groupby('customer_id').last().reset_index()
prev_last['last_app_was_rejected'] = (prev_last['decision'] == 'rejected').astype('float32')
prev_last['last_haircut_ratio'] = prev_last['approved_amount'] / (prev_last['requested_amount'] + 1)
prev_last['days_since_last_app'] = (prev_last['decision_datetime_curr'] - prev_last['decision_datetime']).dt.total_seconds() / 86400.0

train = train.merge(prev_last[['customer_id', 'last_app_was_rejected', 'last_haircut_ratio', 'days_since_last_app']], on='customer_id', how='left')
for col in ['last_app_was_rejected', 'last_haircut_ratio', 'days_since_last_app']:
    corr = train[col].corr(pd.Series(y))
    print(f"3. {col:25s} corr: {corr:.4f}")

# 4. Bureau provider score discrepancy & active debt to income
bureau = pd.read_csv(DATA_DIR / 'bureau_observations.csv', parse_dates=['available_datetime'])
b_m = bureau.merge(train[['application_id', 'decision_datetime', 'declared_monthly_income']], on='application_id')
b_m = b_m[b_m['available_datetime'] <= b_m['decision_datetime']]

b_agg = b_m.groupby('application_id').agg(
    b_score_std=('score', 'std'),
    b_score_min=('score', 'min'),
    b_score_max=('score', 'max'),
    b_debt_total=('outstanding_balance', 'sum'),
    b_inquiries_sum=('inquiries_last_6m', 'sum'),
    b_delinq_sum=('delinquent_accounts_last_12m', 'sum'),
).reset_index()
b_agg['b_score_spread'] = b_agg['b_score_max'] - b_agg['b_score_min']

train = train.merge(b_agg, on='application_id', how='left')
train['b_dti_annual'] = train['b_debt_total'] / (train['declared_monthly_income'] * 12 + 1000)

for col in ['b_score_std', 'b_score_spread', 'b_debt_total', 'b_inquiries_sum', 'b_delinq_sum', 'b_dti_annual']:
    corr = train[col].corr(pd.Series(y))
    print(f"4. {col:25s} corr: {corr:.4f}")

print("\nAll signal checks completed.")
