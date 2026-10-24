"""
check_loan_level_repay.py
Check loan-level aggregations in repayment_history.csv
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import ks_2samp

DATA_DIR = Path('D:/LOGIKA UI/code/data-science-competition-logika-ui-2026')

train = pd.read_csv(DATA_DIR / 'train_applications.csv', parse_dates=['decision_datetime'])
y = train['target'].values
cust_target = dict(zip(train['customer_id'], y))
cust_cutoff = dict(zip(train['customer_id'], train['decision_datetime']))

repay = pd.read_csv(DATA_DIR / 'repayment_history.csv', parse_dates=['due_date', 'last_payment_datetime'])
# filter by customer cutoff
repay['cutoff'] = repay['customer_id'].map(cust_cutoff)
repay = repay[repay['cutoff'].notna()]
repay = repay[repay['due_date'] <= repay['cutoff']]

# Group by loan_id
loan_summary = repay.groupby(['customer_id', 'loan_id']).agg(
    loan_due_sum=('amount_due', 'sum'),
    loan_paid_sum=('amount_paid', 'sum'),
    loan_max_dpd=('max_days_past_due', 'max'),
    loan_installments=('installment_id', 'count'),
    loan_last_due=('due_date', 'max'),
).reset_index()

loan_summary['is_loan_default'] = (loan_summary['loan_max_dpd'] >= 90).astype(int)
loan_summary['is_loan_late'] = (loan_summary['loan_max_dpd'] > 0).astype(int)

# Aggregate to customer level
cust_loan_agg = loan_summary.groupby('customer_id').agg(
    n_loans=('loan_id', 'count'),
    n_defaulted_loans=('is_loan_default', 'sum'),
    has_any_defaulted_loan=('is_loan_default', 'max'),
    n_late_loans=('is_loan_late', 'sum'),
    loan_max_dpd_across_loans=('loan_max_dpd', 'max'),
    loan_mean_dpd_across_loans=('loan_max_dpd', 'mean'),
).reset_index()

# Most recent loan
loan_sorted = loan_summary.sort_values('loan_last_due')
most_recent_loan = loan_sorted.groupby('customer_id').last().reset_index()
most_recent_loan = most_recent_loan[['customer_id', 'loan_max_dpd', 'is_loan_default', 'is_loan_late']].rename(
    columns={
        'loan_max_dpd': 'most_recent_loan_max_dpd',
        'is_loan_default': 'most_recent_loan_is_default',
        'is_loan_late': 'most_recent_loan_is_late',
    }
)

cust_loan_agg = cust_loan_agg.merge(most_recent_loan, on='customer_id', how='left')
train = train.merge(cust_loan_agg, on='customer_id', how='left')

print("Loan-level features evaluation:")
for col in ['n_loans', 'n_defaulted_loans', 'has_any_defaulted_loan', 'loan_max_dpd_across_loans', 'loan_mean_dpd_across_loans', 'most_recent_loan_max_dpd', 'most_recent_loan_is_default', 'most_recent_loan_is_late']:
    corr = train[col].corr(pd.Series(y))
    vals = train[col].fillna(-1).values
    ks = ks_2samp(vals[y==1], vals[y==0]).statistic
    print(f"  {col:30s} corr={corr:.4f} | KS={ks:.4f}")
