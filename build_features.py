"""
build_features.py
Comprehensive Feature Engineering for DSC LOGIKA UI 2026
Strictly following cutoff: event_datetime <= decision_datetime
Saves feature matrices to artifacts/features.joblib
"""

import gc
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings('ignore')

DATA_DIR = Path('D:/LOGIKA UI/code/data-science-competition-logika-ui-2026')
OUTPUT_DIR = Path('D:/LOGIKA UI/code/artifacts')
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("Step 1: Loading Applications and Profiles...")
print("=" * 70)

train = pd.read_csv(DATA_DIR / 'train_applications.csv', parse_dates=['application_datetime', 'decision_datetime'])
test = pd.read_csv(DATA_DIR / 'test_applications.csv', parse_dates=['application_datetime', 'decision_datetime'])

n_train = len(train)
n_test = len(test)
y_train = train['target'].copy().to_numpy()

apps = pd.concat([train.drop(columns=['target'], errors='ignore'), test], ignore_index=True)
apps['_row_id'] = np.arange(len(apps))
apps['_cutoff'] = apps['decision_datetime']

profiles = pd.read_csv(DATA_DIR / 'customer_profiles.csv')
base = apps.merge(profiles, on='customer_id', how='left')

# Application & Profile Derived Features
base['application_hour'] = base['application_datetime'].dt.hour
base['application_dow'] = base['application_datetime'].dt.dayofweek
base['decision_hour'] = base['decision_datetime'].dt.hour
base['decision_dow'] = base['decision_datetime'].dt.dayofweek
base['lead_hours'] = (base['decision_datetime'] - base['application_datetime']).dt.total_seconds() / 3600

# Financial Ratios
income = base['declared_monthly_income'].replace(0, np.nan)
base['dti'] = base['proposed_monthly_installment'] / income
base['loan_to_income'] = base['loan_amount'] / (12 * income)
base['installment_to_loan'] = base['proposed_monthly_installment'] / (base['loan_amount'] + 1)
base['total_repayment'] = base['proposed_monthly_installment'] * base['term_months']
base['total_interest_amount'] = base['total_repayment'] - base['loan_amount']
base['interest_ratio'] = base['total_interest_amount'] / (base['loan_amount'] + 1)
base['disposable_income'] = base['declared_monthly_income'] - base['proposed_monthly_installment']
base['disposable_per_dependent'] = base['disposable_income'] / (base['dependents'] + 1)
base['income_per_dependent'] = base['declared_monthly_income'] / (base['dependents'] + 1)

# Demographic Ratios
base['employment_tenure_years'] = base['employment_tenure_months'] / 12.0
base['emp_to_age_ratio'] = base['employment_tenure_months'] / (base['age'] * 12.0 - 216.0 + 1)

# Interest Rate Tiers (discrete pricing signal)
base['rate_tier'] = base['annual_interest_rate'].round(2)
base['is_highest_rate'] = (base['rate_tier'] >= 0.28).astype('int8')
base['is_lowest_rate'] = (base['rate_tier'] <= 0.16).astype('int8')

print(f"Base application features: {base.shape}")

print("\n" + "=" * 70)
print("Step 2: Processing Bureau Observations...")
print("=" * 70)

bureau = pd.read_csv(DATA_DIR / 'bureau_observations.csv', parse_dates=['available_datetime'])

# Match on application_id for current application report
b_curr = bureau.merge(base[['_row_id', 'application_id', '_cutoff']], on='application_id', how='inner')
b_curr = b_curr[b_curr['available_datetime'] <= b_curr['_cutoff']]

b_cols = ['score', 'active_accounts', 'outstanding_balance', 'revolving_credit_limit',
          'revolving_outstanding_balance', 'utilization', 'inquiries_last_6m', 'delinquent_accounts_last_12m']

b_curr_agg = b_curr.groupby(['_row_id', 'application_id'])[b_cols].agg(['mean', 'max', 'min', 'sum', 'std'])
b_curr_agg.columns = [f'bureau_curr_{c}_{s}' for c, s in b_curr_agg.columns]
b_curr_agg['bureau_curr_obs_count'] = b_curr.groupby(['_row_id', 'application_id']).size()

# Provider specific scores
for prov in ['Bureau_A', 'Bureau_B', 'Bureau_C']:
    prov_sub = b_curr[b_curr['bureau_provider'] == prov]
    if not prov_sub.empty:
        p_agg = prov_sub.groupby(['_row_id', 'application_id'])[['score', 'outstanding_balance', 'utilization', 'inquiries_last_6m', 'delinquent_accounts_last_12m']].mean()
        p_agg.columns = [f'bureau_{prov}_{c}' for c in p_agg.columns]
        b_curr_agg = b_curr_agg.join(p_agg)

# Bureau Score spread
if 'bureau_curr_score_max' in b_curr_agg and 'bureau_curr_score_min' in b_curr_agg:
    b_curr_agg['bureau_curr_score_spread'] = b_curr_agg['bureau_curr_score_max'] - b_curr_agg['bureau_curr_score_min']

# Aggregate utilization
if 'bureau_curr_revolving_outstanding_balance_sum' in b_curr_agg and 'bureau_curr_revolving_credit_limit_sum' in b_curr_agg:
    b_curr_agg['bureau_agg_utilization'] = b_curr_agg['bureau_curr_revolving_outstanding_balance_sum'] / (b_curr_agg['bureau_curr_revolving_credit_limit_sum'] + 1)

base = base.merge(b_curr_agg.reset_index(), on=['_row_id', 'application_id'], how='left')
del b_curr, b_curr_agg, bureau
gc.collect()
print(f"Base after Bureau: {base.shape}")

print("\n" + "=" * 70)
print("Step 3: Processing Previous Applications...")
print("=" * 70)

prev = pd.read_csv(DATA_DIR / 'previous_applications.csv', parse_dates=['decision_datetime'])
prev_m = prev.merge(base[['_row_id', 'customer_id', '_cutoff']], on='customer_id', how='inner')
prev_m = prev_m[prev_m['decision_datetime'] <= prev_m['_cutoff']]

prev_m['is_approved'] = (prev_m['decision'] == 'approved').astype('int8')
prev_m['is_rejected'] = (prev_m['decision'] == 'rejected').astype('int8')
prev_m['is_cancelled'] = (prev_m['decision'] == 'cancelled').astype('int8')

prev_agg = prev_m.groupby(['_row_id', 'customer_id']).agg(
    prev_app_count=('application_id', 'count'),
    prev_approved_count=('is_approved', 'sum'),
    prev_rejected_count=('is_rejected', 'sum'),
    prev_cancelled_count=('is_cancelled', 'sum'),
    prev_approval_rate=('is_approved', 'mean'),
    prev_rejection_rate=('is_rejected', 'mean'),
    prev_req_amt_mean=('requested_amount', 'mean'),
    prev_req_amt_max=('requested_amount', 'max'),
    prev_req_amt_sum=('requested_amount', 'sum'),
    prev_appr_amt_mean=('approved_amount', 'mean'),
    prev_appr_amt_max=('approved_amount', 'max'),
    prev_appr_amt_sum=('approved_amount', 'sum'),
    prev_term_mean=('term_months', 'mean'),
    prev_rate_mean=('annual_interest_rate', 'mean'),
    prev_rate_max=('annual_interest_rate', 'max'),
    prev_installment_mean=('scheduled_monthly_installment', 'mean'),
    prev_installment_sum=('scheduled_monthly_installment', 'sum'),
    prev_latest_decision_dt=('decision_datetime', 'max'),
).reset_index()

base = base.merge(prev_agg, on=['_row_id', 'customer_id'], how='left')
base['days_since_latest_prev_app'] = (base['_cutoff'] - base['prev_latest_decision_dt']).dt.total_seconds() / 86400.0
base.drop(columns=['prev_latest_decision_dt'], inplace=True)
base['has_prev_applications'] = base['prev_app_count'].notna().astype('int8')

# Ratio to current application
base['loan_expansion_ratio'] = base['loan_amount'] / (base['prev_appr_amt_max'].replace(0, np.nan))
base['rate_delta_from_prev'] = base['annual_interest_rate'] - base['prev_rate_mean']
base['installment_growth_ratio'] = base['proposed_monthly_installment'] / (base['prev_installment_mean'].replace(0, np.nan))

del prev, prev_m, prev_agg
gc.collect()
print(f"Base after Previous Applications: {base.shape}")

print("\n" + "=" * 70)
print("Step 4: Processing Repayment History (Critical Signal)...")
print("=" * 70)

repay = pd.read_csv(DATA_DIR / 'repayment_history.csv', parse_dates=['due_date', 'last_payment_datetime'])
rep_m = repay.merge(base[['_row_id', 'customer_id', '_cutoff']], on='customer_id', how='inner')
rep_m = rep_m[rep_m['due_date'] <= rep_m['_cutoff']]

# Determine paid status and amount at cutoff
paid_by_cutoff = rep_m['last_payment_datetime'].notna() & (rep_m['last_payment_datetime'] <= rep_m['_cutoff'])
rep_m['_paid_by_cutoff'] = paid_by_cutoff.astype('int8')
rep_m['_amount_paid_at_cutoff'] = np.where(paid_by_cutoff, rep_m['amount_paid'], 0)
rep_m['_unpaid_balance'] = np.maximum(0, rep_m['amount_due'] - rep_m['_amount_paid_at_cutoff'])

# DPD flags
rep_m['_has_dpd'] = (rep_m['days_past_due'] > 0).astype('int8')
rep_m['_has_max_dpd'] = (rep_m['max_days_past_due'] > 0).astype('int8')
rep_m['_dpd30'] = (rep_m['max_days_past_due'] >= 30).astype('int8')
rep_m['_dpd60'] = (rep_m['max_days_past_due'] >= 60).astype('int8')
rep_m['_dpd90'] = (rep_m['max_days_past_due'] >= 90).astype('int8')

# Days before cutoff
rep_m['_days_before_cutoff'] = (rep_m['_cutoff'] - rep_m['due_date']).dt.total_seconds() / 86400.0

# Overall repayment summary
rep_agg = rep_m.groupby(['_row_id', 'customer_id']).agg(
    rep_total_installments=('installment_id', 'count'),
    rep_paid_count=('_paid_by_cutoff', 'sum'),
    rep_due_sum=('amount_due', 'sum'),
    rep_paid_sum=('_amount_paid_at_cutoff', 'sum'),
    rep_unpaid_sum=('_unpaid_balance', 'sum'),
    rep_unpaid_count=('_unpaid_balance', lambda x: (x > 0).sum()),
    # Key DPD features:
    rep_dpd_mean=('days_past_due', 'mean'),
    rep_dpd_max=('days_past_due', 'max'),
    rep_dpd_sum=('days_past_due', 'sum'),
    rep_max_dpd_mean=('max_days_past_due', 'mean'),
    rep_max_dpd_max=('max_days_past_due', 'max'),
    rep_max_dpd_std=('max_days_past_due', 'std'),
    rep_late_count=('_has_max_dpd', 'sum'),
    rep_dpd30_count=('_dpd30', 'sum'),
    rep_dpd60_count=('_dpd60', 'sum'),
    rep_dpd90_count=('_dpd90', 'sum'),
).reset_index()

# Ratios
n_inst = rep_agg['rep_total_installments'].replace(0, np.nan)
rep_agg['rep_paid_ratio'] = rep_agg['rep_paid_count'] / n_inst
rep_agg['rep_paid_amount_ratio'] = rep_agg['rep_paid_sum'] / (rep_agg['rep_due_sum'] + 1)
rep_agg['rep_late_ratio'] = rep_agg['rep_late_count'] / n_inst
rep_agg['rep_dpd30_ratio'] = rep_agg['rep_dpd30_count'] / n_inst
rep_agg['rep_dpd60_ratio'] = rep_agg['rep_dpd60_count'] / n_inst
rep_agg['rep_dpd90_ratio'] = rep_agg['rep_dpd90_count'] / n_inst

# Recent repayment horizons (90d, 180d, 365d)
for days in [90, 180, 365]:
    recent = rep_m[rep_m['_days_before_cutoff'] <= days]
    if not recent.empty:
        r_agg = recent.groupby(['_row_id', 'customer_id']).agg(**{
            f'rep_{days}d_inst_count': ('installment_id', 'count'),
            f'rep_{days}d_max_dpd_max': ('max_days_past_due', 'max'),
            f'rep_{days}d_max_dpd_mean': ('max_days_past_due', 'mean'),
            f'rep_{days}d_dpd30_count': ('_dpd30', 'sum'),
            f'rep_{days}d_unpaid_sum': ('_unpaid_balance', 'sum'),
        }).reset_index()
        rep_agg = rep_agg.merge(r_agg, on=['_row_id', 'customer_id'], how='left')

base = base.merge(rep_agg, on=['_row_id', 'customer_id'], how='left')
base['has_repayment_history'] = base['rep_total_installments'].notna().astype('int8')

del repay, rep_m, rep_agg
gc.collect()
print(f"Base after Repayment: {base.shape}")

print("\n" + "=" * 70)
print("Step 5: Processing Account Behavior...")
print("=" * 70)

acct = pd.read_csv(DATA_DIR / 'account_behavior.csv', parse_dates=['month'])
acct_m = acct.merge(base[['_row_id', 'customer_id', '_cutoff']], on='customer_id', how='inner')
acct_m = acct_m[acct_m['month'] <= acct_m['_cutoff']]

acct_m['net_inflow'] = acct_m['inflows'] - acct_m['outflows']
acct_m['outflow_to_inflow'] = acct_m['outflows'] / (acct_m['inflows'] + 1)
acct_m['_days_before_cutoff'] = (acct_m['_cutoff'] - acct_m['month']).dt.total_seconds() / 86400.0

acct_val_cols = ['opening_balance', 'closing_balance', 'inflows', 'outflows', 'net_inflow',
                 'income_like_credits', 'average_balance', 'minimum_balance', 'low_balance_days']

acct_agg = acct_m.groupby(['_row_id', 'customer_id']).agg(
    acct_months_count=('month', 'nunique'),
    acct_close_mean=('closing_balance', 'mean'),
    acct_close_min=('closing_balance', 'min'),
    acct_close_std=('closing_balance', 'std'),
    acct_avgbal_mean=('average_balance', 'mean'),
    acct_avgbal_min=('average_balance', 'min'),
    acct_minbal_min=('minimum_balance', 'min'),
    acct_inflows_mean=('inflows', 'mean'),
    acct_inflows_sum=('inflows', 'sum'),
    acct_outflows_mean=('outflows', 'mean'),
    acct_outflows_sum=('outflows', 'sum'),
    acct_netflow_sum=('net_inflow', 'sum'),
    acct_income_cr_mean=('income_like_credits', 'mean'),
    acct_income_cr_sum=('income_like_credits', 'sum'),
    acct_lowbal_mean=('low_balance_days', 'mean'),
    acct_lowbal_sum=('low_balance_days', 'sum'),
    acct_lowbal_max=('low_balance_days', 'max'),
).reset_index()

# Most recent month features (sorted)
acct_last = acct_m.sort_values('month').groupby(['_row_id', 'customer_id']).last().reset_index()
acct_last_feats = acct_last[['_row_id', 'customer_id', 'closing_balance', 'average_balance', 'minimum_balance', 'inflows', 'outflows', 'low_balance_days', 'net_inflow']].rename(
    columns={c: f'acct_latest_{c}' for c in ['closing_balance', 'average_balance', 'minimum_balance', 'inflows', 'outflows', 'low_balance_days', 'net_inflow']}
)
acct_agg = acct_agg.merge(acct_last_feats, on=['_row_id', 'customer_id'], how='left')

# Recent 90d (3 months)
acct_90 = acct_m[acct_m['_days_before_cutoff'] <= 95]
if not acct_90.empty:
    a90_agg = acct_90.groupby(['_row_id', 'customer_id']).agg(
        acct_90d_avgbal_mean=('average_balance', 'mean'),
        acct_90d_inflows_sum=('inflows', 'sum'),
        acct_90d_outflows_sum=('outflows', 'sum'),
        acct_90d_netflow=('net_inflow', 'sum'),
        acct_90d_lowbal_sum=('low_balance_days', 'sum'),
    ).reset_index()
    acct_agg = acct_agg.merge(a90_agg, on=['_row_id', 'customer_id'], how='left')

base = base.merge(acct_agg, on=['_row_id', 'customer_id'], how='left')
base['has_account_behavior'] = base['acct_months_count'].notna().astype('int8')

# Financial interaction ratios
base['installment_coverage_avgbal'] = base['acct_avgbal_mean'] / (base['proposed_monthly_installment'] + 1)
base['installment_coverage_latest'] = base['acct_latest_average_balance'] / (base['proposed_monthly_installment'] + 1)
base['cash_runway_months'] = base['acct_latest_closing_balance'] / (base['acct_outflows_mean'] + 1)
base['income_cred_to_declared'] = base['acct_income_cr_mean'] / (income)

del acct, acct_m, acct_agg, acct_last, acct_last_feats
gc.collect()
print(f"Base after Account Behavior: {base.shape}")

print("\n" + "=" * 70)
print("Step 6: Processing E-Commerce Transactions (Rich Aggregations)...")
print("=" * 70)

# Process in chunks to maintain low memory usage
ecom_chunks_agg = []
customers_set = set(base['customer_id'].unique())

for chunk in pd.read_csv(DATA_DIR / 'ecommerce_transactions.csv', parse_dates=['transaction_datetime'], chunksize=500_000):
    chunk = chunk[chunk['customer_id'].isin(customers_set)]
    if chunk.empty:
        continue
    j = chunk.merge(base[['_row_id', 'customer_id', '_cutoff']], on='customer_id', how='inner')
    j = j[j['transaction_datetime'] <= j['_cutoff']]
    if j.empty:
        continue
    j['_days_before_cutoff'] = (j['_cutoff'] - j['transaction_datetime']).dt.total_seconds() / 86400.0
    j['_is_completed'] = (j['order_status'] == 'completed').astype('int8')
    j['_is_cancelled'] = (j['order_status'] == 'cancelled').astype('int8')
    j['_completed_spend'] = np.where(j['_is_completed'] == 1, j['item_total'], 0)
    
    # Specific categories spend
    for cat in ['groceries', 'fashion', 'electronics', 'personal_care', 'health']:
        j[f'_cat_{cat}'] = np.where((j['_is_completed'] == 1) & (j['product_category'] == cat), j['item_total'], 0)
    
    # Specific payment methods
    for pay in ['COD', 'e_wallet', 'bank_transfer', 'credit_card']:
        j[f'_pay_{pay}'] = np.where((j['_is_completed'] == 1) & (j['payment_method'] == pay), 1, 0)
        j[f'_pay_amt_{pay}'] = np.where((j['_is_completed'] == 1) & (j['payment_method'] == pay), j['item_total'], 0)
    
    g = j.groupby(['_row_id', 'customer_id']).agg(
        ec_event_count=('order_item_id', 'count'),
        ec_completed_count=('_is_completed', 'sum'),
        ec_cancelled_count=('_is_cancelled', 'sum'),
        ec_spend_sum=('_completed_spend', 'sum'),
        ec_spend_max=('_completed_spend', 'max'),
        ec_groceries_sum=('_cat_groceries', 'sum'),
        ec_fashion_sum=('_cat_fashion', 'sum'),
        ec_electronics_sum=('_cat_electronics', 'sum'),
        ec_personal_care_sum=('_cat_personal_care', 'sum'),
        ec_health_sum=('_cat_health', 'sum'),
        ec_cod_count=('_pay_COD', 'sum'),
        ec_cod_spend=('_pay_amt_COD', 'sum'),
        ec_ewallet_count=('_pay_e_wallet', 'sum'),
        ec_bank_transfer_count=('_pay_bank_transfer', 'sum'),
        ec_credit_card_count=('_pay_credit_card', 'sum'),
    ).reset_index()
    
    # Time horizons
    for d in [90, 180]:
        recent_e = j[j['_days_before_cutoff'] <= d]
        if not recent_e.empty:
            re = recent_e.groupby(['_row_id', 'customer_id']).agg(**{
                f'ec_{d}d_completed_count': ('_is_completed', 'sum'),
                f'ec_{d}d_spend_sum': ('_completed_spend', 'sum'),
            }).reset_index()
            g = g.merge(re, on=['_row_id', 'customer_id'], how='left')
    
    ecom_chunks_agg.append(g)
    del chunk, j, g
    gc.collect()

print("  Merging e-commerce chunks...")
if ecom_chunks_agg:
    all_ec = pd.concat(ecom_chunks_agg, ignore_index=True)
    ec_final = all_ec.groupby(['_row_id', 'customer_id'], as_index=False).sum(numeric_only=True)
    
    # Ecom derived ratios
    comp_n = ec_final['ec_completed_count'].replace(0, np.nan)
    ec_final['ec_spend_mean'] = ec_final['ec_spend_sum'] / comp_n
    ec_final['ec_cancel_rate'] = ec_final['ec_cancelled_count'] / (ec_final['ec_completed_count'] + ec_final['ec_cancelled_count'] + 1)
    ec_final['ec_cod_ratio'] = ec_final['ec_cod_count'] / comp_n
    ec_final['ec_groceries_ratio'] = ec_final['ec_groceries_sum'] / (ec_final['ec_spend_sum'] + 1)
    ec_final['ec_discretionary_spend'] = ec_final['ec_fashion_sum'] + ec_final['ec_electronics_sum'] + ec_final['ec_personal_care_sum']
    ec_final['ec_discretionary_ratio'] = ec_final['ec_discretionary_spend'] / (ec_final['ec_spend_sum'] + 1)
    
    base = base.merge(ec_final, on=['_row_id', 'customer_id'], how='left')
    del all_ec, ec_final, ecom_chunks_agg
    gc.collect()

base['has_ecom_history'] = base['ec_completed_count'].notna().astype('int8')
base['ecom_spend_to_income'] = base['ec_spend_sum'] / (income * 12.0)

print(f"Base after E-Commerce: {base.shape}")

print("\n" + "=" * 70)
print("Step 7: Final Feature Matrix Preparation...")
print("=" * 70)

# Drop non-feature identifiers and timestamps
drop_cols = ['_row_id', 'application_id', 'customer_id', 'application_datetime', 'decision_datetime', '_cutoff']
feature_frame = base.drop(columns=[c for c in drop_cols if c in base.columns]).copy()

# One-hot encode categoricals
cat_cols = feature_frame.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
print(f"Categorical columns to encode: {cat_cols}")
X_all = pd.get_dummies(feature_frame, columns=cat_cols, dummy_na=True, dtype='float32')
X_all = X_all.replace([np.inf, -np.inf], np.nan)

# Save split matrices
X_train = X_all.iloc[:n_train].reset_index(drop=True)
X_test = X_all.iloc[n_train:].reset_index(drop=True)
feature_names = list(X_train.columns)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape:  {X_test.shape}")
print(f"Total features: {len(feature_names)}")

bundle = {
    'X_train': X_train,
    'y_train': y_train,
    'X_test': X_test,
    'test_application_ids': test['application_id'].to_numpy(),
    'feature_names': feature_names
}

output_path = OUTPUT_DIR / 'features_v4.joblib'
joblib.dump(bundle, output_path, compress=3)
print(f"\nSaved feature matrices to: {output_path}")
print("Feature Engineering completed successfully!")

