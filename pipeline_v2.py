"""
DSC LOGIKA UI 2026 — Credit Default Prediction Pipeline v2
==========================================================
Optimized for memory: pre-aggregate per customer, then merge
Metric: KS Statistic  |  Target: >= 0.57
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import gc

from scipy.stats import ks_2samp, rankdata
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

SEED = 42
np.random.seed(SEED)

DATA_DIR = "D:/LOGIKA UI/code/data-science-competition-logika-ui-2026"
SUBMISSION_PATH = "D:/LOGIKA UI/code/artifacts/DSC0102_JanjiLombaTerakhirTahunIni_Submisi.csv"

def calculate_ks_score(y_true, y_pred):
    preds_bad = y_pred[y_true == 1]
    preds_good = y_pred[y_true == 0]
    if len(preds_bad) == 0 or len(preds_good) == 0:
        return 0.0
    return float(ks_2samp(preds_bad, preds_good).statistic)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("=" * 60)
print("1. Loading data...")

train = pd.read_csv(f"{DATA_DIR}/train_applications.csv")
test = pd.read_csv(f"{DATA_DIR}/test_applications.csv")
customers = pd.read_csv(f"{DATA_DIR}/customer_profiles.csv")

print(f"Train: {train.shape}, Test: {test.shape}")
print(f"Target dist: {train['target'].value_counts().to_dict()}")

for df in [train, test]:
    df['application_datetime'] = pd.to_datetime(df['application_datetime'])
    df['decision_datetime'] = pd.to_datetime(df['decision_datetime'])

train['is_train'] = 1
test['is_train'] = 0
test['target'] = -1
all_apps = pd.concat([train, test], ignore_index=True)

# ============================================================
# 2A. Application + Customer features
# ============================================================
print("\n2A. Application + Customer features...")

all_apps['loan_to_installment'] = all_apps['loan_amount'] / (all_apps['proposed_monthly_installment'] + 1)
all_apps['decision_time_hours'] = (all_apps['decision_datetime'] - all_apps['application_datetime']).dt.total_seconds() / 3600
all_apps['app_hour'] = all_apps['application_datetime'].dt.hour
all_apps['app_dayofweek'] = all_apps['application_datetime'].dt.dayofweek
all_apps['app_month'] = all_apps['application_datetime'].dt.month
all_apps['loan_purpose_enc'] = all_apps['loan_purpose'].astype('category').cat.codes

all_apps = all_apps.merge(customers, on='customer_id', how='left')

all_apps['income_to_installment'] = all_apps['declared_monthly_income'] / (all_apps['proposed_monthly_installment'] + 1)
all_apps['loan_to_income'] = all_apps['loan_amount'] / (all_apps['declared_monthly_income'] + 1)
all_apps['dti_ratio'] = all_apps['proposed_monthly_installment'] / (all_apps['declared_monthly_income'] + 1)
all_apps['income_per_dependent'] = all_apps['declared_monthly_income'] / (all_apps['dependents'] + 1)
all_apps['age_employment_ratio'] = all_apps['age'] / (all_apps['employment_tenure_months'] + 1)
all_apps['total_repayment'] = all_apps['proposed_monthly_installment'] * all_apps['term_months']
all_apps['total_interest'] = all_apps['total_repayment'] - all_apps['loan_amount']
all_apps['interest_to_loan'] = all_apps['total_interest'] / (all_apps['loan_amount'] + 1)

all_apps['region_enc'] = all_apps['region'].astype('category').cat.codes
all_apps['employment_type_enc'] = all_apps['employment_type'].astype('category').cat.codes
all_apps['housing_status_enc'] = all_apps['housing_status'].astype('category').cat.codes

print(f"  Shape after customer merge: {all_apps.shape}")

# ============================================================
# 2B. Previous Applications features (loop per row — optimized)
# ============================================================
print("\n2B. Previous applications features...")

prev_apps = pd.read_csv(f"{DATA_DIR}/previous_applications.csv")
prev_apps['decision_datetime'] = pd.to_datetime(prev_apps['decision_datetime'])

# Build a dict: customer_id -> sorted list of prev apps
prev_by_cust = {}
for _, row in prev_apps.iterrows():
    cid = row['customer_id']
    if cid not in prev_by_cust:
        prev_by_cust[cid] = []
    prev_by_cust[cid].append(row)

def get_prev_features(cust_id, cutoff_dt):
    if cust_id not in prev_by_cust:
        return {}
    
    apps = [a for a in prev_by_cust[cust_id] if a['decision_datetime'] <= cutoff_dt]
    if not apps:
        return {}
    
    n = len(apps)
    approved = [a for a in apps if a['decision'] == 'approved']
    rejected = [a for a in apps if a['decision'] == 'rejected']
    
    feats = {
        'prev_app_count': n,
        'prev_approved_count': len(approved),
        'prev_rejected_count': len(rejected),
        'prev_cancelled_count': n - len(approved) - len(rejected),
        'prev_approval_rate': len(approved) / n,
        'prev_requested_mean': np.mean([a['requested_amount'] for a in apps]),
        'prev_requested_max': np.max([a['requested_amount'] for a in apps]),
        'prev_requested_sum': np.sum([a['requested_amount'] for a in apps]),
        'prev_term_mean': np.mean([a['term_months'] for a in apps]),
    }
    
    if approved:
        approved_amts = [a['approved_amount'] for a in approved if pd.notna(a['approved_amount'])]
        rates = [a['annual_interest_rate'] for a in approved if pd.notna(a['annual_interest_rate'])]
        installments = [a['scheduled_monthly_installment'] for a in approved if pd.notna(a['scheduled_monthly_installment'])]
        
        if approved_amts:
            feats['prev_approved_amt_mean'] = np.mean(approved_amts)
            feats['prev_approved_amt_max'] = np.max(approved_amts)
            feats['prev_approved_amt_sum'] = np.sum(approved_amts)
        if rates:
            feats['prev_rate_mean'] = np.mean(rates)
        if installments:
            feats['prev_installment_mean'] = np.mean(installments)
    
    # Days since last prev app
    last_dt = max(a['decision_datetime'] for a in apps)
    feats['days_since_last_prev'] = (cutoff_dt - last_dt).days
    
    return feats

# This is too slow for 100K rows iterating python dicts. Use vectorized approach instead.
# Better: merge + filter + groupby
print("  Using vectorized merge approach...")

# Create a mapping: for each app, get decision_datetime
app_cutoffs = all_apps[['application_id', 'customer_id', 'decision_datetime']].copy()

# Merge prev_apps with app cutoffs on customer_id
prev_merged = prev_apps.merge(
    app_cutoffs,
    on='customer_id',
    how='inner',
    suffixes=('_prev', '_main')
)
print(f"  prev_merged size before filter: {len(prev_merged)}")

# Apply time cutoff
prev_merged = prev_merged[prev_merged['decision_datetime_prev'] <= prev_merged['decision_datetime_main']]
print(f"  prev_merged size after filter: {len(prev_merged)}")

# Aggregations
prev_agg = prev_merged.groupby('application_id').agg(
    prev_app_count=('application_id_prev', 'count'),
    prev_approved_count=('decision', lambda x: (x == 'approved').sum()),
    prev_rejected_count=('decision', lambda x: (x == 'rejected').sum()),
    prev_requested_mean=('requested_amount', 'mean'),
    prev_requested_max=('requested_amount', 'max'),
    prev_requested_sum=('requested_amount', 'sum'),
    prev_approved_amt_mean=('approved_amount', 'mean'),
    prev_approved_amt_max=('approved_amount', 'max'),
    prev_approved_amt_sum=('approved_amount', 'sum'),
    prev_term_mean=('term_months', 'mean'),
    prev_rate_mean=('annual_interest_rate_prev', 'mean'),
    prev_installment_mean=('scheduled_monthly_installment', 'mean'),
    prev_last_dt=('decision_datetime_prev', 'max'),
).reset_index()

prev_agg['prev_approval_rate'] = prev_agg['prev_approved_count'] / (prev_agg['prev_app_count'] + 1)

all_apps = all_apps.merge(prev_agg, on='application_id', how='left')
all_apps['days_since_last_prev'] = (all_apps['decision_datetime'] - all_apps['prev_last_dt']).dt.days

# Get loan_ids for repayment
approved_loans = prev_merged[prev_merged['decision'] == 'approved'][['application_id', 'loan_id', 'decision_datetime_main']].dropna(subset=['loan_id']).copy()
approved_loans = approved_loans.rename(columns={'application_id': 'main_app_id'})

del prev_merged
gc.collect()
print(f"  Shape after prev apps: {all_apps.shape}")

# ============================================================
# 2C. Repayment History features
# ============================================================
print("\n2C. Repayment history features...")

repayment = pd.read_csv(f"{DATA_DIR}/repayment_history.csv")
repayment['due_date'] = pd.to_datetime(repayment['due_date'])

# Merge repayment with approved loans
repay_merged = repayment.merge(approved_loans, on='loan_id', how='inner')
repay_merged = repay_merged[repay_merged['due_date'] <= repay_merged['decision_datetime_main']]
print(f"  repay_merged size: {len(repay_merged)}")

repay_agg = repay_merged.groupby('main_app_id').agg(
    repay_count=('installment_id', 'count'),
    repay_dpd_mean=('days_past_due', 'mean'),
    repay_dpd_max=('days_past_due', 'max'),
    repay_dpd_std=('days_past_due', 'std'),
    repay_dpd_sum=('days_past_due', 'sum'),
    repay_max_dpd_mean=('max_days_past_due', 'mean'),
    repay_max_dpd_max=('max_days_past_due', 'max'),
    repay_amount_due_mean=('amount_due', 'mean'),
    repay_amount_due_sum=('amount_due', 'sum'),
    repay_amount_paid_mean=('amount_paid', 'mean'),
    repay_amount_paid_sum=('amount_paid', 'sum'),
    repay_dpd_gt0=('days_past_due', lambda x: (x > 0).sum()),
    repay_dpd_gt30=('days_past_due', lambda x: (x > 30).sum()),
    repay_dpd_gt60=('days_past_due', lambda x: (x > 60).sum()),
    repay_dpd_gt90=('days_past_due', lambda x: (x > 90).sum()),
).reset_index().rename(columns={'main_app_id': 'application_id'})

repay_agg['repay_paid_ratio'] = repay_agg['repay_amount_paid_sum'] / (repay_agg['repay_amount_due_sum'] + 1)
repay_agg['repay_late_ratio'] = repay_agg['repay_dpd_gt0'] / (repay_agg['repay_count'] + 1)
repay_agg['repay_severe_late_ratio'] = repay_agg['repay_dpd_gt30'] / (repay_agg['repay_count'] + 1)

all_apps = all_apps.merge(repay_agg, on='application_id', how='left')

del repayment, repay_merged
gc.collect()
print(f"  Shape after repayment: {all_apps.shape}")

# ============================================================
# 2D. Account Behavior features  
# ============================================================
print("\n2D. Account behavior features...")

acct_beh = pd.read_csv(f"{DATA_DIR}/account_behavior.csv")
acct_beh['month'] = pd.to_datetime(acct_beh['month'])

# Pre-aggregate per customer per month (already 1 row per cust-month)
# Merge with cutoffs
acct_cutoffs = all_apps[['application_id', 'customer_id', 'decision_datetime']].copy()
acct_merged = acct_beh.merge(acct_cutoffs, on='customer_id', how='inner')
acct_merged = acct_merged[acct_merged['month'] <= acct_merged['decision_datetime']]
print(f"  acct_merged size: {len(acct_merged)}")

acct_agg = acct_merged.groupby('application_id').agg(
    acct_months=('month', 'nunique'),
    acct_open_bal_mean=('opening_balance', 'mean'),
    acct_close_bal_mean=('closing_balance', 'mean'),
    acct_close_bal_min=('closing_balance', 'min'),
    acct_close_bal_std=('closing_balance', 'std'),
    acct_inflows_mean=('inflows', 'mean'),
    acct_inflows_sum=('inflows', 'sum'),
    acct_inflows_std=('inflows', 'std'),
    acct_outflows_mean=('outflows', 'mean'),
    acct_outflows_sum=('outflows', 'sum'),
    acct_income_credits_mean=('income_like_credits', 'mean'),
    acct_income_credits_sum=('income_like_credits', 'sum'),
    acct_avg_bal_mean=('average_balance', 'mean'),
    acct_avg_bal_min=('average_balance', 'min'),
    acct_min_bal_mean=('minimum_balance', 'mean'),
    acct_min_bal_min=('minimum_balance', 'min'),
    acct_low_bal_days_mean=('low_balance_days', 'mean'),
    acct_low_bal_days_sum=('low_balance_days', 'sum'),
    acct_low_bal_days_max=('low_balance_days', 'max'),
).reset_index()

acct_agg['acct_net_flow'] = acct_agg['acct_inflows_sum'] - acct_agg['acct_outflows_sum']
acct_agg['acct_out_to_in'] = acct_agg['acct_outflows_mean'] / (acct_agg['acct_inflows_mean'] + 1)

# Last 3 months
acct_last3 = acct_merged.sort_values('month').groupby('application_id').tail(3)
acct_l3 = acct_last3.groupby('application_id').agg(
    acct_l3_close_mean=('closing_balance', 'mean'),
    acct_l3_inflows_mean=('inflows', 'mean'),
    acct_l3_outflows_mean=('outflows', 'mean'),
    acct_l3_low_bal_mean=('low_balance_days', 'mean'),
    acct_l3_avg_bal_mean=('average_balance', 'mean'),
    acct_l3_min_bal_min=('minimum_balance', 'min'),
).reset_index()

all_apps = all_apps.merge(acct_agg, on='application_id', how='left')
all_apps = all_apps.merge(acct_l3, on='application_id', how='left')

del acct_beh, acct_merged, acct_last3
gc.collect()
print(f"  Shape after account: {all_apps.shape}")

# ============================================================
# 2E. Bureau features (joined via application_id directly!)
# ============================================================
print("\n2E. Bureau features...")

bureau = pd.read_csv(f"{DATA_DIR}/bureau_observations.csv")
bureau['available_datetime'] = pd.to_datetime(bureau['available_datetime'])

# Bureau has application_id directly — join on it
bureau_cutoffs = all_apps[['application_id', 'decision_datetime']].copy()
bureau_merged = bureau.merge(bureau_cutoffs, on='application_id', how='inner')
bureau_merged = bureau_merged[bureau_merged['available_datetime'] <= bureau_merged['decision_datetime']]
print(f"  bureau_merged size: {len(bureau_merged)}")

bureau_agg = bureau_merged.groupby('application_id').agg(
    bur_count=('bureau_observation_id', 'count'),
    bur_providers=('bureau_provider', 'nunique'),
    bur_score_mean=('score', 'mean'),
    bur_score_min=('score', 'min'),
    bur_score_max=('score', 'max'),
    bur_score_std=('score', 'std'),
    bur_active_accts_mean=('active_accounts', 'mean'),
    bur_active_accts_max=('active_accounts', 'max'),
    bur_outstanding_mean=('outstanding_balance', 'mean'),
    bur_outstanding_max=('outstanding_balance', 'max'),
    bur_outstanding_sum=('outstanding_balance', 'sum'),
    bur_rev_limit_mean=('revolving_credit_limit', 'mean'),
    bur_rev_limit_sum=('revolving_credit_limit', 'sum'),
    bur_rev_out_mean=('revolving_outstanding_balance', 'mean'),
    bur_util_mean=('utilization', 'mean'),
    bur_util_max=('utilization', 'max'),
    bur_inquiries_mean=('inquiries_last_6m', 'mean'),
    bur_inquiries_max=('inquiries_last_6m', 'max'),
    bur_inquiries_sum=('inquiries_last_6m', 'sum'),
    bur_delinq_mean=('delinquent_accounts_last_12m', 'mean'),
    bur_delinq_max=('delinquent_accounts_last_12m', 'max'),
    bur_delinq_sum=('delinquent_accounts_last_12m', 'sum'),
).reset_index()

bur_score_range = bureau_merged.groupby('application_id')['score'].agg(lambda x: x.max() - x.min()).reset_index()
bur_score_range.columns = ['application_id', 'bur_score_range']

bureau_agg['bur_debt_to_limit'] = bureau_agg['bur_rev_out_mean'] / (bureau_agg['bur_rev_limit_mean'] + 1)
bureau_agg['bur_has_delinq'] = (bureau_agg['bur_delinq_sum'] > 0).astype(int)

all_apps = all_apps.merge(bureau_agg, on='application_id', how='left')
all_apps = all_apps.merge(bur_score_range, on='application_id', how='left')

del bureau, bureau_merged
gc.collect()
print(f"  Shape after bureau: {all_apps.shape}")

# ============================================================
# 2F. E-commerce features — CHUNKED approach
# ============================================================
print("\n2F. E-commerce features (chunked)...")

ecom_cutoffs = all_apps[['application_id', 'customer_id', 'decision_datetime']].copy()

# Read e-commerce in chunks and aggregate
chunk_size = 500000
ecom_results = {}

for chunk in pd.read_csv(f"{DATA_DIR}/ecommerce_transactions.csv", chunksize=chunk_size):
    chunk['transaction_datetime'] = pd.to_datetime(chunk['transaction_datetime'])
    
    # Merge with cutoffs
    merged = chunk.merge(ecom_cutoffs, on='customer_id', how='inner')
    merged = merged[merged['transaction_datetime'] <= merged['decision_datetime']]
    
    if len(merged) == 0:
        continue
    
    # Separate completed vs cancelled
    completed = merged[merged['order_status'] == 'completed']
    cancelled = merged[merged['order_status'] == 'cancelled']
    
    for app_id, grp in completed.groupby('application_id'):
        if app_id not in ecom_results:
            ecom_results[app_id] = {
                'items': [], 'totals': [], 'orders': set(), 'categories': set(),
                'quantities': [], 'payments': [], 'cancelled_count': 0, 'cancelled_value': 0
            }
        ecom_results[app_id]['items'].extend(grp['item_total'].tolist())
        ecom_results[app_id]['totals'].extend(grp['order_total'].tolist())
        ecom_results[app_id]['orders'].update(grp['order_id'].tolist())
        ecom_results[app_id]['categories'].update(grp['product_category'].tolist())
        ecom_results[app_id]['quantities'].extend(grp['quantity'].tolist())
        ecom_results[app_id]['payments'].extend(grp['payment_method'].tolist())
    
    for app_id, grp in cancelled.groupby('application_id'):
        if app_id not in ecom_results:
            ecom_results[app_id] = {
                'items': [], 'totals': [], 'orders': set(), 'categories': set(),
                'quantities': [], 'payments': [], 'cancelled_count': 0, 'cancelled_value': 0
            }
        ecom_results[app_id]['cancelled_count'] += len(grp)
        ecom_results[app_id]['cancelled_value'] += grp['item_total'].sum()
    
    del merged, completed, cancelled
    gc.collect()
    print(f"  Processed chunk...")

# Convert to DataFrame
print("  Converting e-commerce features to DataFrame...")
ecom_feats = []
for app_id, data in ecom_results.items():
    items = data['items']
    feat = {
        'application_id': app_id,
        'ecom_completed_count': len(items),
        'ecom_total_spending': sum(items) if items else 0,
        'ecom_avg_item': np.mean(items) if items else 0,
        'ecom_max_item': max(items) if items else 0,
        'ecom_categories': len(data['categories']),
        'ecom_unique_orders': len(data['orders']),
        'ecom_quantity_mean': np.mean(data['quantities']) if data['quantities'] else 0,
        'ecom_quantity_sum': sum(data['quantities']) if data['quantities'] else 0,
        'ecom_cancelled_count': data['cancelled_count'],
        'ecom_cancelled_value': data['cancelled_value'],
    }
    
    # Payment method counts
    if data['payments']:
        from collections import Counter
        pay_counts = Counter(data['payments'])
        for method in ['COD', 'e_wallet', 'bank_transfer', 'debit_card', 'credit_card']:
            feat[f'ecom_pay_{method}'] = pay_counts.get(method, 0)
    
    ecom_feats.append(feat)

ecom_df = pd.DataFrame(ecom_feats)

if len(ecom_df) > 0:
    ecom_df['ecom_cancel_rate'] = ecom_df['ecom_cancelled_count'] / (ecom_df['ecom_completed_count'] + ecom_df['ecom_cancelled_count'] + 1)
    ecom_df['ecom_avg_order_value'] = ecom_df['ecom_total_spending'] / (ecom_df['ecom_unique_orders'] + 1)
    all_apps = all_apps.merge(ecom_df, on='application_id', how='left')

del ecom_results, ecom_feats
gc.collect()
print(f"  Shape after ecom: {all_apps.shape}")

# ============================================================
# 2G. Interaction features
# ============================================================
print("\n2G. Interaction features...")

all_apps['bur_score_x_loan'] = all_apps.get('bur_score_mean', 0) * all_apps['loan_amount']
all_apps['bur_score_x_dti'] = all_apps.get('bur_score_mean', 0) * all_apps['dti_ratio']
all_apps['avg_bal_to_loan'] = all_apps.get('acct_avg_bal_mean', 0) / (all_apps['loan_amount'] + 1)
all_apps['income_cr_to_inst'] = all_apps.get('acct_income_credits_mean', 0) / (all_apps['proposed_monthly_installment'] + 1)
all_apps['ecom_spend_to_income'] = all_apps.get('ecom_total_spending', 0) / (all_apps['declared_monthly_income'] + 1)

all_apps['has_prev'] = all_apps['prev_app_count'].notna().astype(int)
all_apps['has_repay'] = all_apps['repay_count'].notna().astype(int)
all_apps['has_bureau'] = all_apps['bur_count'].notna().astype(int)
all_apps['has_ecom'] = all_apps.get('ecom_completed_count', pd.Series(dtype=float)).notna().astype(int)

# ============================================================
# 3. PREPARE FOR MODELING
# ============================================================
print("\n" + "=" * 60)
print("3. Preparing for modeling...")

drop_cols = ['application_id', 'customer_id', 'application_datetime', 'decision_datetime',
             'is_train', 'target', 'loan_purpose', 'region', 'employment_type', 'housing_status',
             'prev_last_dt']

feature_cols = [c for c in all_apps.columns if c not in drop_cols]
print(f"Features: {len(feature_cols)}")

X_all = all_apps[feature_cols].copy()
y_all = all_apps['target'].copy()
is_train = all_apps['is_train'] == 1

X_train = X_all[is_train].reset_index(drop=True)
y_train = y_all[is_train].reset_index(drop=True).astype(int)
X_test = X_all[~is_train].reset_index(drop=True)

print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

# ============================================================
# 4. LightGBM
# ============================================================
print("\n" + "=" * 60)
print("4. Training LightGBM...")

lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.02,
    'num_leaves': 63,
    'max_depth': 8,
    'min_child_samples': 50,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'verbose': -1,
    'n_jobs': -1,
    'seed': SEED,
    'is_unbalance': True,
}

N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

lgb_oof = np.zeros(len(X_train))
lgb_test = np.zeros(len(X_test))

for fold, (ti, vi) in enumerate(skf.split(X_train, y_train)):
    dtrain = lgb.Dataset(X_train.iloc[ti], y_train.iloc[ti])
    dval = lgb.Dataset(X_train.iloc[vi], y_train.iloc[vi], reference=dtrain)
    
    model = lgb.train(lgb_params, dtrain, 3000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)])
    
    lgb_oof[vi] = model.predict(X_train.iloc[vi])
    lgb_test += model.predict(X_test) / N_FOLDS
    
    ks = calculate_ks_score(y_train.iloc[vi].values, lgb_oof[vi])
    print(f"  Fold {fold+1}: KS={ks:.4f}")

lgb_ks = calculate_ks_score(y_train.values, lgb_oof)
print(f"LightGBM OOF KS: {lgb_ks:.4f}")

# Feature importance
fi = model.feature_importance(importance_type='gain')
fi_names = model.feature_name()
fi_df = pd.DataFrame({'feature': fi_names, 'importance': fi}).sort_values('importance', ascending=False)
print("\nTop 20 features (LGB):")
print(fi_df.head(20).to_string(index=False))

# ============================================================
# 5. XGBoost
# ============================================================
print("\n" + "=" * 60)
print("5. Training XGBoost...")

xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'learning_rate': 0.02,
    'max_depth': 7,
    'min_child_weight': 50,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'gamma': 0.1,
    'seed': SEED,
    'scale_pos_weight': float((y_train == 0).sum() / (y_train == 1).sum()),
    'tree_method': 'hist',
    'nthread': -1,
}

xgb_oof = np.zeros(len(X_train))
xgb_test = np.zeros(len(X_test))

for fold, (ti, vi) in enumerate(skf.split(X_train, y_train)):
    dtrain = xgb.DMatrix(X_train.iloc[ti], y_train.iloc[ti])
    dval = xgb.DMatrix(X_train.iloc[vi], y_train.iloc[vi])
    dtest = xgb.DMatrix(X_test)
    
    model = xgb.train(xgb_params, dtrain, 3000, evals=[(dval, 'val')],
                      early_stopping_rounds=100, verbose_eval=500)
    
    xgb_oof[vi] = model.predict(dval)
    xgb_test += model.predict(dtest) / N_FOLDS
    
    ks = calculate_ks_score(y_train.iloc[vi].values, xgb_oof[vi])
    print(f"  Fold {fold+1}: KS={ks:.4f}")

xgb_ks = calculate_ks_score(y_train.values, xgb_oof)
print(f"XGBoost OOF KS: {xgb_ks:.4f}")

# ============================================================
# 6. CatBoost
# ============================================================
print("\n" + "=" * 60)
print("6. Training CatBoost...")

cb_oof = np.zeros(len(X_train))
cb_test = np.zeros(len(X_test))

for fold, (ti, vi) in enumerate(skf.split(X_train, y_train)):
    model = CatBoostClassifier(
        iterations=3000, learning_rate=0.03, depth=7,
        l2_leaf_reg=3.0, min_data_in_leaf=50,
        subsample=0.8, colsample_bylevel=0.7,
        random_seed=SEED, auto_class_weights='Balanced',
        early_stopping_rounds=100, verbose=500, task_type='CPU',
    )
    model.fit(X_train.iloc[ti], y_train.iloc[ti],
              eval_set=(X_train.iloc[vi], y_train.iloc[vi]), use_best_model=True)
    
    cb_oof[vi] = model.predict_proba(X_train.iloc[vi])[:, 1]
    cb_test += model.predict_proba(X_test)[:, 1] / N_FOLDS
    
    ks = calculate_ks_score(y_train.iloc[vi].values, cb_oof[vi])
    print(f"  Fold {fold+1}: KS={ks:.4f}")

cb_ks = calculate_ks_score(y_train.values, cb_oof)
print(f"CatBoost OOF KS: {cb_ks:.4f}")

# ============================================================
# 7. ENSEMBLE
# ============================================================
print("\n" + "=" * 60)
print("7. Ensemble...")

# Optimal weight search
best_ks = 0
best_w = (1/3, 1/3, 1/3)

for w1 in np.arange(0.1, 0.8, 0.05):
    for w2 in np.arange(0.1, 0.8 - w1, 0.05):
        w3 = 1 - w1 - w2
        if w3 < 0.05:
            continue
        blend = w1 * lgb_oof + w2 * xgb_oof + w3 * cb_oof
        ks = calculate_ks_score(y_train.values, blend)
        if ks > best_ks:
            best_ks = ks
            best_w = (w1, w2, w3)

print(f"Prob blend: w=({best_w[0]:.2f},{best_w[1]:.2f},{best_w[2]:.2f}), KS={best_ks:.4f}")

# Rank blend
r_lgb = rankdata(lgb_oof) / len(lgb_oof)
r_xgb = rankdata(xgb_oof) / len(xgb_oof)
r_cb = rankdata(cb_oof) / len(cb_oof)

best_rks = 0
best_rw = (1/3, 1/3, 1/3)

for w1 in np.arange(0.1, 0.8, 0.05):
    for w2 in np.arange(0.1, 0.8 - w1, 0.05):
        w3 = 1 - w1 - w2
        if w3 < 0.05:
            continue
        blend = w1 * r_lgb + w2 * r_xgb + w3 * r_cb
        ks = calculate_ks_score(y_train.values, blend)
        if ks > best_rks:
            best_rks = ks
            best_rw = (w1, w2, w3)

print(f"Rank blend: w=({best_rw[0]:.2f},{best_rw[1]:.2f},{best_rw[2]:.2f}), KS={best_rks:.4f}")

if best_rks > best_ks:
    print("=> Using rank blend")
    final_test = best_rw[0] * rankdata(lgb_test)/len(lgb_test) + \
                 best_rw[1] * rankdata(xgb_test)/len(xgb_test) + \
                 best_rw[2] * rankdata(cb_test)/len(cb_test)
    final_ks = best_rks
else:
    print("=> Using prob blend")
    final_test = best_w[0] * lgb_test + best_w[1] * xgb_test + best_w[2] * cb_test
    final_ks = best_ks

# ============================================================
# 8. SUBMISSION
# ============================================================
print("\n" + "=" * 60)
print("8. Generating submission...")

test_ids = test['application_id'].values
sub = pd.DataFrame({'application_id': test_ids, 'default_probability': final_test})
sub['default_probability'] = sub['default_probability'].clip(0, 1)

assert sub['default_probability'].isna().sum() == 0
assert len(sub) == 30000

sub.to_csv(SUBMISSION_PATH, index=False)
print(f"Saved: {SUBMISSION_PATH}")
print(sub['default_probability'].describe())

print(f"\n{'='*60}")
print(f"LGB KS:      {lgb_ks:.4f}")
print(f"XGB KS:      {xgb_ks:.4f}")
print(f"CB KS:       {cb_ks:.4f}")
print(f"Ensemble KS: {final_ks:.4f}")
print(f"Target: >= 0.57  |  {'ACHIEVED!' if final_ks >= 0.57 else 'Need improvement'}")
print(f"{'='*60}")
