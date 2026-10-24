"""
DSC LOGIKA UI 2026 — Credit Default Prediction Pipeline
========================================================
Metric: KS Statistic  |  Target: >= 0.57
Models: LightGBM + XGBoost + CatBoost ensemble
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from scipy.stats import ks_2samp
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool

SEED = 42
np.random.seed(SEED)

DATA_DIR = "D:/LOGIKA UI/code/data-science-competition-logika-ui-2026"
SUBMISSION_PATH = "D:/LOGIKA UI/code/artifacts/DSC0102_JanjiLombaTerakhirTahunIni_Submisi.csv"

# ============================================================
# 1. LOAD DATA
# ============================================================
print("=" * 60)
print("1. Loading data...")
print("=" * 60)

train = pd.read_csv(f"{DATA_DIR}/train_applications.csv")
test = pd.read_csv(f"{DATA_DIR}/test_applications.csv")
customers = pd.read_csv(f"{DATA_DIR}/customer_profiles.csv")
prev_apps = pd.read_csv(f"{DATA_DIR}/previous_applications.csv")
repayment = pd.read_csv(f"{DATA_DIR}/repayment_history.csv")
acct_beh = pd.read_csv(f"{DATA_DIR}/account_behavior.csv")
bureau = pd.read_csv(f"{DATA_DIR}/bureau_observations.csv")
ecom = pd.read_csv(f"{DATA_DIR}/ecommerce_transactions.csv")

print(f"Train: {train.shape}, Test: {test.shape}")
print(f"Target distribution:\n{train['target'].value_counts(normalize=True)}")

# Parse datetime columns
for df in [train, test]:
    df['application_datetime'] = pd.to_datetime(df['application_datetime'])
    df['decision_datetime'] = pd.to_datetime(df['decision_datetime'])

prev_apps['application_datetime'] = pd.to_datetime(prev_apps['application_datetime'])
prev_apps['decision_datetime'] = pd.to_datetime(prev_apps['decision_datetime'])
repayment['due_date'] = pd.to_datetime(repayment['due_date'])
repayment['last_payment_datetime'] = pd.to_datetime(repayment['last_payment_datetime'])
acct_beh['month'] = pd.to_datetime(acct_beh['month'])
bureau['report_datetime'] = pd.to_datetime(bureau['report_datetime'])
bureau['available_datetime'] = pd.to_datetime(bureau['available_datetime'])
ecom['transaction_datetime'] = pd.to_datetime(ecom['transaction_datetime'])

# Combine train + test for feature engineering
train['is_train'] = 1
test['is_train'] = 0
test['target'] = -1
all_apps = pd.concat([train, test], ignore_index=True)

print(f"\nAll applications: {all_apps.shape}")

# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 60)
print("2. Feature Engineering...")
print("=" * 60)

# --- 2A. Application-level features ---
print("  2A. Application features...")
all_apps['loan_to_installment_ratio'] = all_apps['loan_amount'] / (all_apps['proposed_monthly_installment'] + 1)
all_apps['decision_time_hours'] = (all_apps['decision_datetime'] - all_apps['application_datetime']).dt.total_seconds() / 3600
all_apps['app_hour'] = all_apps['application_datetime'].dt.hour
all_apps['app_dayofweek'] = all_apps['application_datetime'].dt.dayofweek
all_apps['app_month'] = all_apps['application_datetime'].dt.month
all_apps['app_day'] = all_apps['application_datetime'].dt.day
all_apps['loan_purpose_enc'] = all_apps['loan_purpose'].map({
    'education': 0, 'home_improvement': 1, 'household': 2, 'working_capital': 3, 'medical': 4
})

# --- 2B. Customer profile features ---
print("  2B. Customer profile features...")
all_apps = all_apps.merge(customers, on='customer_id', how='left')

# Derived from customer profile
all_apps['income_to_installment'] = all_apps['declared_monthly_income'] / (all_apps['proposed_monthly_installment'] + 1)
all_apps['loan_to_income'] = all_apps['loan_amount'] / (all_apps['declared_monthly_income'] + 1)
all_apps['dti_ratio'] = all_apps['proposed_monthly_installment'] / (all_apps['declared_monthly_income'] + 1)
all_apps['income_per_dependent'] = all_apps['declared_monthly_income'] / (all_apps['dependents'] + 1)
all_apps['age_employment_ratio'] = all_apps['age'] / (all_apps['employment_tenure_months'] + 1)
all_apps['total_repayment'] = all_apps['proposed_monthly_installment'] * all_apps['term_months']
all_apps['total_interest'] = all_apps['total_repayment'] - all_apps['loan_amount']
all_apps['interest_to_loan'] = all_apps['total_interest'] / (all_apps['loan_amount'] + 1)

# Encode categoricals
all_apps['region_enc'] = all_apps['region'].map({
    'Region_A': 0, 'Region_B': 1, 'Region_C': 2, 'Region_D': 3, 'Region_E': 4
})
all_apps['employment_type_enc'] = all_apps['employment_type'].map({
    'salaried': 0, 'self_employed': 1, 'variable_income': 2
})
all_apps['housing_status_enc'] = all_apps['housing_status'].map({
    'owned': 0, 'rented': 1, 'family': 2
})

# --- 2C. Previous applications features (with time cutoff) ---
print("  2C. Previous applications features...")

# Merge decision_datetime from main app to prev_apps for cutoff
app_dates = all_apps[['customer_id', 'application_id', 'decision_datetime']].rename(
    columns={'application_id': 'main_app_id', 'decision_datetime': 'main_decision_dt'}
)

prev_merged = prev_apps.merge(app_dates, on='customer_id', how='inner')
# Filter: only previous applications before the current decision_datetime
prev_merged = prev_merged[prev_merged['decision_datetime'] <= prev_merged['main_decision_dt']]

# Aggregations per main application
prev_agg = prev_merged.groupby('main_app_id').agg(
    prev_app_count=('application_id', 'count'),
    prev_approved_count=('decision', lambda x: (x == 'approved').sum()),
    prev_rejected_count=('decision', lambda x: (x == 'rejected').sum()),
    prev_cancelled_count=('decision', lambda x: (x == 'cancelled').sum()),
    prev_requested_amount_mean=('requested_amount', 'mean'),
    prev_requested_amount_max=('requested_amount', 'max'),
    prev_requested_amount_sum=('requested_amount', 'sum'),
    prev_approved_amount_mean=('approved_amount', 'mean'),
    prev_approved_amount_max=('approved_amount', 'max'),
    prev_approved_amount_sum=('approved_amount', 'sum'),
    prev_term_mean=('term_months', 'mean'),
    prev_interest_rate_mean=('annual_interest_rate', 'mean'),
    prev_installment_mean=('scheduled_monthly_installment', 'mean'),
).reset_index().rename(columns={'main_app_id': 'application_id'})

prev_agg['prev_approval_rate'] = prev_agg['prev_approved_count'] / (prev_agg['prev_app_count'] + 1)
prev_agg['prev_rejection_rate'] = prev_agg['prev_rejected_count'] / (prev_agg['prev_app_count'] + 1)

# Days since last previous application
prev_last = prev_merged.groupby('main_app_id').agg(
    prev_last_decision_dt=('decision_datetime', 'max')
).reset_index().rename(columns={'main_app_id': 'application_id'})

all_apps = all_apps.merge(prev_agg, on='application_id', how='left')
all_apps = all_apps.merge(prev_last, on='application_id', how='left')
all_apps['days_since_last_prev_app'] = (all_apps['decision_datetime'] - all_apps['prev_last_decision_dt']).dt.days

# --- 2D. Repayment history features (with time cutoff via loan dates) ---
print("  2D. Repayment history features...")

# Get loan_ids from previous approved applications with cutoff
prev_approved = prev_merged[prev_merged['decision'] == 'approved'][['main_app_id', 'loan_id', 'main_decision_dt']].dropna(subset=['loan_id'])

# Merge repayment with approved prev apps
repay_merged = repayment.merge(prev_approved, on='loan_id', how='inner')
# Filter: only repayments before the main application's decision_datetime
repay_merged = repay_merged[repay_merged['due_date'] <= repay_merged['main_decision_dt']]

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
    repay_dpd_gt0_count=('days_past_due', lambda x: (x > 0).sum()),
    repay_dpd_gt30_count=('days_past_due', lambda x: (x > 30).sum()),
    repay_dpd_gt60_count=('days_past_due', lambda x: (x > 60).sum()),
    repay_dpd_gt90_count=('days_past_due', lambda x: (x > 90).sum()),
).reset_index().rename(columns={'main_app_id': 'application_id'})

repay_agg['repay_paid_ratio'] = repay_agg['repay_amount_paid_sum'] / (repay_agg['repay_amount_due_sum'] + 1)
repay_agg['repay_late_ratio'] = repay_agg['repay_dpd_gt0_count'] / (repay_agg['repay_count'] + 1)
repay_agg['repay_severe_late_ratio'] = repay_agg['repay_dpd_gt30_count'] / (repay_agg['repay_count'] + 1)

all_apps = all_apps.merge(repay_agg, on='application_id', how='left')

# --- 2E. Account behavior features (with time cutoff) ---
print("  2E. Account behavior features...")

acct_dates = all_apps[['customer_id', 'application_id', 'decision_datetime']].copy()
acct_merged = acct_beh.merge(acct_dates, on='customer_id', how='inner')
acct_merged = acct_merged[acct_merged['month'] <= acct_merged['decision_datetime']]

acct_agg = acct_merged.groupby('application_id').agg(
    acct_months_count=('month', 'nunique'),
    acct_opening_balance_mean=('opening_balance', 'mean'),
    acct_closing_balance_mean=('closing_balance', 'mean'),
    acct_closing_balance_min=('closing_balance', 'min'),
    acct_closing_balance_std=('closing_balance', 'std'),
    acct_inflows_mean=('inflows', 'mean'),
    acct_inflows_sum=('inflows', 'sum'),
    acct_inflows_std=('inflows', 'std'),
    acct_outflows_mean=('outflows', 'mean'),
    acct_outflows_sum=('outflows', 'sum'),
    acct_income_credits_mean=('income_like_credits', 'mean'),
    acct_income_credits_sum=('income_like_credits', 'sum'),
    acct_avg_balance_mean=('average_balance', 'mean'),
    acct_avg_balance_min=('average_balance', 'min'),
    acct_min_balance_mean=('minimum_balance', 'mean'),
    acct_min_balance_min=('minimum_balance', 'min'),
    acct_low_balance_days_mean=('low_balance_days', 'mean'),
    acct_low_balance_days_sum=('low_balance_days', 'sum'),
    acct_low_balance_days_max=('low_balance_days', 'max'),
).reset_index()

# Additional derived features
acct_agg['acct_net_flow'] = acct_agg['acct_inflows_sum'] - acct_agg['acct_outflows_sum']
acct_agg['acct_outflow_to_inflow'] = acct_agg['acct_outflows_mean'] / (acct_agg['acct_inflows_mean'] + 1)
acct_agg['acct_income_to_inflow'] = acct_agg['acct_income_credits_mean'] / (acct_agg['acct_inflows_mean'] + 1)

# Last 3 months features
acct_last3 = acct_merged.sort_values('month').groupby('application_id').tail(3)
acct_last3_agg = acct_last3.groupby('application_id').agg(
    acct_last3_closing_mean=('closing_balance', 'mean'),
    acct_last3_inflows_mean=('inflows', 'mean'),
    acct_last3_outflows_mean=('outflows', 'mean'),
    acct_last3_low_bal_days_mean=('low_balance_days', 'mean'),
    acct_last3_avg_balance_mean=('average_balance', 'mean'),
    acct_last3_min_balance_min=('minimum_balance', 'min'),
).reset_index()

all_apps = all_apps.merge(acct_agg, on='application_id', how='left')
all_apps = all_apps.merge(acct_last3_agg, on='application_id', how='left')

# --- 2F. Bureau features (with time cutoff) ---
print("  2F. Bureau features...")

bureau_dates = all_apps[['application_id', 'decision_datetime']].copy()
bureau_merged = bureau.merge(bureau_dates, on='application_id', how='inner', suffixes=('', '_main'))
# Filter: available_datetime <= decision_datetime
bureau_merged = bureau_merged[bureau_merged['available_datetime'] <= bureau_merged['decision_datetime']]

bureau_agg = bureau_merged.groupby('application_id').agg(
    bureau_count=('bureau_observation_id', 'count'),
    bureau_providers=('bureau_provider', 'nunique'),
    bureau_score_mean=('score', 'mean'),
    bureau_score_min=('score', 'min'),
    bureau_score_max=('score', 'max'),
    bureau_score_std=('score', 'std'),
    bureau_active_accounts_mean=('active_accounts', 'mean'),
    bureau_active_accounts_max=('active_accounts', 'max'),
    bureau_active_accounts_sum=('active_accounts', 'sum'),
    bureau_outstanding_mean=('outstanding_balance', 'mean'),
    bureau_outstanding_max=('outstanding_balance', 'max'),
    bureau_outstanding_sum=('outstanding_balance', 'sum'),
    bureau_revolving_limit_mean=('revolving_credit_limit', 'mean'),
    bureau_revolving_limit_sum=('revolving_credit_limit', 'sum'),
    bureau_revolving_outstanding_mean=('revolving_outstanding_balance', 'mean'),
    bureau_utilization_mean=('utilization', 'mean'),
    bureau_utilization_max=('utilization', 'max'),
    bureau_inquiries_mean=('inquiries_last_6m', 'mean'),
    bureau_inquiries_max=('inquiries_last_6m', 'max'),
    bureau_inquiries_sum=('inquiries_last_6m', 'sum'),
    bureau_delinquent_mean=('delinquent_accounts_last_12m', 'mean'),
    bureau_delinquent_max=('delinquent_accounts_last_12m', 'max'),
    bureau_delinquent_sum=('delinquent_accounts_last_12m', 'sum'),
).reset_index()

# Bureau-derived ratios
bureau_agg['bureau_debt_to_limit'] = bureau_agg['bureau_revolving_outstanding_mean'] / (bureau_agg['bureau_revolving_limit_mean'] + 1)
bureau_agg['bureau_has_delinquent'] = (bureau_agg['bureau_delinquent_sum'] > 0).astype(int)

all_apps = all_apps.merge(bureau_agg, on='application_id', how='left')

# --- 2G. E-commerce features (with time cutoff) ---
print("  2G. E-commerce features...")

ecom_dates = all_apps[['customer_id', 'application_id', 'decision_datetime']].copy()
ecom_merged = ecom.merge(ecom_dates, on='customer_id', how='inner')
ecom_merged = ecom_merged[ecom_merged['transaction_datetime'] <= ecom_merged['decision_datetime']]

# Only completed transactions for spending
ecom_completed = ecom_merged[ecom_merged['order_status'] == 'completed']
ecom_cancelled = ecom_merged[ecom_merged['order_status'] == 'cancelled']

ecom_agg = ecom_completed.groupby('application_id').agg(
    ecom_completed_count=('order_item_id', 'count'),
    ecom_total_spending=('item_total', 'sum'),
    ecom_avg_item_total=('item_total', 'mean'),
    ecom_max_item_total=('item_total', 'max'),
    ecom_categories=('product_category', 'nunique'),
    ecom_unique_orders=('order_id', 'nunique'),
    ecom_quantity_mean=('quantity', 'mean'),
    ecom_quantity_sum=('quantity', 'sum'),
).reset_index()

ecom_cancel_agg = ecom_cancelled.groupby('application_id').agg(
    ecom_cancelled_count=('order_item_id', 'count'),
    ecom_cancelled_value=('item_total', 'sum'),
).reset_index()

# Payment method counts
ecom_pay = ecom_completed.groupby(['application_id', 'payment_method']).size().unstack(fill_value=0)
ecom_pay.columns = [f'ecom_pay_{c}' for c in ecom_pay.columns]
ecom_pay = ecom_pay.reset_index()

# Product category spending
ecom_cat = ecom_completed.groupby(['application_id', 'product_category'])['item_total'].sum().unstack(fill_value=0)
ecom_cat.columns = [f'ecom_cat_{c}' for c in ecom_cat.columns]
ecom_cat = ecom_cat.reset_index()

all_apps = all_apps.merge(ecom_agg, on='application_id', how='left')
all_apps = all_apps.merge(ecom_cancel_agg, on='application_id', how='left')
all_apps = all_apps.merge(ecom_pay, on='application_id', how='left')
all_apps = all_apps.merge(ecom_cat, on='application_id', how='left')

# E-commerce derived
all_apps['ecom_cancel_rate'] = all_apps['ecom_cancelled_count'].fillna(0) / (all_apps['ecom_completed_count'].fillna(0) + all_apps['ecom_cancelled_count'].fillna(0) + 1)
all_apps['ecom_avg_order_value'] = all_apps['ecom_total_spending'] / (all_apps['ecom_unique_orders'] + 1)
all_apps['ecom_spending_to_income'] = all_apps['ecom_total_spending'] / (all_apps['declared_monthly_income'] + 1)

# --- 2H. Cross-table interaction features ---
print("  2H. Cross-table interaction features...")

# Bureau score vs loan amount
all_apps['bureau_score_x_loan'] = all_apps['bureau_score_mean'] * all_apps['loan_amount']
all_apps['bureau_score_x_dti'] = all_apps['bureau_score_mean'] * all_apps['dti_ratio']

# Account balance vs loan
all_apps['avg_balance_to_loan'] = all_apps['acct_avg_balance_mean'] / (all_apps['loan_amount'] + 1)
all_apps['income_credits_to_installment'] = all_apps['acct_income_credits_mean'] / (all_apps['proposed_monthly_installment'] + 1)

# Repayment history risk indicators
all_apps['has_prev_apps'] = (~all_apps['prev_app_count'].isna()).astype(int)
all_apps['has_repay_history'] = (~all_apps['repay_count'].isna()).astype(int)
all_apps['has_bureau'] = (~all_apps['bureau_count'].isna()).astype(int)
all_apps['has_ecom'] = (~all_apps['ecom_completed_count'].isna()).astype(int)

# ============================================================
# 3. PREPARE FOR MODELING
# ============================================================
print("\n" + "=" * 60)
print("3. Preparing for modeling...")
print("=" * 60)

# Drop non-feature columns
drop_cols = ['application_id', 'customer_id', 'application_datetime', 'decision_datetime',
             'is_train', 'target', 'loan_purpose', 'region', 'employment_type', 'housing_status',
             'prev_last_decision_dt']

feature_cols = [c for c in all_apps.columns if c not in drop_cols]
print(f"Number of features: {len(feature_cols)}")

X_all = all_apps[feature_cols].copy()
y_all = all_apps['target'].copy()
is_train = all_apps['is_train'] == 1

X_train = X_all[is_train].reset_index(drop=True)
y_train = y_all[is_train].reset_index(drop=True)
X_test = X_all[~is_train].reset_index(drop=True)

print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"Features with NaN: {X_train.isnull().any().sum()}")

# ============================================================
# 4. KS METRIC
# ============================================================
def calculate_ks_score(y_true, y_pred):
    preds_bad = y_pred[y_true == 1]
    preds_good = y_pred[y_true == 0]
    if len(preds_bad) == 0 or len(preds_good) == 0:
        return 0.0
    return float(ks_2samp(preds_bad, preds_good).statistic)

# ============================================================
# 5. MODEL TRAINING — LightGBM
# ============================================================
print("\n" + "=" * 60)
print("5. Training LightGBM...")
print("=" * 60)

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
    'min_split_gain': 0.01,
    'verbose': -1,
    'n_jobs': -1,
    'random_state': SEED,
    'is_unbalance': True,
}

N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

lgb_oof = np.zeros(len(X_train))
lgb_test_preds = np.zeros(len(X_test))
lgb_ks_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    dtrain = lgb.Dataset(X_tr, y_tr)
    dval = lgb.Dataset(X_val, y_val, reference=dtrain)
    
    model = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=3000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)],
    )
    
    lgb_oof[val_idx] = model.predict(X_val)
    lgb_test_preds += model.predict(X_test) / N_FOLDS
    
    ks = calculate_ks_score(y_val.values, lgb_oof[val_idx])
    lgb_ks_scores.append(ks)
    print(f"  Fold {fold+1}: KS = {ks:.4f}")

lgb_overall_ks = calculate_ks_score(y_train.values, lgb_oof)
print(f"\nLightGBM Overall OOF KS: {lgb_overall_ks:.4f}")
print(f"LightGBM Mean Fold KS: {np.mean(lgb_ks_scores):.4f}")

# ============================================================
# 6. MODEL TRAINING — XGBoost
# ============================================================
print("\n" + "=" * 60)
print("6. Training XGBoost...")
print("=" * 60)

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
    'n_jobs': -1,
    'random_state': SEED,
    'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum(),
    'tree_method': 'hist',
}

xgb_oof = np.zeros(len(X_train))
xgb_test_preds = np.zeros(len(X_test))
xgb_ks_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    dtrain = xgb.DMatrix(X_tr, y_tr, enable_categorical=False)
    dval = xgb.DMatrix(X_val, y_val, enable_categorical=False)
    dtest = xgb.DMatrix(X_test, enable_categorical=False)
    
    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=3000,
        evals=[(dval, 'val')],
        early_stopping_rounds=100,
        verbose_eval=500,
    )
    
    xgb_oof[val_idx] = model.predict(dval)
    xgb_test_preds += model.predict(dtest) / N_FOLDS
    
    ks = calculate_ks_score(y_val.values, xgb_oof[val_idx])
    xgb_ks_scores.append(ks)
    print(f"  Fold {fold+1}: KS = {ks:.4f}")

xgb_overall_ks = calculate_ks_score(y_train.values, xgb_oof)
print(f"\nXGBoost Overall OOF KS: {xgb_overall_ks:.4f}")
print(f"XGBoost Mean Fold KS: {np.mean(xgb_ks_scores):.4f}")

# ============================================================
# 7. MODEL TRAINING — CatBoost
# ============================================================
print("\n" + "=" * 60)
print("7. Training CatBoost...")
print("=" * 60)

cb_oof = np.zeros(len(X_train))
cb_test_preds = np.zeros(len(X_test))
cb_ks_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    model = CatBoostClassifier(
        iterations=3000,
        learning_rate=0.03,
        depth=7,
        l2_leaf_reg=3.0,
        min_data_in_leaf=50,
        subsample=0.8,
        colsample_bylevel=0.7,
        random_seed=SEED,
        auto_class_weights='Balanced',
        early_stopping_rounds=100,
        verbose=500,
        task_type='CPU',
    )
    
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)
    
    cb_oof[val_idx] = model.predict_proba(X_val)[:, 1]
    cb_test_preds += model.predict_proba(X_test)[:, 1] / N_FOLDS
    
    ks = calculate_ks_score(y_val.values, cb_oof[val_idx])
    cb_ks_scores.append(ks)
    print(f"  Fold {fold+1}: KS = {ks:.4f}")

cb_overall_ks = calculate_ks_score(y_train.values, cb_oof)
print(f"\nCatBoost Overall OOF KS: {cb_overall_ks:.4f}")
print(f"CatBoost Mean Fold KS: {np.mean(cb_ks_scores):.4f}")

# ============================================================
# 8. ENSEMBLE & OPTIMAL BLENDING
# ============================================================
print("\n" + "=" * 60)
print("8. Ensemble & Blending...")
print("=" * 60)

# Find optimal weights via grid search
best_ks = 0
best_weights = (1/3, 1/3, 1/3)

for w1 in np.arange(0.1, 0.8, 0.05):
    for w2 in np.arange(0.1, 0.8, 0.05):
        w3 = 1 - w1 - w2
        if w3 < 0.05:
            continue
        blend_oof = w1 * lgb_oof + w2 * xgb_oof + w3 * cb_oof
        ks = calculate_ks_score(y_train.values, blend_oof)
        if ks > best_ks:
            best_ks = ks
            best_weights = (w1, w2, w3)

print(f"Best blend weights: LGB={best_weights[0]:.2f}, XGB={best_weights[1]:.2f}, CB={best_weights[2]:.2f}")
print(f"Best blend OOF KS: {best_ks:.4f}")

# Also try rank-based blending
from scipy.stats import rankdata
rank_lgb = rankdata(lgb_oof) / len(lgb_oof)
rank_xgb = rankdata(xgb_oof) / len(xgb_oof)
rank_cb = rankdata(cb_oof) / len(cb_oof)

best_rank_ks = 0
best_rank_weights = (1/3, 1/3, 1/3)

for w1 in np.arange(0.1, 0.8, 0.05):
    for w2 in np.arange(0.1, 0.8, 0.05):
        w3 = 1 - w1 - w2
        if w3 < 0.05:
            continue
        blend_rank = w1 * rank_lgb + w2 * rank_xgb + w3 * rank_cb
        ks = calculate_ks_score(y_train.values, blend_rank)
        if ks > best_rank_ks:
            best_rank_ks = ks
            best_rank_weights = (w1, w2, w3)

print(f"\nBest rank-blend weights: LGB={best_rank_weights[0]:.2f}, XGB={best_rank_weights[1]:.2f}, CB={best_rank_weights[2]:.2f}")
print(f"Best rank-blend OOF KS: {best_rank_ks:.4f}")

# Choose the best method
if best_rank_ks > best_ks:
    print("\n=> Using RANK-BASED blending (better KS)")
    final_test = best_rank_weights[0] * rankdata(lgb_test_preds)/len(lgb_test_preds) + \
                 best_rank_weights[1] * rankdata(xgb_test_preds)/len(xgb_test_preds) + \
                 best_rank_weights[2] * rankdata(cb_test_preds)/len(cb_test_preds)
    final_ks = best_rank_ks
else:
    print("\n=> Using PROBABILITY blending (better KS)")
    final_test = best_weights[0] * lgb_test_preds + best_weights[1] * xgb_test_preds + best_weights[2] * cb_test_preds
    final_ks = best_ks

# Individual model KS summary
print(f"\n{'='*40}")
print(f"SUMMARY OF KS SCORES")
print(f"{'='*40}")
print(f"LightGBM OOF KS:  {lgb_overall_ks:.4f}")
print(f"XGBoost OOF KS:   {xgb_overall_ks:.4f}")
print(f"CatBoost OOF KS:  {cb_overall_ks:.4f}")
print(f"Ensemble OOF KS:  {final_ks:.4f}")
print(f"{'='*40}")

# ============================================================
# 9. GENERATE SUBMISSION
# ============================================================
print("\n" + "=" * 60)
print("9. Generating submission...")
print("=" * 60)

test_ids = test['application_id'].values
submission = pd.DataFrame({
    'application_id': test_ids,
    'default_probability': final_test
})

# Clip to [0, 1]
submission['default_probability'] = submission['default_probability'].clip(0, 1)

# Sanity checks
assert submission['default_probability'].isna().sum() == 0, "NaN found in predictions!"
assert len(submission) == 30000, f"Expected 30000 rows, got {len(submission)}"
assert (submission['default_probability'] >= 0).all() and (submission['default_probability'] <= 1).all(), "Values out of [0,1]!"

submission.to_csv(SUBMISSION_PATH, index=False)
print(f"\nSubmission saved to: {SUBMISSION_PATH}")
print(f"Shape: {submission.shape}")
print(f"Stats:\n{submission['default_probability'].describe()}")

print("\n" + "=" * 60)
print("PIPELINE COMPLETE!")
print(f"Final OOF KS Score: {final_ks:.4f}")
print(f"Target KS: >= 0.57")
print(f"{'ACHIEVED!' if final_ks >= 0.57 else 'Need improvement...'}")
print("=" * 60)
