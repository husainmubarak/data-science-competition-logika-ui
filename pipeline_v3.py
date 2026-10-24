"""
DSC LOGIKA UI 2026 — Credit Default Prediction Pipeline v3
==========================================================
Fixed column naming + optimized e-commerce processing
Metric: KS Statistic  |  Target: >= 0.57
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import gc
from collections import Counter

from scipy.stats import ks_2samp, rankdata
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

SEED = 42
np.random.seed(SEED)

DATA_DIR = "D:/LOGIKA UI/code/data-science-competition-logika-ui-2026"
SUBMISSION_PATH = "D:/LOGIKA UI/code/artifacts/DSC0102_JanjiLombaTerakhirTahunIni_Submisi.csv"

def ks_score(y_true, y_pred):
    return float(ks_2samp(y_pred[y_true == 1], y_pred[y_true == 0]).statistic)

# ============================================================
# 1. LOAD
# ============================================================
print("=" * 60)
print("1. Loading data...")

train = pd.read_csv(f"{DATA_DIR}/train_applications.csv")
test = pd.read_csv(f"{DATA_DIR}/test_applications.csv")
customers = pd.read_csv(f"{DATA_DIR}/customer_profiles.csv")

for df in [train, test]:
    df['application_datetime'] = pd.to_datetime(df['application_datetime'])
    df['decision_datetime'] = pd.to_datetime(df['decision_datetime'])

train['is_train'] = 1
test['is_train'] = 0
test['target'] = -1
apps = pd.concat([train, test], ignore_index=True)

print(f"Train: {train.shape}, Test: {test.shape}")
print(f"Target: {train['target'].value_counts().to_dict()}")

# ============================================================
# 2A. App + Customer features
# ============================================================
print("\n2A. Application + Customer features...")

apps['loan_to_installment'] = apps['loan_amount'] / (apps['proposed_monthly_installment'] + 1)
apps['decision_hours'] = (apps['decision_datetime'] - apps['application_datetime']).dt.total_seconds() / 3600
apps['app_hour'] = apps['application_datetime'].dt.hour
apps['app_dow'] = apps['application_datetime'].dt.dayofweek
apps['app_month'] = apps['application_datetime'].dt.month
apps['app_day'] = apps['application_datetime'].dt.day
apps['loan_purpose_enc'] = apps['loan_purpose'].astype('category').cat.codes

apps = apps.merge(customers, on='customer_id', how='left')

apps['income_to_inst'] = apps['declared_monthly_income'] / (apps['proposed_monthly_installment'] + 1)
apps['loan_to_income'] = apps['loan_amount'] / (apps['declared_monthly_income'] + 1)
apps['dti'] = apps['proposed_monthly_installment'] / (apps['declared_monthly_income'] + 1)
apps['income_per_dep'] = apps['declared_monthly_income'] / (apps['dependents'] + 1)
apps['age_emp_ratio'] = apps['age'] / (apps['employment_tenure_months'] + 1)
apps['total_repay'] = apps['proposed_monthly_installment'] * apps['term_months']
apps['total_interest'] = apps['total_repay'] - apps['loan_amount']
apps['interest_to_loan'] = apps['total_interest'] / (apps['loan_amount'] + 1)
apps['monthly_rate'] = apps['annual_interest_rate'] / 12
apps['emp_years'] = apps['employment_tenure_months'] / 12
apps['age_squared'] = apps['age'] ** 2

apps['region_enc'] = apps['region'].astype('category').cat.codes
apps['emp_type_enc'] = apps['employment_type'].astype('category').cat.codes
apps['housing_enc'] = apps['housing_status'].astype('category').cat.codes

# ============================================================
# 2B. Previous Applications
# ============================================================
print("\n2B. Previous applications features...")

prev = pd.read_csv(f"{DATA_DIR}/previous_applications.csv")
prev['decision_datetime'] = pd.to_datetime(prev['decision_datetime'])
prev = prev.rename(columns={
    'application_id': 'prev_app_id',
    'decision_datetime': 'prev_decision_dt',
    'application_datetime': 'prev_app_dt',
    'annual_interest_rate': 'prev_rate',
    'term_months': 'prev_term',
})

cutoffs = apps[['application_id', 'customer_id', 'decision_datetime']].copy()

prev_m = prev.merge(cutoffs, on='customer_id', how='inner')
prev_m = prev_m[prev_m['prev_decision_dt'] <= prev_m['decision_datetime']]
print(f"  prev_merged after cutoff: {len(prev_m)}")

prev_agg = prev_m.groupby('application_id').agg(
    prev_count=('prev_app_id', 'count'),
    prev_approved=('decision', lambda x: (x == 'approved').sum()),
    prev_rejected=('decision', lambda x: (x == 'rejected').sum()),
    prev_req_mean=('requested_amount', 'mean'),
    prev_req_max=('requested_amount', 'max'),
    prev_req_sum=('requested_amount', 'sum'),
    prev_appr_amt_mean=('approved_amount', 'mean'),
    prev_appr_amt_max=('approved_amount', 'max'),
    prev_appr_amt_sum=('approved_amount', 'sum'),
    prev_term_mean=('prev_term', 'mean'),
    prev_rate_mean=('prev_rate', 'mean'),
    prev_inst_mean=('scheduled_monthly_installment', 'mean'),
    prev_last_dt=('prev_decision_dt', 'max'),
).reset_index()

prev_agg['prev_appr_rate'] = prev_agg['prev_approved'] / (prev_agg['prev_count'] + 1)
prev_agg['prev_rej_rate'] = prev_agg['prev_rejected'] / (prev_agg['prev_count'] + 1)

apps = apps.merge(prev_agg, on='application_id', how='left')
apps['days_since_prev'] = (apps['decision_datetime'] - apps['prev_last_dt']).dt.days

# Get approved loan_ids for repayment
approved_loans = prev_m[prev_m['decision'] == 'approved'][['application_id', 'loan_id', 'decision_datetime']].dropna(subset=['loan_id']).copy()
approved_loans = approved_loans.rename(columns={'decision_datetime': 'main_dt'})

del prev, prev_m
gc.collect()

# ============================================================
# 2C. Repayment History
# ============================================================
print("\n2C. Repayment history features...")

repay = pd.read_csv(f"{DATA_DIR}/repayment_history.csv")
repay['due_date'] = pd.to_datetime(repay['due_date'])

repay_m = repay.merge(approved_loans, on='loan_id', how='inner')
repay_m = repay_m[repay_m['due_date'] <= repay_m['main_dt']]
print(f"  repay_merged after cutoff: {len(repay_m)}")

if len(repay_m) > 0:
    repay_agg = repay_m.groupby('application_id').agg(
        rp_count=('installment_id', 'count'),
        rp_dpd_mean=('days_past_due', 'mean'),
        rp_dpd_max=('days_past_due', 'max'),
        rp_dpd_std=('days_past_due', 'std'),
        rp_dpd_sum=('days_past_due', 'sum'),
        rp_max_dpd_mean=('max_days_past_due', 'mean'),
        rp_max_dpd_max=('max_days_past_due', 'max'),
        rp_due_mean=('amount_due', 'mean'),
        rp_due_sum=('amount_due', 'sum'),
        rp_paid_mean=('amount_paid', 'mean'),
        rp_paid_sum=('amount_paid', 'sum'),
        rp_late=('days_past_due', lambda x: (x > 0).sum()),
        rp_late30=('days_past_due', lambda x: (x > 30).sum()),
        rp_late60=('days_past_due', lambda x: (x > 60).sum()),
        rp_late90=('days_past_due', lambda x: (x > 90).sum()),
    ).reset_index()
    
    repay_agg['rp_paid_ratio'] = repay_agg['rp_paid_sum'] / (repay_agg['rp_due_sum'] + 1)
    repay_agg['rp_late_ratio'] = repay_agg['rp_late'] / (repay_agg['rp_count'] + 1)
    repay_agg['rp_severe_ratio'] = repay_agg['rp_late30'] / (repay_agg['rp_count'] + 1)
    repay_agg['rp_underpay'] = repay_agg['rp_due_sum'] - repay_agg['rp_paid_sum']
    
    apps = apps.merge(repay_agg, on='application_id', how='left')

del repay, repay_m, approved_loans
gc.collect()

# ============================================================
# 2D. Account Behavior
# ============================================================
print("\n2D. Account behavior features...")

acct = pd.read_csv(f"{DATA_DIR}/account_behavior.csv")
acct['month'] = pd.to_datetime(acct['month'])

acct_cutoffs = apps[['application_id', 'customer_id', 'decision_datetime']].copy()
acct_m = acct.merge(acct_cutoffs, on='customer_id', how='inner')
acct_m = acct_m[acct_m['month'] <= acct_m['decision_datetime']]
print(f"  acct_merged after cutoff: {len(acct_m)}")

acct_agg = acct_m.groupby('application_id').agg(
    ab_months=('month', 'nunique'),
    ab_open_mean=('opening_balance', 'mean'),
    ab_close_mean=('closing_balance', 'mean'),
    ab_close_min=('closing_balance', 'min'),
    ab_close_std=('closing_balance', 'std'),
    ab_inflow_mean=('inflows', 'mean'),
    ab_inflow_sum=('inflows', 'sum'),
    ab_inflow_std=('inflows', 'std'),
    ab_outflow_mean=('outflows', 'mean'),
    ab_outflow_sum=('outflows', 'sum'),
    ab_income_cr_mean=('income_like_credits', 'mean'),
    ab_income_cr_sum=('income_like_credits', 'sum'),
    ab_avgbal_mean=('average_balance', 'mean'),
    ab_avgbal_min=('average_balance', 'min'),
    ab_minbal_mean=('minimum_balance', 'mean'),
    ab_minbal_min=('minimum_balance', 'min'),
    ab_lowbal_mean=('low_balance_days', 'mean'),
    ab_lowbal_sum=('low_balance_days', 'sum'),
    ab_lowbal_max=('low_balance_days', 'max'),
).reset_index()

acct_agg['ab_netflow'] = acct_agg['ab_inflow_sum'] - acct_agg['ab_outflow_sum']
acct_agg['ab_out_to_in'] = acct_agg['ab_outflow_mean'] / (acct_agg['ab_inflow_mean'] + 1)
acct_agg['ab_income_to_in'] = acct_agg['ab_income_cr_mean'] / (acct_agg['ab_inflow_mean'] + 1)

# Last 3 months
acct_l3 = acct_m.sort_values('month').groupby('application_id').tail(3)
acct_l3_agg = acct_l3.groupby('application_id').agg(
    ab_l3_close=('closing_balance', 'mean'),
    ab_l3_inflow=('inflows', 'mean'),
    ab_l3_outflow=('outflows', 'mean'),
    ab_l3_lowbal=('low_balance_days', 'mean'),
    ab_l3_avgbal=('average_balance', 'mean'),
    ab_l3_minbal=('minimum_balance', 'min'),
).reset_index()

apps = apps.merge(acct_agg, on='application_id', how='left')
apps = apps.merge(acct_l3_agg, on='application_id', how='left')

del acct, acct_m, acct_l3
gc.collect()

# ============================================================
# 2E. Bureau
# ============================================================
print("\n2E. Bureau features...")

bureau = pd.read_csv(f"{DATA_DIR}/bureau_observations.csv")
bureau['available_datetime'] = pd.to_datetime(bureau['available_datetime'])

bur_cutoffs = apps[['application_id', 'decision_datetime']].copy()
bur_m = bureau.merge(bur_cutoffs, on='application_id', how='inner')
bur_m = bur_m[bur_m['available_datetime'] <= bur_m['decision_datetime']]
print(f"  bureau_merged after cutoff: {len(bur_m)}")

bur_agg = bur_m.groupby('application_id').agg(
    bur_n=('bureau_observation_id', 'count'),
    bur_providers=('bureau_provider', 'nunique'),
    bur_score_mean=('score', 'mean'),
    bur_score_min=('score', 'min'),
    bur_score_max=('score', 'max'),
    bur_score_std=('score', 'std'),
    bur_actv_mean=('active_accounts', 'mean'),
    bur_actv_max=('active_accounts', 'max'),
    bur_outst_mean=('outstanding_balance', 'mean'),
    bur_outst_max=('outstanding_balance', 'max'),
    bur_outst_sum=('outstanding_balance', 'sum'),
    bur_revlim_mean=('revolving_credit_limit', 'mean'),
    bur_revlim_sum=('revolving_credit_limit', 'sum'),
    bur_revout_mean=('revolving_outstanding_balance', 'mean'),
    bur_util_mean=('utilization', 'mean'),
    bur_util_max=('utilization', 'max'),
    bur_inq_mean=('inquiries_last_6m', 'mean'),
    bur_inq_max=('inquiries_last_6m', 'max'),
    bur_inq_sum=('inquiries_last_6m', 'sum'),
    bur_delinq_mean=('delinquent_accounts_last_12m', 'mean'),
    bur_delinq_max=('delinquent_accounts_last_12m', 'max'),
    bur_delinq_sum=('delinquent_accounts_last_12m', 'sum'),
).reset_index()

bur_agg['bur_debt_lim'] = bur_agg['bur_revout_mean'] / (bur_agg['bur_revlim_mean'] + 1)
bur_agg['bur_has_delinq'] = (bur_agg['bur_delinq_sum'] > 0).astype(int)
bur_agg['bur_score_range'] = bur_agg['bur_score_max'] - bur_agg['bur_score_min']

apps = apps.merge(bur_agg, on='application_id', how='left')

del bureau, bur_m
gc.collect()

# ============================================================
# 2F. E-commerce — optimized: pre-agg by customer, no time cutoff 
#     (simplified for speed — most txns are before decision anyway)
# ============================================================
print("\n2F. E-commerce features...")

# Actually let's use a smarter approach: pre-aggregate per customer_id,
# and then for cutoff, compute per-app using chunks with batch groupby

ecom = pd.read_csv(f"{DATA_DIR}/ecommerce_transactions.csv")
ecom['transaction_datetime'] = pd.to_datetime(ecom['transaction_datetime'])
print(f"  ecom raw: {len(ecom)}")

ecom_cutoffs = apps[['application_id', 'customer_id', 'decision_datetime']].copy()

# Merge - this can be large but customer_id merge should be manageable
ecom_m = ecom.merge(ecom_cutoffs, on='customer_id', how='inner')
ecom_m = ecom_m[ecom_m['transaction_datetime'] <= ecom_m['decision_datetime']]
print(f"  ecom after cutoff: {len(ecom_m)}")

completed = ecom_m[ecom_m['order_status'] == 'completed']
cancelled = ecom_m[ecom_m['order_status'] == 'cancelled']

ecom_agg = completed.groupby('application_id').agg(
    ec_count=('order_item_id', 'count'),
    ec_spend=('item_total', 'sum'),
    ec_avg_item=('item_total', 'mean'),
    ec_max_item=('item_total', 'max'),
    ec_cats=('product_category', 'nunique'),
    ec_orders=('order_id', 'nunique'),
    ec_qty_mean=('quantity', 'mean'),
    ec_qty_sum=('quantity', 'sum'),
).reset_index()

ec_cancel = cancelled.groupby('application_id').agg(
    ec_cancel_n=('order_item_id', 'count'),
    ec_cancel_val=('item_total', 'sum'),
).reset_index()

# Payment method pivot
ec_pay = completed.groupby(['application_id', 'payment_method']).size().unstack(fill_value=0)
ec_pay.columns = [f'ec_pay_{c}' for c in ec_pay.columns]
ec_pay = ec_pay.reset_index()

# Category spending pivot
ec_cat = completed.groupby(['application_id', 'product_category'])['item_total'].sum().unstack(fill_value=0)
ec_cat.columns = [f'ec_cat_{c}' for c in ec_cat.columns]
ec_cat = ec_cat.reset_index()

apps = apps.merge(ecom_agg, on='application_id', how='left')
apps = apps.merge(ec_cancel, on='application_id', how='left')
apps = apps.merge(ec_pay, on='application_id', how='left')
apps = apps.merge(ec_cat, on='application_id', how='left')

apps['ec_cancel_rate'] = apps['ec_cancel_n'].fillna(0) / (apps['ec_count'].fillna(0) + apps['ec_cancel_n'].fillna(0) + 1)
apps['ec_avg_order'] = apps['ec_spend'] / (apps['ec_orders'].fillna(0) + 1)
apps['ec_spend_to_income'] = apps['ec_spend'].fillna(0) / (apps['declared_monthly_income'] + 1)

del ecom, ecom_m, completed, cancelled
gc.collect()

# ============================================================
# 2G. Interaction features
# ============================================================
print("\n2G. Interaction features...")

apps['bur_score_x_loan'] = apps['bur_score_mean'].fillna(0) * apps['loan_amount']
apps['bur_score_x_dti'] = apps['bur_score_mean'].fillna(0) * apps['dti']
apps['avgbal_to_loan'] = apps['ab_avgbal_mean'].fillna(0) / (apps['loan_amount'] + 1)
apps['income_cr_to_inst'] = apps['ab_income_cr_mean'].fillna(0) / (apps['proposed_monthly_installment'] + 1)
apps['bur_outst_to_income'] = apps['bur_outst_mean'].fillna(0) / (apps['declared_monthly_income'] + 1)
apps['installment_to_avgbal'] = apps['proposed_monthly_installment'] / (apps['ab_avgbal_mean'].fillna(0) + 1)

apps['has_prev'] = apps['prev_count'].notna().astype(int)
apps['has_repay'] = apps['rp_count'].notna().astype(int) if 'rp_count' in apps.columns else 0
apps['has_bureau'] = apps['bur_n'].notna().astype(int)
apps['has_ecom'] = apps['ec_count'].notna().astype(int)

# ============================================================
# 3. PREPARE
# ============================================================
print("\n" + "=" * 60)
print("3. Preparing features...")

drop_cols = ['application_id', 'customer_id', 'application_datetime', 'decision_datetime',
             'is_train', 'target', 'loan_purpose', 'region', 'employment_type', 'housing_status',
             'prev_last_dt']

feature_cols = [c for c in apps.columns if c not in drop_cols]
print(f"Total features: {len(feature_cols)}")

X = apps[feature_cols].copy()
y = apps['target'].copy()
mask = apps['is_train'] == 1

X_tr = X[mask].reset_index(drop=True)
y_tr = y[mask].reset_index(drop=True).astype(int)
X_te = X[~mask].reset_index(drop=True)

print(f"X_train: {X_tr.shape}, X_test: {X_te.shape}")

# ============================================================
# 4. LightGBM
# ============================================================
print("\n" + "=" * 60)
print("4. LightGBM (5-fold)...")

lgb_p = {
    'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
    'learning_rate': 0.02, 'num_leaves': 63, 'max_depth': 8,
    'min_child_samples': 50, 'subsample': 0.8, 'colsample_bytree': 0.7,
    'reg_alpha': 0.1, 'reg_lambda': 1.0, 'verbose': -1,
    'n_jobs': -1, 'seed': SEED, 'is_unbalance': True,
}

N = 5
skf = StratifiedKFold(n_splits=N, shuffle=True, random_state=SEED)

lgb_oof = np.zeros(len(X_tr))
lgb_te = np.zeros(len(X_te))

for f, (ti, vi) in enumerate(skf.split(X_tr, y_tr)):
    d_tr = lgb.Dataset(X_tr.iloc[ti], y_tr.iloc[ti])
    d_va = lgb.Dataset(X_tr.iloc[vi], y_tr.iloc[vi], reference=d_tr)
    m = lgb.train(lgb_p, d_tr, 3000, valid_sets=[d_va],
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)])
    lgb_oof[vi] = m.predict(X_tr.iloc[vi])
    lgb_te += m.predict(X_te) / N
    print(f"  Fold {f+1}: KS={ks_score(y_tr.iloc[vi].values, lgb_oof[vi]):.4f}")

lgb_ks = ks_score(y_tr.values, lgb_oof)
print(f"LGB OOF KS: {lgb_ks:.4f}")

# Feature importance
fi = pd.DataFrame({'f': m.feature_name(), 'i': m.feature_importance('gain')}).sort_values('i', ascending=False)
print("\nTop 20:")
print(fi.head(20).to_string(index=False))

# ============================================================
# 5. XGBoost
# ============================================================
print("\n" + "=" * 60)
print("5. XGBoost (5-fold)...")

xgb_p = {
    'objective': 'binary:logistic', 'eval_metric': 'auc',
    'learning_rate': 0.02, 'max_depth': 7, 'min_child_weight': 50,
    'subsample': 0.8, 'colsample_bytree': 0.7,
    'reg_alpha': 0.1, 'reg_lambda': 1.0, 'gamma': 0.1,
    'seed': SEED, 'scale_pos_weight': float((y_tr==0).sum()/(y_tr==1).sum()),
    'tree_method': 'hist', 'nthread': -1,
}

xgb_oof = np.zeros(len(X_tr))
xgb_te = np.zeros(len(X_te))

for f, (ti, vi) in enumerate(skf.split(X_tr, y_tr)):
    dt = xgb.DMatrix(X_tr.iloc[ti], y_tr.iloc[ti])
    dv = xgb.DMatrix(X_tr.iloc[vi], y_tr.iloc[vi])
    dte = xgb.DMatrix(X_te)
    m = xgb.train(xgb_p, dt, 3000, evals=[(dv,'v')],
                  early_stopping_rounds=100, verbose_eval=500)
    xgb_oof[vi] = m.predict(dv)
    xgb_te += m.predict(dte) / N
    print(f"  Fold {f+1}: KS={ks_score(y_tr.iloc[vi].values, xgb_oof[vi]):.4f}")

xgb_ks = ks_score(y_tr.values, xgb_oof)
print(f"XGB OOF KS: {xgb_ks:.4f}")

# ============================================================
# 6. CatBoost
# ============================================================
print("\n" + "=" * 60)
print("6. CatBoost (5-fold)...")

cb_oof = np.zeros(len(X_tr))
cb_te = np.zeros(len(X_te))

for f, (ti, vi) in enumerate(skf.split(X_tr, y_tr)):
    m = CatBoostClassifier(
        iterations=3000, learning_rate=0.03, depth=7,
        l2_leaf_reg=3.0, min_data_in_leaf=50,
        subsample=0.8, colsample_bylevel=0.7,
        random_seed=SEED, auto_class_weights='Balanced',
        early_stopping_rounds=100, verbose=500, task_type='CPU',
    )
    m.fit(X_tr.iloc[ti], y_tr.iloc[ti],
          eval_set=(X_tr.iloc[vi], y_tr.iloc[vi]), use_best_model=True)
    cb_oof[vi] = m.predict_proba(X_tr.iloc[vi])[:, 1]
    cb_te += m.predict_proba(X_te)[:, 1] / N
    print(f"  Fold {f+1}: KS={ks_score(y_tr.iloc[vi].values, cb_oof[vi]):.4f}")

cb_ks = ks_score(y_tr.values, cb_oof)
print(f"CB OOF KS: {cb_ks:.4f}")

# ============================================================
# 7. ENSEMBLE
# ============================================================
print("\n" + "=" * 60)
print("7. Ensemble optimization...")

# Prob blend
best_ks, best_w = 0, (1/3, 1/3, 1/3)
for w1 in np.arange(0.1, 0.8, 0.05):
    for w2 in np.arange(0.05, 0.85-w1, 0.05):
        w3 = 1 - w1 - w2
        if w3 < 0.05: continue
        bl = w1*lgb_oof + w2*xgb_oof + w3*cb_oof
        k = ks_score(y_tr.values, bl)
        if k > best_ks:
            best_ks, best_w = k, (w1,w2,w3)

print(f"Prob: w=({best_w[0]:.2f},{best_w[1]:.2f},{best_w[2]:.2f}), KS={best_ks:.4f}")

# Rank blend
r1 = rankdata(lgb_oof)/len(lgb_oof)
r2 = rankdata(xgb_oof)/len(xgb_oof)
r3 = rankdata(cb_oof)/len(cb_oof)

brk, brw = 0, (1/3,1/3,1/3)
for w1 in np.arange(0.1, 0.8, 0.05):
    for w2 in np.arange(0.05, 0.85-w1, 0.05):
        w3 = 1-w1-w2
        if w3 < 0.05: continue
        bl = w1*r1 + w2*r2 + w3*r3
        k = ks_score(y_tr.values, bl)
        if k > brk:
            brk, brw = k, (w1,w2,w3)

print(f"Rank: w=({brw[0]:.2f},{brw[1]:.2f},{brw[2]:.2f}), KS={brk:.4f}")

if brk > best_ks:
    print("=> Rank blend wins")
    final = brw[0]*rankdata(lgb_te)/len(lgb_te) + brw[1]*rankdata(xgb_te)/len(xgb_te) + brw[2]*rankdata(cb_te)/len(cb_te)
    final_ks = brk
else:
    print("=> Prob blend wins")
    final = best_w[0]*lgb_te + best_w[1]*xgb_te + best_w[2]*cb_te
    final_ks = best_ks

# ============================================================
# 8. SUBMISSION
# ============================================================
print("\n" + "=" * 60)
print("8. Submission...")

sub = pd.DataFrame({
    'application_id': test['application_id'].values,
    'default_probability': final
})
sub['default_probability'] = sub['default_probability'].clip(0, 1)

assert sub['default_probability'].isna().sum() == 0
assert len(sub) == 30000

sub.to_csv(SUBMISSION_PATH, index=False)

print(f"Saved: {SUBMISSION_PATH}")
print(sub['default_probability'].describe())
print(f"\n{'='*60}")
print(f"LGB: {lgb_ks:.4f} | XGB: {xgb_ks:.4f} | CB: {cb_ks:.4f}")
print(f"Ensemble: {final_ks:.4f}")
print(f"Target>=0.57: {'YES!' if final_ks >= 0.57 else 'No'}")
print(f"{'='*60}")
