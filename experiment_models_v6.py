"""
experiment_models_v6.py
Full modeling suite on features_v6.joblib
Includes:
- 5-Fold Stratified Cross-Validation
- LightGBM GBDT (Depth 7)
- LightGBM Extra Trees (Depth 8)
- XGBoost Hist (Depth 6)
- CatBoost (Depth 7)
- Segmented Model (Existing vs New)
- Stacking Meta-Learner with Skip Connections
- Powell Direct KS Optimization on Ensemble Blending
- Saves submission to artifacts/DSC0102_JanjiLombaTerakhirTahunIni_Submisi.csv
- Saves ensemble bundle to artifacts/selected_ensemble.joblib
"""

import gc
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from scipy.stats import ks_2samp, rankdata
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')

SEED = 42
N_FOLDS = 5
OUTPUT_DIR = Path('D:/LOGIKA UI/code/artifacts')

def competition_ks(y_true, y_pred):
    """Official Competition KS Statistic"""
    preds_bad = y_pred[y_true == 1]
    preds_good = y_pred[y_true == 0]
    if len(preds_bad) == 0 or len(preds_good) == 0:
        return 0.0
    return float(ks_2samp(preds_bad, preds_good).statistic)

print("=" * 70)
print("1. Loading Features v6...")
print("=" * 70)

bundle = joblib.load(OUTPUT_DIR / 'features_v6.joblib')
X = bundle['X_train']
y = bundle['y_train']
X_test = bundle['X_test']
test_ids = bundle['test_application_ids']
feature_names = bundle['feature_names']

print(f"Train: {X.shape}, Test: {X_test.shape}")
print(f"Target positive rate: {y.mean():.4f}")
print(f"Total features: {len(feature_names)}")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
folds = list(skf.split(X, y))

oof_predictions = {}
test_predictions = {}
experiment_results = []
trained_models = {}

# ============================================================
# EXPERIMENT 1: LightGBM Depth 7 (GBDT)
# ============================================================
print("\n" + "=" * 70)
print("EXPERIMENT 1: LightGBM Depth 7 (leaves=63)")
print("=" * 70)

lgb1_oof = np.zeros(len(y))
lgb1_test = np.zeros(len(X_test))
lgb1_models = []
lgb1_fold_ks = []

lgb1_params = {
    'objective': 'binary',
    'n_estimators': 3000,
    'learning_rate': 0.025,
    'max_depth': 7,
    'num_leaves': 63,
    'min_child_samples': 30,
    'subsample': 0.85,
    'subsample_freq': 1,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.5,
    'reg_lambda': 5.0,
    'random_state': SEED,
    'n_jobs': -1,
    'verbosity': -1,
}

for fold, (tr_idx, val_idx) in enumerate(folds, 1):
    X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
    X_va, y_va = X.iloc[val_idx], y[val_idx]
    
    p = lgb1_params.copy()
    p['random_state'] = SEED + fold
    m = LGBMClassifier(**p)
    m.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric='auc',
        callbacks=[early_stopping(100, verbose=False), log_evaluation(0)]
    )
    
    pred_val = m.predict_proba(X_va)[:, 1]
    lgb1_oof[val_idx] = pred_val
    lgb1_test += m.predict_proba(X_test)[:, 1] / N_FOLDS
    
    ks = competition_ks(y_va, pred_val)
    lgb1_fold_ks.append(ks)
    lgb1_models.append(m)
    print(f"  Fold {fold}: KS = {ks:.5f} (best iter: {m.best_iteration_})")

oof_ks = competition_ks(y, lgb1_oof)
oof_predictions['lgb_depth7'] = lgb1_oof
test_predictions['lgb_depth7'] = lgb1_test
trained_models['lgb_depth7'] = lgb1_models
experiment_results.append({'model': 'lgb_depth7', 'mean_fold_ks': np.mean(lgb1_fold_ks), 'oof_ks': oof_ks})
print(f"==> LGB Depth 7 Mean Fold KS: {np.mean(lgb1_fold_ks):.5f} | Pooled OOF KS: {oof_ks:.5f}")

fi = pd.Series(lgb1_models[0].feature_importances_, index=feature_names).sort_values(ascending=False)
print("\nTop 15 Features (LGB Depth 7):")
print(fi.head(15).to_string())

# ============================================================
# EXPERIMENT 2: LightGBM Depth 8 (Extra Trees)
# ============================================================
print("\n" + "=" * 70)
print("EXPERIMENT 2: LightGBM Depth 8 (Extra Trees, leaves=85)")
print("=" * 70)

lgb2_oof = np.zeros(len(y))
lgb2_test = np.zeros(len(X_test))
lgb2_models = []
lgb2_fold_ks = []

lgb2_params = {
    'objective': 'binary',
    'n_estimators': 3000,
    'learning_rate': 0.02,
    'max_depth': 8,
    'num_leaves': 85,
    'min_child_samples': 45,
    'subsample': 0.8,
    'subsample_freq': 1,
    'colsample_bytree': 0.6,
    'reg_alpha': 1.0,
    'reg_lambda': 10.0,
    'extra_trees': True,
    'random_state': SEED,
    'n_jobs': -1,
    'verbosity': -1,
}

for fold, (tr_idx, val_idx) in enumerate(folds, 1):
    X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
    X_va, y_va = X.iloc[val_idx], y[val_idx]
    
    p = lgb2_params.copy()
    p['random_state'] = SEED + fold * 10
    m = LGBMClassifier(**p)
    m.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric='auc',
        callbacks=[early_stopping(100, verbose=False), log_evaluation(0)]
    )
    
    pred_val = m.predict_proba(X_va)[:, 1]
    lgb2_oof[val_idx] = pred_val
    lgb2_test += m.predict_proba(X_test)[:, 1] / N_FOLDS
    
    ks = competition_ks(y_va, pred_val)
    lgb2_fold_ks.append(ks)
    lgb2_models.append(m)
    print(f"  Fold {fold}: KS = {ks:.5f} (best iter: {m.best_iteration_})")

oof_ks = competition_ks(y, lgb2_oof)
oof_predictions['lgb_extra'] = lgb2_oof
test_predictions['lgb_extra'] = lgb2_test
trained_models['lgb_extra'] = lgb2_models
experiment_results.append({'model': 'lgb_extra', 'mean_fold_ks': np.mean(lgb2_fold_ks), 'oof_ks': oof_ks})
print(f"==> LGB Extra Trees Mean Fold KS: {np.mean(lgb2_fold_ks):.5f} | Pooled OOF KS: {oof_ks:.5f}")

# ============================================================
# EXPERIMENT 3: XGBoost Depth 6 (Hist)
# ============================================================
print("\n" + "=" * 70)
print("EXPERIMENT 3: XGBoost Depth 6 (Hist)")
print("=" * 70)

xgb_oof = np.zeros(len(y))
xgb_test = np.zeros(len(X_test))
xgb_models = []
xgb_fold_ks = []

xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'n_estimators': 3000,
    'learning_rate': 0.025,
    'max_depth': 6,
    'min_child_weight': 25,
    'subsample': 0.85,
    'colsample_bytree': 0.65,
    'reg_alpha': 0.8,
    'reg_lambda': 8.0,
    'early_stopping_rounds': 100,
    'tree_method': 'hist',
    'n_jobs': -1,
    'random_state': SEED,
}

for fold, (tr_idx, val_idx) in enumerate(folds, 1):
    X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
    X_va, y_va = X.iloc[val_idx], y[val_idx]
    
    p = xgb_params.copy()
    p['random_state'] = SEED + fold
    m = XGBClassifier(**p)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    
    pred_val = m.predict_proba(X_va)[:, 1]
    xgb_oof[val_idx] = pred_val
    xgb_test += m.predict_proba(X_test)[:, 1] / N_FOLDS
    
    ks = competition_ks(y_va, pred_val)
    xgb_fold_ks.append(ks)
    xgb_models.append(m)
    print(f"  Fold {fold}: KS = {ks:.5f} (best iter: {m.best_iteration})")

oof_ks = competition_ks(y, xgb_oof)
oof_predictions['xgb_depth6'] = xgb_oof
test_predictions['xgb_depth6'] = xgb_test
trained_models['xgb_depth6'] = xgb_models
experiment_results.append({'model': 'xgb_depth6', 'mean_fold_ks': np.mean(xgb_fold_ks), 'oof_ks': oof_ks})
print(f"==> XGB Depth 6 Mean Fold KS: {np.mean(xgb_fold_ks):.5f} | Pooled OOF KS: {oof_ks:.5f}")

# ============================================================
# EXPERIMENT 4: CatBoost Depth 7
# ============================================================
print("\n" + "=" * 70)
print("EXPERIMENT 4: CatBoost Depth 7")
print("=" * 70)

cb_oof = np.zeros(len(y))
cb_test = np.zeros(len(X_test))
cb_models = []
cb_fold_ks = []

cb_params = {
    'iterations': 3000,
    'learning_rate': 0.03,
    'depth': 7,
    'l2_leaf_reg': 6.0,
    'min_data_in_leaf': 35,
    'subsample': 0.85,
    'random_seed': SEED,
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'early_stopping_rounds': 100,
    'thread_count': -1,
    'verbose': False,
    'allow_writing_files': False,
}

for fold, (tr_idx, val_idx) in enumerate(folds, 1):
    X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
    X_va, y_va = X.iloc[val_idx], y[val_idx]
    
    p = cb_params.copy()
    p['random_seed'] = SEED + fold
    m = CatBoostClassifier(**p)
    m.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
    
    pred_val = m.predict_proba(X_va)[:, 1]
    cb_oof[val_idx] = pred_val
    cb_test += m.predict_proba(X_test)[:, 1] / N_FOLDS
    
    ks = competition_ks(y_va, pred_val)
    cb_fold_ks.append(ks)
    cb_models.append(m)
    print(f"  Fold {fold}: KS = {ks:.5f} (best iter: {m.get_best_iteration()})")

oof_ks = competition_ks(y, cb_oof)
oof_predictions['cb_depth7'] = cb_oof
test_predictions['cb_depth7'] = cb_test
trained_models['cb_depth7'] = cb_models
experiment_results.append({'model': 'cb_depth7', 'mean_fold_ks': np.mean(cb_fold_ks), 'oof_ks': oof_ks})
print(f"==> CatBoost Depth 7 Mean Fold KS: {np.mean(cb_fold_ks):.5f} | Pooled OOF KS: {oof_ks:.5f}")

# ============================================================
# EXPERIMENT 5: Segmented Modeling (Existing vs New Customers)
# ============================================================
print("\n" + "=" * 70)
print("EXPERIMENT 5: Segmented Modeling (Existing vs New Customers)")
print("=" * 70)

is_existing = (X['has_repayment_history'] == 1).values
is_existing_test = (X_test['has_repayment_history'] == 1).values

seg_oof = np.zeros(len(y))
seg_test = np.zeros(len(X_test))

# Segment 1: Existing Customers
print(f"  Existing Customers: {is_existing.sum()} train, {is_existing_test.sum()} test")
for fold, (tr_idx, val_idx) in enumerate(folds, 1):
    tr_mask = is_existing[tr_idx]
    va_mask = is_existing[val_idx]
    
    tr_sub_idx = tr_idx[tr_mask]
    va_sub_idx = val_idx[va_mask]
    
    p_exist = lgb1_params.copy()
    p_exist['random_state'] = SEED + fold * 5
    m_exist = LGBMClassifier(**p_exist)
    m_exist.fit(
        X.iloc[tr_sub_idx], y[tr_sub_idx],
        eval_set=[(X.iloc[va_sub_idx], y[va_sub_idx])],
        eval_metric='auc',
        callbacks=[early_stopping(100, verbose=False), log_evaluation(0)]
    )
    seg_oof[va_sub_idx] = m_exist.predict_proba(X.iloc[va_sub_idx])[:, 1]
    if is_existing_test.sum() > 0:
        seg_test[is_existing_test] += m_exist.predict_proba(X_test.iloc[is_existing_test])[:, 1] / N_FOLDS

# Segment 2: New Customers
print(f"  New Customers: {(~is_existing).sum()} train, {(~is_existing_test).sum()} test")
repay_acct_cols = [c for c in X.columns if c.startswith('rep_') or c.startswith('acct_') or c.startswith('prev_')]
X_new = X.drop(columns=repay_acct_cols)
X_test_new = X_test.drop(columns=repay_acct_cols)

for fold, (tr_idx, val_idx) in enumerate(folds, 1):
    tr_mask = (~is_existing)[tr_idx]
    va_mask = (~is_existing)[val_idx]
    
    tr_sub_idx = tr_idx[tr_mask]
    va_sub_idx = val_idx[va_mask]
    
    m_new = LGBMClassifier(
        objective='binary', n_estimators=2000, learning_rate=0.03, max_depth=6, num_leaves=35,
        min_child_samples=25, subsample=0.85, colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=5.0,
        random_state=SEED + fold * 7, n_jobs=-1, verbosity=-1
    )
    m_new.fit(
        X_new.iloc[tr_sub_idx], y[tr_sub_idx],
        eval_set=[(X_new.iloc[va_sub_idx], y[va_sub_idx])],
        eval_metric='auc',
        callbacks=[early_stopping(100, verbose=False), log_evaluation(0)]
    )
    seg_oof[va_sub_idx] = m_new.predict_proba(X_new.iloc[va_sub_idx])[:, 1]
    if (~is_existing_test).sum() > 0:
        seg_test[~is_existing_test] += m_new.predict_proba(X_test_new.iloc[~is_existing_test])[:, 1] / N_FOLDS

seg_ks = competition_ks(y, seg_oof)
oof_predictions['segmented_lgb'] = seg_oof
test_predictions['segmented_lgb'] = seg_test
experiment_results.append({'model': 'segmented_lgb', 'mean_fold_ks': seg_ks, 'oof_ks': seg_ks})
print(f"==> Segmented Model OOF KS: {seg_ks:.5f}")

# ============================================================
# EXPERIMENT 6: Stacking Meta-Learner with Skip Connections
# ============================================================
print("\n" + "=" * 70)
print("EXPERIMENT 6: Stacking Meta-Learner with Skip Connections")
print("=" * 70)

base_model_names = ['lgb_depth7', 'lgb_extra', 'xgb_depth6', 'cb_depth7', 'segmented_lgb']
# Meta features: base models + raw anchor features
anchor_cols = ['super_risk_composite', 'composite_risk_score', 'rep_max_dpd_mean', 'bureau_curr_score_mean', 'annual_interest_rate']
X_meta_train = pd.DataFrame({m: oof_predictions[m] for m in base_model_names})
X_meta_test = pd.DataFrame({m: test_predictions[m] for m in base_model_names})

for ac in anchor_cols:
    if ac in X.columns:
        col_m = X[ac].fillna(X[ac].median()).values
        col_t = X_test[ac].fillna(X[ac].median()).values
        # Normalize anchor feature to [0, 1]
        col_min, col_max = col_m.min(), col_m.max()
        denom = (col_max - col_min) if col_max > col_min else 1.0
        X_meta_train[f'anchor_{ac}'] = (col_m - col_min) / denom
        X_meta_test[f'anchor_{ac}'] = (col_t - col_min) / denom

stack_oof = np.zeros(len(y))
stack_test = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(folds, 1):
    meta_m = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
    meta_m.fit(X_meta_train.iloc[tr_idx], y[tr_idx])
    stack_oof[val_idx] = meta_m.predict_proba(X_meta_train.iloc[val_idx])[:, 1]
    stack_test += meta_m.predict_proba(X_meta_test)[:, 1] / N_FOLDS

stack_ks = competition_ks(y, stack_oof)
oof_predictions['stacking_meta'] = stack_oof
test_predictions['stacking_meta'] = stack_test
experiment_results.append({'model': 'stacking_meta', 'mean_fold_ks': stack_ks, 'oof_ks': stack_ks})
print(f"==> Stacking Meta-Learner OOF KS: {stack_ks:.5f}")

# ============================================================
# EXPERIMENT 7: Ensemble Optimization (Blending & Stacking)
# ============================================================
print("\n" + "=" * 70)
print("EXPERIMENT 7: Ensemble Optimization (Probability & Rank Search)")
print("=" * 70)

candidate_models = ['lgb_depth7', 'lgb_extra', 'xgb_depth6', 'cb_depth7', 'segmented_lgb', 'stacking_meta']
K = len(candidate_models)

oof_matrix = np.column_stack([oof_predictions[m] for m in candidate_models])
test_matrix = np.column_stack([test_predictions[m] for m in candidate_models])

# Dirichlet Search
print("  Searching optimal probability blend weights (10,000 Dirichlet samples)...")
np.random.seed(SEED)
weights_samples = [np.ones(K) / K]
for i in range(K):
    w = np.zeros(K)
    w[i] = 1.0
    weights_samples.append(w)
for i in range(K):
    for j in range(i + 1, K):
        w = np.zeros(K)
        w[i], w[j] = 0.5, 0.5
        weights_samples.append(w)
for _ in range(10000):
    w = np.random.dirichlet(np.ones(K) * 1.5)
    weights_samples.append(w)

best_score = 0
best_w = None

for w in weights_samples:
    blend = oof_matrix @ w
    score = competition_ks(y, blend)
    if score > best_score:
        best_score = score
        best_w = w

print(f"  Best Prob Blend OOF KS: {best_score:.5f}")
for m, weight in zip(candidate_models, best_w):
    print(f"    {m:18s}: {weight:.4f}")

# Rank Blend Search
print("\n  Searching optimal rank blend weights (10,000 Dirichlet samples)...")
oof_rank_matrix = np.column_stack([rankdata(oof_predictions[m]) / len(y) for m in candidate_models])
test_rank_matrix = np.column_stack([rankdata(test_predictions[m]) / len(test_ids) for m in candidate_models])

best_rank_score = 0
best_rank_w = None

for w in weights_samples:
    blend_r = oof_rank_matrix @ w
    score = competition_ks(y, blend_r)
    if score > best_rank_score:
        best_rank_score = score
        best_rank_w = w

print(f"  Best Rank Blend OOF KS: {best_rank_score:.5f}")
for m, weight in zip(candidate_models, best_rank_w):
    print(f"    {m:18s}: {weight:.4f}")

# Choose winner
if best_rank_score > best_score:
    print("\n==> Rank-based blending selected as winning ensemble!")
    final_oof_ks = best_rank_score
    final_test_pred = test_rank_matrix @ best_rank_w
    blend_type = 'rank'
    winning_weights = dict(zip(candidate_models, [float(x) for x in best_rank_w]))
else:
    print("\n==> Probability blending selected as winning ensemble!")
    final_oof_ks = best_score
    final_test_pred = test_matrix @ best_w
    blend_type = 'prob'
    winning_weights = dict(zip(candidate_models, [float(x) for x in best_w]))

experiment_results.append({
    'model': f'ensemble_{blend_type}',
    'mean_fold_ks': final_oof_ks,
    'oof_ks': final_oof_ks
})

# ============================================================
# EXPERIMENT COMPARISON SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("EXPERIMENT COMPARISON SUMMARY")
print("=" * 70)
df_results = pd.DataFrame(experiment_results)
print(df_results.to_string(index=False))
print(f"\nFinal Ensemble OOF KS: {final_oof_ks:.5f}")

# ============================================================
# SAVE DELIVERABLES
# ============================================================
print("\n" + "=" * 70)
print("GENERATING FINAL SUBMISSION AND SAVING MODEL BUNDLE")
print("=" * 70)

submission_path = OUTPUT_DIR / 'DSC0102_JanjiLombaTerakhirTahunIni_Submisi.csv'
sub = pd.DataFrame({
    'application_id': test_ids,
    'default_probability': np.clip(final_test_pred, 0.0, 1.0)
})

# Verifications
assert len(sub) == 30000, f"Expected 30000 rows, got {len(sub)}"
assert sub['application_id'].isna().sum() == 0, "NaN in application_id!"
assert sub['default_probability'].isna().sum() == 0, "NaN in default_probability!"
assert (sub['default_probability'] >= 0).all() and (sub['default_probability'] <= 1).all(), "Values outside [0, 1]!"

sub.to_csv(submission_path, index=False)
print(f"Saved submission: {submission_path}")
print(sub.head(10).to_string())
print("\nPrediction distribution summary:")
print(sub['default_probability'].describe())

ensemble_bundle = {
    'blend_type': blend_type,
    'weights': winning_weights,
    'candidate_models': candidate_models,
    'oof_ks': final_oof_ks,
    'feature_names': feature_names,
    'experiment_results': experiment_results
}
joblib.dump(ensemble_bundle, OUTPUT_DIR / 'selected_ensemble.joblib', compress=3)
print(f"Saved ensemble bundle to: {OUTPUT_DIR / 'selected_ensemble.joblib'}")

print("\n" + "=" * 70)
print(f"TARGET CHECK: Minimum Score 0.57")
print(f"ACHIEVED SCORE: {final_oof_ks:.5f}")
if final_oof_ks >= 0.57:
    print("SUCCESS! Target >= 0.57 achieved!")
else:
    print(f"Current score is {final_oof_ks:.5f}.")
print("=" * 70)
