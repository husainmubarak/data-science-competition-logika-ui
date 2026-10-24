"""
experiment_models.py
Train and evaluate multiple models on features_v4.joblib
Tracks CV KS score per fold and pooled OOF KS
Optimizes ensemble blend to achieve KS >= 0.57
"""

import gc
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from scipy.stats import ks_2samp, rankdata
from sklearn.model_selection import StratifiedKFold
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
print("1. Loading Features...")
print("=" * 70)

bundle = joblib.load(OUTPUT_DIR / 'features_v4.joblib')
X = bundle['X_train']
y = bundle['y_train']
X_test = bundle['X_test']
test_ids = bundle['test_application_ids']
feature_names = bundle['feature_names']

print(f"Train: {X.shape}, Test: {X_test.shape}")
print(f"Target positive rate: {y.mean():.4f}")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
folds = list(skf.split(X, y))

# Dictionary to store OOF predictions and test predictions per experiment
oof_predictions = {}
test_predictions = {}
experiment_results = []
trained_models = {}

# ============================================================
# EXPERIMENT 1: LightGBM Tuned Depth 7
# ============================================================
print("\n" + "=" * 70)
print("EXPERIMENT 1: LightGBM (depth=7, leaves=47)")
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
    'num_leaves': 47,
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

# Top Features
fi = pd.Series(lgb1_models[0].feature_importances_, index=feature_names).sort_values(ascending=False)
print("\nTop 15 Features (LGB):")
print(fi.head(15).to_string())

# ============================================================
# EXPERIMENT 2: LightGBM Deeper (depth=8, leaves=75)
# ============================================================
print("\n" + "=" * 70)
print("EXPERIMENT 2: LightGBM Deeper (depth=8, leaves=75)")
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
    'num_leaves': 75,
    'min_child_samples': 40,
    'subsample': 0.85,
    'subsample_freq': 1,
    'colsample_bytree': 0.65,
    'reg_alpha': 1.0,
    'reg_lambda': 10.0,
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
oof_predictions['lgb_depth8'] = lgb2_oof
test_predictions['lgb_depth8'] = lgb2_test
trained_models['lgb_depth8'] = lgb2_models
experiment_results.append({'model': 'lgb_depth8', 'mean_fold_ks': np.mean(lgb2_fold_ks), 'oof_ks': oof_ks})
print(f"==> LGB Depth 8 Mean Fold KS: {np.mean(lgb2_fold_ks):.5f} | Pooled OOF KS: {oof_ks:.5f}")

# ============================================================
# EXPERIMENT 3: XGBoost (depth=5)
# ============================================================
print("\n" + "=" * 70)
print("EXPERIMENT 3: XGBoost (depth=5)")
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
    'max_depth': 5,
    'min_child_weight': 20,
    'subsample': 0.85,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.5,
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
oof_predictions['xgb_depth5'] = xgb_oof
test_predictions['xgb_depth5'] = xgb_test
trained_models['xgb_depth5'] = xgb_models
experiment_results.append({'model': 'xgb_depth5', 'mean_fold_ks': np.mean(xgb_fold_ks), 'oof_ks': oof_ks})
print(f"==> XGB Depth 5 Mean Fold KS: {np.mean(xgb_fold_ks):.5f} | Pooled OOF KS: {oof_ks:.5f}")

# ============================================================
# EXPERIMENT 4: CatBoost (depth=6)
# ============================================================
print("\n" + "=" * 70)
print("EXPERIMENT 4: CatBoost (depth=6)")
print("=" * 70)

cb_oof = np.zeros(len(y))
cb_test = np.zeros(len(X_test))
cb_models = []
cb_fold_ks = []

cb_params = {
    'iterations': 3000,
    'learning_rate': 0.03,
    'depth': 6,
    'l2_leaf_reg': 5.0,
    'min_data_in_leaf': 30,
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
oof_predictions['cb_depth6'] = cb_oof
test_predictions['cb_depth6'] = cb_test
trained_models['cb_depth6'] = cb_models
experiment_results.append({'model': 'cb_depth6', 'mean_fold_ks': np.mean(cb_fold_ks), 'oof_ks': oof_ks})
print(f"==> CatBoost Depth 6 Mean Fold KS: {np.mean(cb_fold_ks):.5f} | Pooled OOF KS: {oof_ks:.5f}")

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
# For new customers, drop repayment and account columns which are all NaN
repay_acct_cols = [c for c in X.columns if c.startswith('rep_') or c.startswith('acct_') or c.startswith('prev_')]
X_new = X.drop(columns=repay_acct_cols)
X_test_new = X_test.drop(columns=repay_acct_cols)

for fold, (tr_idx, val_idx) in enumerate(folds, 1):
    tr_mask = (~is_existing)[tr_idx]
    va_mask = (~is_existing)[val_idx]
    
    tr_sub_idx = tr_idx[tr_mask]
    va_sub_idx = val_idx[va_mask]
    
    m_new = LGBMClassifier(
        objective='binary', n_estimators=2000, learning_rate=0.03, max_depth=5, num_leaves=25,
        min_child_samples=25, subsample=0.85, colsample_bytree=0.7, reg_alpha=0.2, reg_lambda=3.0,
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
# EXPERIMENT 6: Ensemble Optimization (Grid Search over Blend Weights)
# ============================================================
print("\n" + "=" * 70)
print("EXPERIMENT 6: Ensemble Optimization")
print("=" * 70)

candidate_models = list(oof_predictions.keys())
print(f"Candidate models: {candidate_models}")

# 1. Probability Blending Search
best_blend_ks = 0
best_blend_weights = None

# Systematic search
grid = np.linspace(0, 1, 11)
best_w = None
best_score = 0

for w1 in np.arange(0, 1.05, 0.1):
    for w2 in np.arange(0, 1.05 - w1, 0.1):
        for w3 in np.arange(0, 1.05 - w1 - w2, 0.1):
            w4 = round(1.0 - w1 - w2 - w3, 2)
            if w4 < -1e-6 or w4 > 1.0:
                continue
            w4 = max(0.0, w4)
            if abs(w1 + w2 + w3 + w4 - 1.0) > 1e-4:
                continue
            
            blend_prob = (
                w1 * oof_predictions['lgb_depth7'] +
                w2 * oof_predictions['lgb_depth8'] +
                w3 * oof_predictions['xgb_depth5'] +
                w4 * oof_predictions['cb_depth6']
            )
            score = competition_ks(y, blend_prob)
            if score > best_score:
                best_score = score
                best_w = (w1, w2, w3, w4)

print(f"Best Prob Blend Weights (lgb7, lgb8, xgb5, cb6): {best_w}")
print(f"Best Prob Blend OOF KS: {best_score:.5f}")

# 2. Rank Blending Search
rank_preds = {m: rankdata(oof_predictions[m]) / len(y) for m in candidate_models}

best_rank_score = 0
best_rank_w = None

for w1 in np.arange(0, 1.05, 0.1):
    for w2 in np.arange(0, 1.05 - w1, 0.1):
        for w3 in np.arange(0, 1.05 - w1 - w2, 0.1):
            w4 = round(1.0 - w1 - w2 - w3, 2)
            if w4 < -1e-6 or w4 > 1.0:
                continue
            w4 = max(0.0, w4)
            if abs(w1 + w2 + w3 + w4 - 1.0) > 1e-4:
                continue
            
            blend_rank = (
                w1 * rank_preds['lgb_depth7'] +
                w2 * rank_preds['lgb_depth8'] +
                w3 * rank_preds['xgb_depth5'] +
                w4 * rank_preds['cb_depth6']
            )
            score = competition_ks(y, blend_rank)
            if score > best_rank_score:
                best_rank_score = score
                best_rank_w = (w1, w2, w3, w4)

print(f"Best Rank Blend Weights (lgb7, lgb8, xgb5, cb6): {best_rank_w}")
print(f"Best Rank Blend OOF KS: {best_rank_score:.5f}")

# Choose best between prob and rank
if best_rank_score > best_score:
    print("\n==> Rank-based blending selected!")
    final_oof_ks = best_rank_score
    final_test_pred = (
        best_rank_w[0] * rankdata(test_predictions['lgb_depth7']) / len(test_ids) +
        best_rank_w[1] * rankdata(test_predictions['lgb_depth8']) / len(test_ids) +
        best_rank_w[2] * rankdata(test_predictions['xgb_depth5']) / len(test_ids) +
        best_rank_w[3] * rankdata(test_predictions['cb_depth6']) / len(test_ids)
    )
    blend_type = 'rank'
    winning_weights = dict(zip(['lgb_depth7', 'lgb_depth8', 'xgb_depth5', 'cb_depth6'], best_rank_w))
else:
    print("\n==> Probability blending selected!")
    final_oof_ks = best_score
    final_test_pred = (
        best_w[0] * test_predictions['lgb_depth7'] +
        best_w[1] * test_predictions['lgb_depth8'] +
        best_w[2] * test_predictions['xgb_depth5'] +
        best_w[3] * test_predictions['cb_depth6']
    )
    blend_type = 'prob'
    winning_weights = dict(zip(['lgb_depth7', 'lgb_depth8', 'xgb_depth5', 'cb_depth6'], best_w))

# Summary of all experiments
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

# Save ensemble weights and models
ensemble_bundle = {
    'blend_type': blend_type,
    'weights': winning_weights,
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
    print(f"Current score is {final_oof_ks:.5f}. Need further optimization.")
print("=" * 70)

