"""
test_new_customer_model.py
Evaluate model performance exclusively on new customers
"""
import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier, early_stopping
from catboost import CatBoostClassifier

b = joblib.load('artifacts/features_v5.joblib')
X = b['X_train']
y = b['y_train']

is_ex = (X['has_repayment_history'] == 1).values

# Filter to new customers only
X_new = X[~is_ex].copy()
y_new = y[~is_ex]

# Drop all repayment, account, and previous application columns that are all NaN
drop_cols = [c for c in X_new.columns if c.startswith('rep_') or c.startswith('acct_') or c.startswith('prev_')]
X_new = X_new.drop(columns=drop_cols)

print(f"New customers matrix: {X_new.shape}, positive rate: {y_new.mean():.4f}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(skf.split(X_new, y_new))

# Model 1: LightGBM
lgb_oof = np.zeros(len(y_new))
fold_ks_lgb = []

for tr, val in folds:
    m = LGBMClassifier(
        objective='binary',
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=6,
        num_leaves=35,
        min_child_samples=25,
        subsample=0.85,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=5.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )
    m.fit(X_new.iloc[tr], y_new[tr], eval_set=[(X_new.iloc[val], y_new[val])], eval_metric='auc', callbacks=[early_stopping(80, verbose=False)])
    preds = m.predict_proba(X_new.iloc[val])[:, 1]
    lgb_oof[val] = preds
    fold_ks_lgb.append(ks_2samp(preds[y_new[val]==1], preds[y_new[val]==0]).statistic)

pooled_lgb = ks_2samp(lgb_oof[y_new==1], lgb_oof[y_new==0]).statistic
print(f"LGBM New Customers: Mean Fold KS: {np.mean(fold_ks_lgb):.5f} | Pooled KS: {pooled_lgb:.5f}")

# Top 15 features for new customers
fi = pd.Series(m.feature_importances_, index=X_new.columns).sort_values(ascending=False)
print("\nTop 15 Features for New Customers:")
print(fi.head(15).to_string())
