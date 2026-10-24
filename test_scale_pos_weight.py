"""
test_scale_pos_weight.py
Test if scale_pos_weight or focal loss or learning rate changes KS
"""
import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier, early_stopping

b = joblib.load('artifacts/features_v5.joblib')
X = b['X_train']
y = b['y_train']

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(skf.split(X, y))

for spw in [1.0, 1.5, 2.0, 2.5, 3.5, 5.0]:
    oof = np.zeros(len(y))
    fold_ks = []
    for tr, val in folds:
        m = LGBMClassifier(
            objective='binary',
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=7,
            num_leaves=63,
            min_child_samples=30,
            subsample=0.85,
            colsample_bytree=0.7,
            scale_pos_weight=spw,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )
        m.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[val], y[val])], eval_metric='auc', callbacks=[early_stopping(80, verbose=False)])
        preds = m.predict_proba(X.iloc[val])[:, 1]
        oof[val] = preds
        fold_ks.append(ks_2samp(preds[y[val]==1], preds[y[val]==0]).statistic)
    
    pooled = ks_2samp(oof[y==1], oof[y==0]).statistic
    print(f"scale_pos_weight={spw:4.1f} | Mean Fold KS: {np.mean(fold_ks):.5f} | Pooled OOF KS: {pooled:.5f}")
