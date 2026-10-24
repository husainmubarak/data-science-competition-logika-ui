"""
deep_signal_search.py
Exhaustively search for hidden patterns, leaks, ID patterns, datetime patterns, or strong feature combinations
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import ks_2samp

DATA_DIR = Path('D:/LOGIKA UI/code/data-science-competition-logika-ui-2026')

train = pd.read_csv(DATA_DIR / 'train_applications.csv', parse_dates=['application_datetime', 'decision_datetime'])
y = train['target'].values
n = len(train)

# 1. ID properties
print("1. Checking ID properties...")
train['id_num'] = train['application_id'].str.extract('(\d+)').astype(float)
train['id_len'] = train['application_id'].str.len()
print("  ID num corr:", train['id_num'].corr(pd.Series(y)))

# 2. Datetime properties
print("2. Checking Datetime properties...")
train['app_year'] = train['application_datetime'].dt.year
train['app_month'] = train['application_datetime'].dt.month
train['app_day'] = train['application_datetime'].dt.day
train['app_dayofyear'] = train['application_datetime'].dt.dayofyear
train['app_sec'] = train['application_datetime'].dt.second
train['app_min'] = train['application_datetime'].dt.minute
train['app_epoch'] = train['application_datetime'].astype('int64') // 10**9

for col in ['app_year', 'app_month', 'app_day', 'app_dayofyear', 'app_sec', 'app_epoch']:
    corr = train[col].corr(pd.Series(y))
    print(f"  {col:15s} corr: {corr:.4f}")

# 3. Monthly default rates over time
print("\n3. Default rates by month:")
print(train.groupby('app_month')['target'].agg(['count', 'mean']))

# 4. Loan purpose default rates
print("\n4. Default rates by loan purpose:")
print(train.groupby('loan_purpose')['target'].agg(['count', 'mean']))

# 5. Loan amount & term combinations
print("\n5. Default rates by term months:")
print(train.groupby('term_months')['target'].agg(['count', 'mean']))

# 6. Interest rate default rates
print("\n6. Default rates by annual interest rate:")
print(train.groupby('annual_interest_rate')['target'].agg(['count', 'mean']))
