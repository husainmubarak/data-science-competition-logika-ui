"""
test_ecom_signals.py
Test additional e-commerce signals:
- days_since_last_order
- cod_ratio
- credit_card_ratio
- essential_vs_discretionary
- transaction frequency
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import ks_2samp

DATA_DIR = Path('D:/LOGIKA UI/code/data-science-competition-logika-ui-2026')

train = pd.read_csv(DATA_DIR / 'train_applications.csv', parse_dates=['decision_datetime'])
y = train['target'].values
cust_cutoff = dict(zip(train['customer_id'], train['decision_datetime']))

print("Reading ecommerce sample (1,000,000 rows) to test signals...")
ec_chunks = []
for chunk in pd.read_csv(DATA_DIR / 'ecommerce_transactions.csv', parse_dates=['transaction_datetime'], chunksize=500_000):
    chunk['cutoff'] = chunk['customer_id'].map(cust_cutoff)
    chunk = chunk[chunk['cutoff'].notna()]
    chunk = chunk[chunk['transaction_datetime'] <= chunk['cutoff']]
    if chunk.empty:
        continue
    chunk['days_before'] = (chunk['cutoff'] - chunk['transaction_datetime']).dt.total_seconds() / 86400.0
    ec_chunks.append(chunk)
    if len(ec_chunks) >= 3:
        break

ec = pd.concat(ec_chunks, ignore_index=True)
print(f"Processed {len(ec)} transactions.")

ec['is_completed'] = (ec['order_status'] == 'completed').astype(int)
ec['is_cod'] = (ec['payment_method'] == 'COD').astype(int)
ec['is_cc'] = (ec['payment_method'] == 'credit_card').astype(int)
ec['is_groceries'] = (ec['product_category'] == 'groceries').astype(int)

# Group by customer
ec_agg = ec.groupby('customer_id').agg(
    ec_days_since_last=('days_before', 'min'),
    ec_tx_count=('order_item_id', 'count'),
    ec_cod_count=('is_cod', 'sum'),
    ec_cc_count=('is_cc', 'sum'),
    ec_groceries_count=('is_groceries', 'sum'),
    ec_item_total_mean=('item_total', 'mean'),
    ec_item_total_std=('item_total', 'std'),
).reset_index()

ec_agg['cod_ratio'] = ec_agg['ec_cod_count'] / (ec_agg['ec_tx_count'] + 1)
ec_agg['cc_ratio'] = ec_agg['ec_cc_count'] / (ec_agg['ec_tx_count'] + 1)
ec_agg['groceries_ratio'] = ec_agg['ec_groceries_count'] / (ec_agg['ec_tx_count'] + 1)

m = train.merge(ec_agg, on='customer_id', how='left')

for col in ['ec_days_since_last', 'ec_tx_count', 'cod_ratio', 'cc_ratio', 'groceries_ratio', 'ec_item_total_mean', 'ec_item_total_std']:
    corr = m[col].corr(pd.Series(y))
    vals = m[col].fillna(m[col].median()).values
    ks = ks_2samp(vals[y == 1], vals[y == 0]).statistic
    print(f"  {col:25s} | corr: {corr:+.4f} | KS: {ks:.4f}")
