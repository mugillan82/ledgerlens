"""Diagnostic: print raw feature values for the identical-confidence pairs."""
import pandas as pd
from rapidfuzz import fuzz
import sys, joblib, numpy as np
sys.path.insert(0, '.')
from src.exact_matcher import normalize_ref
from src.ml_matcher import compute_features

bank = pd.read_csv('data/raw/bank_statement.csv')
gw   = pd.read_csv('data/raw/gateway_settlement.csv')
matches_csv = pd.read_csv('data/processed/matches.csv')
print("matches.csv columns:", list(matches_csv.columns))
print()

bank_map = bank.set_index('txn_id').to_dict(orient='index')
gw_map   = gw.set_index('order_id').to_dict(orient='index')

# Load the trained model so we can re-run predict_proba
clf = joblib.load('models/match_classifier.pkl')

target_pairs = [
    ('TXN-00020','ORD-00244'),
    ('TXN-00022','ORD-00040'),
    ('TXN-00032','ORD-00487'),
    ('TXN-00038','ORD-00135'),
    ('TXN-00014','ORD-00151'),   # different pair for comparison
    ('TXN-00017','ORD-00441'),
]

print("=== RAW FEATURE VALUES + LIVE predict_proba ===")
for b_id, g_id in target_pairs:
    br = bank_map.get(b_id)
    gr = gw_map.get(g_id)
    if not br or not gr:
        print(f"{b_id}/{g_id}: NOT FOUND IN DATA")
        continue
    b_ref = normalize_ref(str(br['reference_note']))
    g_ref = normalize_ref(str(gr['reference_note']))
    feats = compute_features(br, gr)
    live_prob = clf.predict_proba([feats])[0][1]
    row = matches_csv[matches_csv['bank_txn_id'] == b_id]
    saved_conf = row.iloc[0]['confidence'] if not row.empty else 'NOT IN CSV'
    fee_col    = row.iloc[0]['fee'] if ('fee' in matches_csv.columns and not row.empty) else 'NO COL'
    fee_pct_col= row.iloc[0]['fee_pct'] if ('fee_pct' in matches_csv.columns and not row.empty) else 'NO COL'
    print(f"{b_id}/{g_id}")
    print(f"  bank_ref={br['reference_note']!r}  gw_ref={gr['reference_note']!r}")
    print(f"  norm_bank={b_ref!r}  norm_gw={g_ref!r}")
    print(f"  features: amt_diff_pct={feats[0]:.6f}  date_diff={feats[1]}  ref_sim={feats[2]}  in_fee={feats[3]}")
    print(f"  live_prob={live_prob*100:.4f}%  saved_conf={saved_conf}")
    print(f"  csv fee={fee_col}  csv fee_pct={fee_pct_col}")
    print()

print("=== ALL ML MATCHES — confidence distribution ===")
ml = matches_csv[matches_csv['match_type'] == 'ML_MATCH']
print(f"Total ML matches: {len(ml)}")
print(f"Unique confidence values: {ml['confidence'].nunique()}")
print("Top 10 confidence value counts:")
print(ml['confidence'].value_counts().head(10))
print()
if 'fee_pct' in ml.columns:
    print("fee_pct non-null:", ml['fee_pct'].notna().sum(), "/ null:", ml['fee_pct'].isna().sum())
    print("fee_pct sample:", ml['fee_pct'].dropna().head(5).tolist())
else:
    print("fee_pct column NOT in matches.csv")
if 'fee' in ml.columns:
    print("fee non-null:", ml['fee'].notna().sum(), "/ null:", ml['fee'].isna().sum())
    print("fee sample:", ml['fee'].head(5).tolist())
