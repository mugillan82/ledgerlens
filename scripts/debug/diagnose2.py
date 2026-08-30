"""Trace TYPO/CONFUSABLE ground truth records to understand the reference format mismatch."""
import pandas as pd, sys
sys.path.insert(0, '.')
from src.exact_matcher import normalize_ref
from rapidfuzz import fuzz

bank = pd.read_csv('data/raw/bank_statement.csv')
gw   = pd.read_csv('data/raw/gateway_settlement.csv')
gt   = pd.read_csv('data/raw/ground_truth.csv')

b20 = bank[bank.txn_id=='TXN-00020'].iloc[0]
gt20 = gt[gt.bank_txn_id=='TXN-00020'].iloc[0]
print("GT for TXN-00020:", dict(gt20))
print("Bank ref:", b20['reference_note'])

g_id = gt20['gateway_order_id']
if g_id != 'NO_MATCH':
    g20 = gw[gw.order_id==g_id].iloc[0]
    print("True GW ref:", g20['reference_note'])
    nb = normalize_ref(b20['reference_note'])
    ng = normalize_ref(g20['reference_note'])
    print("norm_bank:", nb)
    print("norm_gw  :", ng)
    print("fuzz_sim :", fuzz.ratio(nb, ng))
else:
    print("GT says NO_MATCH for this bank record")

print()
print("=== TYPO/CONFUSABLE ground truth sample ===")
for _, r in gt[gt['match_type'].isin(['TYPO','CONFUSABLE'])].head(6).iterrows():
    b_id = r['bank_txn_id']
    g_id = r['gateway_order_id']
    mtype = r['match_type']
    if b_id == 'NO_MATCH' or g_id == 'NO_MATCH':
        continue
    b_row = bank[bank.txn_id==b_id]
    g_row = gw[gw.order_id==g_id]
    if b_row.empty or g_row.empty:
        continue
    b_ref = b_row.iloc[0]['reference_note']
    g_ref = g_row.iloc[0]['reference_note']
    nb = normalize_ref(b_ref)
    ng = normalize_ref(g_ref)
    sim = fuzz.ratio(nb, ng)
    print(f"type={mtype}  {b_id}<->{g_id}")
    print(f"  bank_ref={b_ref!r}  norm={nb!r}")
    print(f"  gw_ref  ={g_ref!r}  norm={ng!r}")
    print(f"  fuzz_sim={sim}")
    print()

print("=== Confidence distribution before and after fix ===")
m = pd.read_csv('data/processed/matches.csv')
ml = m[m['match_type']=='ML_MATCH']
print("Unique confidences:", ml['confidence'].nunique())
print(ml['confidence'].value_counts().head(10))
