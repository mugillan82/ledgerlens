"""Final verification: print 5 ML pairs with features, confidence, fee — confirm all distinct."""
import pandas as pd, sys, joblib
sys.path.insert(0, '.')
from src.ml_matcher import compute_features

bank = pd.read_csv('data/raw/bank_statement.csv')
gw   = pd.read_csv('data/raw/gateway_settlement.csv')
m    = pd.read_csv('data/processed/matches.csv')
clf  = joblib.load('models/match_classifier.pkl')
bank_map = bank.set_index('txn_id').to_dict(orient='index')
gw_map   = gw.set_index('order_id').to_dict(orient='index')

ml = m[m['match_type'] == 'ML_MATCH'].head(8)
print("=== PER-PAIR VERIFICATION: features, confidence, fee ===")
for _, row in ml.iterrows():
    b_id = row['bank_txn_id']
    g_id = row['gateway_order_id']
    br = bank_map.get(b_id, {})
    gr = gw_map.get(g_id, {})
    if not br or not gr:
        continue
    feats = compute_features(br, gr)
    conf  = clf.predict_proba([feats])[0][1] * 100
    fee   = row['fee']
    fee_pct = row['fee_pct']
    expl  = row['explanation']
    print(f"{b_id} <-> {g_id}")
    print(f"  features: amt_diff={feats[0]*100:.4f}%  date_gap={feats[1]:.0f}d  ref_sim={feats[2]:.1f}  in_fee_band={feats[3]}  ref_exact={feats[4]}")
    print(f"  confidence={conf:.4f}%  |  fee_pct={fee_pct:.4f}%  |  fee=Rs.{fee:.2f}")
    print(f"  explanation: {expl}")
    print()

print()
print("=== CONFIDENCE DISTRIBUTION (all ML matches) ===")
ml_all = m[m['match_type'] == 'ML_MATCH']
print(f"Total ML: {len(ml_all)}  |  Unique confidence values: {ml_all['confidence'].nunique()}")
print(ml_all['confidence'].value_counts().head(12).to_string())
print()
print(f"fee non-null: {ml_all['fee'].notna().sum()}  |  fee=None: {ml_all['fee'].isna().sum()}")
print(f"fee range: Rs.{ml_all['fee'].min():.2f} to Rs.{ml_all['fee'].max():.2f}")
print(f"fee_pct range: {ml_all['fee_pct'].min():.4f}% to {ml_all['fee_pct'].max():.4f}%")
