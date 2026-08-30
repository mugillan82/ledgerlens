import re
import pandas as pd

def normalize_ref(ref):
    """Normalize a reference string for comparison.
    Strip common prefixes (REF-, TXN-, etc.) and whitespace so that
    'TXN1469' and 'REF-TXN1469' both reduce to the same base token.
    This prevents structural prefix differences from causing identical
    fuzz.ratio values across unrelated pairs.
    """
    if pd.isna(ref) or not isinstance(ref, str):
        ref = str(ref) if not pd.isna(ref) else ""
    ref = ref.upper().strip()
    # Strip all known prefixes iteratively
    for prefix in ["REF-", "TXN-", "REF", "TXN"]:
        if ref.startswith(prefix):
            ref = ref[len(prefix):]
            break  # only strip one prefix level per pass
    # Strip a second layer for 'REF-TXN1234' -> 'TXN1234' -> '1234'
    for prefix in ["REF-", "TXN-", "REF", "TXN"]:
        if ref.startswith(prefix):
            ref = ref[len(prefix):]
            break
    ref = ref.replace("-", "").replace(" ", "")
    return ref

def run_exact_matcher(df_bank, df_gateway):
    """
    Matches records where reference numbers (normalized), amounts, and dates match exactly.
    """
    # Create copies to avoid side effects
    bank = df_bank.copy()
    gateway = df_gateway.copy()
    
    # Normalize reference notes
    bank["norm_ref"] = bank["reference_note"].apply(normalize_ref)
    gateway["norm_ref"] = gateway["reference_note"].apply(normalize_ref)
    
    # Ensure amount and date are comparable
    bank["date_str"] = pd.to_datetime(bank["date"]).dt.strftime("%Y-%m-%d")
    gateway["date_str"] = pd.to_datetime(gateway["settlement_date"]).dt.strftime("%Y-%m-%d")
    bank["amount_val"] = bank["amount"].astype(float).round(2)
    gateway["amount_val"] = gateway["amount"].astype(float).round(2)
    
    matched_pairs = []
    matched_bank_ids = set()
    matched_gateway_ids = set()
    
    # Build lookup map for gateway to make matching fast and 1:1
    # Key: (norm_ref, amount_val, date_str) -> list of order_ids/indices
    gateway_lookup = {}
    for idx, row in gateway.iterrows():
        key = (row["norm_ref"], row["amount_val"], row["date_str"])
        gateway_lookup.setdefault(key, []).append(row["order_id"])
        
    for idx, row in bank.iterrows():
        key = (row["norm_ref"], row["amount_val"], row["date_str"])
        if key in gateway_lookup and len(gateway_lookup[key]) > 0:
            # Match found
            g_order_id = gateway_lookup[key].pop(0) # 1:1 pop
            matched_pairs.append({
                "bank_txn_id": row["txn_id"],
                "gateway_order_id": g_order_id,
                "match_type": "EXACT",
                "confidence": 100.0,
                "fee": 0.00
            })
            matched_bank_ids.add(row["txn_id"])
            matched_gateway_ids.add(g_order_id)
            
    # filter out matched records to get unmatched leftovers
    unmatched_bank = df_bank[~df_bank["txn_id"].isin(matched_bank_ids)].copy()
    unmatched_gateway = df_gateway[~df_gateway["order_id"].isin(matched_gateway_ids)].copy()
    
    return matched_pairs, unmatched_bank, unmatched_gateway
