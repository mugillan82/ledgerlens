import pandas as pd
from datetime import datetime
from rapidfuzz import fuzz
from .exact_matcher import normalize_ref

def run_fuzzy_matcher(unmatched_bank, unmatched_gateway, threshold=75.0, ref_similarity_threshold=85.0):
    """
    Runs fuzzy matching on remaining unmatched records.
    Tolerates:
    - Amount difference: Gateway amount is 0% to 3% lower than Bank amount (fee deduction).
    - Date difference: Gateway date is 0 to 3 days after Bank date.
    - Reference similarity: rapidfuzz ratio > 85%.
    """
    bank = unmatched_bank.copy()
    gateway = unmatched_gateway.copy()
    
    # Pre-normalize and parse for performance
    bank["norm_ref"] = bank["reference_note"].apply(normalize_ref)
    gateway["norm_ref"] = gateway["reference_note"].apply(normalize_ref)
    
    bank["parsed_date"] = pd.to_datetime(bank["date"])
    gateway["parsed_date"] = pd.to_datetime(gateway["settlement_date"])
    
    matched_pairs = []
    matched_bank_ids = set()
    matched_gateway_ids = set()
    
    # Convert gateway to dict/list for easier iteration
    gateway_list = gateway.to_dict(orient="records")
    
    for _, b_row in bank.iterrows():
        b_id = b_row["txn_id"]
        b_amount = float(b_row["amount"])
        b_date = b_row["parsed_date"]
        b_ref = b_row["norm_ref"]
        
        best_candidate = None
        best_score = -1
        best_fee = 0.00
        best_gateway_idx = -1
        
        for g_idx, g_row in enumerate(gateway_list):
            g_id = g_row["order_id"]
            if g_id in matched_gateway_ids:
                continue
                
            g_amount = float(g_row["amount"])
            g_date = g_row["parsed_date"]
            g_ref = g_row["norm_ref"]
            
            # 1. Date tolerance: settlement date is 0 to 3 days after bank date
            date_diff = (g_date - b_date).days
            if not (0 <= date_diff <= 3):
                continue
                
            # 2. Amount tolerance: bank - gateway is 0% to 3% of bank amount
            amt_diff = b_amount - g_amount
            fee_pct = amt_diff / b_amount if b_amount > 0 else 0
            if not (0.0 <= fee_pct <= 0.0305): # small float allowance
                continue
                
            # 3. Reference similarity (normalized)
            ref_sim = fuzz.ratio(b_ref, g_ref)
            if ref_sim < ref_similarity_threshold:
                continue
                
            # Closeness calculations for confidence
            # date score: 0 days = 100, 1 day = 90, 2 days = 80, 3 days = 70
            date_score = 100 - (date_diff * 10)
            
            # amount/fee score is 100 since it passed the fee criteria
            amount_score = 100
            
            # Weighted average
            confidence = (0.5 * ref_sim) + (0.3 * amount_score) + (0.2 * date_score)
            
            if confidence >= threshold and confidence > best_score:
                best_score = confidence
                best_candidate = g_id
                best_fee = round(amt_diff, 2)
                best_gateway_idx = g_idx
                
        if best_candidate:
            matched_pairs.append({
                "bank_txn_id": b_id,
                "gateway_order_id": best_candidate,
                "match_type": "FUZZY_DATE" if best_score < 90 and (best_fee == 0) else "FUZZY_AMOUNT" if best_fee > 0 else "FUZZY",
                "confidence": round(best_score, 2),
                "fee": best_fee
            })
            matched_bank_ids.add(b_id)
            matched_gateway_ids.add(best_candidate)
            
    # Filter leftover unmatched
    still_unmatched_bank = df_bank = unmatched_bank[~unmatched_bank["txn_id"].isin(matched_bank_ids)].copy()
    still_unmatched_gateway = unmatched_gateway[~unmatched_gateway["order_id"].isin(matched_gateway_ids)].copy()
    
    return matched_pairs, still_unmatched_bank, still_unmatched_gateway
