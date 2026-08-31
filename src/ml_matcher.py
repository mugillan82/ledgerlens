import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from rapidfuzz import fuzz
from .exact_matcher import normalize_ref

# Helper to compute features between a bank row and a gateway row
def compute_features(bank_row, gate_row):
    b_amt = float(bank_row["amount"])
    g_amt = float(gate_row["amount"])
    
    b_date = pd.to_datetime(bank_row["date"])
    g_date = pd.to_datetime(gate_row["settlement_date"])
    
    b_ref = normalize_ref(bank_row["reference_note"])
    g_ref = normalize_ref(gate_row["reference_note"])
    
    # 1. amount_diff_pct (as a fraction, e.g. 0.025 = 2.5%)
    amt_diff = abs(b_amt - g_amt)
    amount_diff_pct = (amt_diff / b_amt) if b_amt > 0 else 0.0
    
    # 2. date_diff_days
    date_diff_days = float(abs((g_date - b_date).days))
    
    # 3. reference_similarity (0-100 fuzz ratio on fully stripped refs)
    ref_sim = float(fuzz.ratio(b_ref, g_ref))
    
    # 4. amount_within_fee_range
    fee_pct = (b_amt - g_amt) / b_amt if b_amt > 0 else -1.0
    amount_within_fee_range = 1.0 if (0.01 <= fee_pct <= 0.0305) else 0.0

    # 5. ref_exact_match: 1.0 when the stripped numeric cores are identical.
    # This cleanly separates same-ref pairs (score=100, exact=1) from
    # near-duplicate confusables where refs happen to score 75% after stripping.
    ref_exact_match = 1.0 if b_ref == g_ref else 0.0
    
    return [amount_diff_pct, date_diff_days, ref_sim, amount_within_fee_range, ref_exact_match]


def train_ml_classifier(raw_dir="data/raw", model_dir="models"):
    """
    Offline training phase. Compares Logistic Regression, Random Forest, and Gradient Boosting,
    selects the highest F1 model, and saves it.
    """
    from src.model_comparison import compare_and_train_models
    results_df, best_model = compare_and_train_models(raw_dir, model_dir)
    return results_df, best_model

def run_ml_matcher(unmatched_bank, unmatched_gateway, model_path="models/match_classifier.pkl", threshold=0.7):
    """
    Live matching inference. Predicts matches using the saved classifier model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Please run training first.")
        
    clf = joblib.load(model_path)
    
    bank = unmatched_bank.copy()
    gateway = unmatched_gateway.copy()
    
    matched_pairs = []
    matched_bank_ids = set()
    matched_gateway_ids = set()
    
    gateway_list = gateway.to_dict(orient="records")
    
    feature_names = ["amount_diff_pct", "date_diff_days", "reference_similarity",
                     "amount_within_fee_range", "ref_exact_match"]
    
    for _, b_row in bank.iterrows():
        b_id = b_row["txn_id"]
        
        best_candidate = None
        best_prob = -1.0
        best_features = []
        
        for g_row in gateway_list:
            g_id = g_row["order_id"]
            if g_id in matched_gateway_ids:
                continue
                
            features = compute_features(b_row, g_row)
            prob = clf.predict_proba([features])[0][1]
            
            if prob >= threshold and prob > best_prob:
                best_prob = prob
                best_candidate = g_id
                best_features = features
                
        if best_candidate:
            # ── Explainability: real per-pair values ────────────────────────────
            # Unpack actual feature values for this specific pair
            amt_diff_pct_val  = best_features[0]   # fraction, e.g. 0.0193
            date_diff_val     = best_features[1]   # days
            ref_sim_val       = best_features[2]   # 0-100
            fee_range_val     = best_features[3]   # 0 or 1
            ref_exact_val     = best_features[4]   # 0 or 1

            # Build unique explanation using the pair's actual numbers
            parts = []
            if ref_exact_val == 1.0:
                parts.append(f"exact reference match (ref similarity {ref_sim_val:.1f}%)")
            elif ref_sim_val >= 85:
                parts.append(f"high reference similarity ({ref_sim_val:.1f}%, refs differ slightly)")
            elif ref_sim_val >= 60:
                parts.append(f"partial reference similarity ({ref_sim_val:.1f}%, refs differ)")
            else:
                parts.append(f"low reference similarity ({ref_sim_val:.1f}%) — numeric signals dominated")

            if amt_diff_pct_val == 0.0:
                parts.append("identical amounts")
            elif fee_range_val > 0:
                parts.append(f"amount within 1-3% fee band ({amt_diff_pct_val * 100:.2f}% diff)")
            else:
                parts.append(f"amount diff {amt_diff_pct_val * 100:.2f}% (outside normal fee band)")

            if date_diff_val == 0:
                parts.append("same-day settlement")
            else:
                parts.append(f"{int(date_diff_val)}-day date gap")

            explanation = f"ML match: {'; '.join(parts)}"

            # ── Fee: percentage of bank amount (gateway processing fee) ─────────
            b_amt = float(b_row["amount"])
            g_row_match = [g for g in gateway_list if g["order_id"] == best_candidate][0]
            g_amt = float(g_row_match["amount"])
            raw_fee_pct = (b_amt - g_amt) / b_amt if b_amt > 0 else 0.0
            # Cap at ±10% to filter data anomalies; store as a percentage value
            fee_pct = max(-10.0, min(10.0, raw_fee_pct * 100))

            matched_pairs.append({
                "bank_txn_id": b_id,
                "gateway_order_id": best_candidate,
                "match_type": "ML_MATCH",
                "confidence": round(best_prob * 100, 2),
                "fee_pct": round(fee_pct, 4),
                "fee": round(b_amt - g_amt, 2),   # kept for reporter use
                "explanation": explanation
            })
            matched_bank_ids.add(b_id)
            matched_gateway_ids.add(best_candidate)
            
    still_unmatched_bank = unmatched_bank[~unmatched_bank["txn_id"].isin(matched_bank_ids)].copy()
    still_unmatched_gateway = unmatched_gateway[~unmatched_gateway["order_id"].isin(matched_gateway_ids)].copy()
    
    return matched_pairs, still_unmatched_bank, still_unmatched_gateway
