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
    Offline training phase. Builds a training set from ground_truth.csv and raw statements,
    trains a Logistic Regression classifier, and saves it.
    """
    print("\n--- Training ML Match Classifier ---")
    df_bank = pd.read_csv(os.path.join(raw_dir, "bank_statement.csv"))
    df_gateway = pd.read_csv(os.path.join(raw_dir, "gateway_settlement.csv"))
    df_gt = pd.read_csv(os.path.join(raw_dir, "ground_truth.csv"))
    
    # Maps for lookup
    bank_map = df_bank.set_index("txn_id").to_dict(orient="index")
    gateway_map = df_gateway.set_index("order_id").to_dict(orient="index")
    
    X = []
    y = []
    
    # Process positive examples from ground truth matches
    true_pairs = []
    for _, row in df_gt.iterrows():
        b_id = row["bank_txn_id"]
        g_id = row["gateway_order_id"]
        if b_id != "NO_MATCH" and g_id != "NO_MATCH":
            true_pairs.append((b_id, g_id))
            
    print(f"Extracting {len(true_pairs)} positive match samples...")
    for b_id, g_id in true_pairs:
        if b_id in bank_map and g_id in gateway_map:
            features = compute_features(bank_map[b_id], gateway_map[g_id])
            X.append(features)
            y.append(1)
            
    # Process negative examples (Hard Negatives and Random Negatives)
    print("Synthesizing negative match samples...")
    np.random.seed(42)
    bank_ids = list(bank_map.keys())
    gate_ids = list(gateway_map.keys())
    
    true_set = set(true_pairs)
    
    for b_id in bank_ids:
        # Sample hard negatives: gate records that share close date or amount but aren't the correct match
        b_row = bank_map[b_id]
        
        # Select 2 random ones
        for _ in range(3):
            random_g_id = np.random.choice(gate_ids)
            if (b_id, random_g_id) not in true_set:
                features = compute_features(b_row, gateway_map[random_g_id])
                X.append(features)
                y.append(0)
                
    X = np.array(X)
    y = np.array(y)
    
    # Train Logistic Regression
    clf = LogisticRegression(class_weight='balanced', random_state=42)
    clf.fit(X, y)
    
    # Evaluate
    y_pred = clf.predict(X)
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    
    print(f"Model Training Accuracy: {acc * 100:.2f}%")
    print(f"Model Training Precision: {prec * 100:.2f}%")
    print(f"Model Training Recall: {rec * 100:.2f}%")
    print("Learned Feature Coefficients:")
    feature_names = ["amount_diff_pct", "date_diff_days", "reference_similarity",
                     "amount_within_fee_range", "ref_exact_match"]
    for name, coef in zip(feature_names, clf.coef_[0]):
        print(f"  - {name}: {coef:.4f}")
    print(f"  - Intercept: {clf.intercept_[0]:.4f}")
    
    # Save model
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(model_dir, "match_classifier.pkl"))
    print(f"Saved trained classifier to {os.path.join(model_dir, 'match_classifier.pkl')}")

def run_ml_matcher(unmatched_bank, unmatched_gateway, model_path="models/match_classifier.pkl", threshold=0.7):
    """
    Live matching inference. Predicts matches using the saved Logistic Regression model.
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
    
    # Feature names map
    coefs = clf.coef_[0] # coefficients
    
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
            feature_names = ["amount_diff_pct", "date_diff_days", "reference_similarity",
                             "amount_within_fee_range", "ref_exact_match"]
            contributions = []
            for name, coef, val in zip(feature_names, coefs, best_features):
                contributions.append((name, coef * val, val))

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
