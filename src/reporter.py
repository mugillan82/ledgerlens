import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from rapidfuzz import fuzz

def format_inr(val):
    s = f"{abs(val):.2f}"
    parts = s.split('.')
    integer = parts[0]
    fraction = parts[1]
    
    if len(integer) <= 3:
        res = integer
    else:
        last_three = integer[-3:]
        remaining = integer[:-3]
        out = []
        while len(remaining) > 0:
            out.insert(0, remaining[-2:])
            remaining = remaining[:-2]
        res = ",".join(out) + "," + last_three
        
    prefix = "-" if val < 0 else ""
    return f"{prefix}₹{res}.{fraction}"

def generate_report(matched_results, unmatched_bank, unmatched_gateway, original_bank, original_gateway, raw_dir="data/raw", processed_dir="data/processed"):
    """
    Generates a markdown summary report, a matplotlib bar chart, and a JSON metrics file.
    Validates pipeline predictions against the ground truth.
    Includes business impact reporting.
    """
    os.makedirs(processed_dir, exist_ok=True)
    
    total_bank = len(original_bank)
    total_gateway = len(original_gateway)
    
    # Calculate counts
    exact_matches = [m for m in matched_results if m["match_type"] == "EXACT"]
    ml_matches = [m for m in matched_results if m["match_type"] == "ML_MATCH"]
    
    exact_count = len(exact_matches)
    ml_count = len(ml_matches)
    matched_count = exact_count + ml_count
    
    avg_ml_conf = (
        sum(m["confidence"] for m in ml_matches) / ml_count if ml_count > 0 else 0.0
    )
    
    bank_exc_count = len(unmatched_bank)
    gate_exc_count = len(unmatched_gateway)
    
    # Overall match rates
    match_rate_bank = (matched_count / total_bank * 100) if total_bank > 0 else 0.0
    
    # --- BUSINESS IMPACT CALCULATIONS ---
    total_value_processed = float(original_bank["amount"].sum())
    
    matched_bank_ids = {m["bank_txn_id"] for m in matched_results}
    total_value_reconciled = float(original_bank[original_bank["txn_id"].isin(matched_bank_ids)]["amount"].sum())
    reconciled_pct = (total_value_reconciled / total_value_processed * 100) if total_value_processed > 0 else 0.0
    
    total_value_exceptions = float(unmatched_bank["amount"].sum())
    
    # Estimated time saved: 3 minutes per matched record
    hours_saved = (matched_count * 3) / 60.0
    
    # Save Business Metrics to JSON for the Streamlit dashboard to consume
    business_metrics = {
        "total_value_processed": total_value_processed,
        "total_value_reconciled": total_value_reconciled,
        "reconciled_pct": reconciled_pct,
        "total_value_exceptions": total_value_exceptions,
        "hours_saved": hours_saved,
        "exact_count": exact_count,
        "ml_count": ml_count,
        "bank_exc_count": bank_exc_count,
        "gate_exc_count": gate_exc_count,
        "matched_count": matched_count
    }
    
    with open(os.path.join(processed_dir, "business_metrics.json"), "w") as f:
        json.dump(business_metrics, f, indent=4)
        
    # --- VALIDATION AGAINST GROUND TRUTH ---
    gt_path = os.path.join(raw_dir, "ground_truth.csv")
    evaluation_available = os.path.exists(gt_path)
    
    accuracy = 0.0
    precision = 0.0
    recall = 0.0
    false_matches = []
    
    # Define confusion matrix elements
    tp = 0 # True Positives
    fp = 0 # False Positives
    fn = 0 # False Negatives
    tn = 0 # True Negatives
    
    if evaluation_available:
        df_gt = pd.read_csv(gt_path)
        
        # Build mapping dictionaries
        # True mappings: bank_txn_id -> gateway_order_id
        gt_bank_to_gate = {}
        for _, row in df_gt.iterrows():
            b_id = row["bank_txn_id"]
            g_id = row["gateway_order_id"]
            if b_id != "NO_MATCH":
                gt_bank_to_gate[b_id] = g_id
                
        # Pipeline mappings: bank_txn_id -> gateway_order_id
        pipe_bank_to_gate = {}
        for m in matched_results:
            pipe_bank_to_gate[m["bank_txn_id"]] = m["gateway_order_id"]
            
        correct_predictions = 0
        total_evaluations = 0
        
        for b_id, true_g_id in gt_bank_to_gate.items():
            pipe_g_id = pipe_bank_to_gate.get(b_id, "NO_MATCH")
            total_evaluations += 1
            
            if true_g_id == pipe_g_id:
                correct_predictions += 1
                if true_g_id != "NO_MATCH":
                    tp += 1
                else:
                    tn += 1
            else:
                if true_g_id != "NO_MATCH" and pipe_g_id != "NO_MATCH":
                    false_matches.append({
                        "bank_txn_id": b_id,
                        "pipeline_match": pipe_g_id,
                        "ground_truth_match": true_g_id,
                        "reason": f"Mismatched (GT: {true_g_id}, Pipe: {pipe_g_id})"
                    })
                    fp += 1
                    fn += 1
                elif true_g_id == "NO_MATCH" and pipe_g_id != "NO_MATCH":
                    false_matches.append({
                        "bank_txn_id": b_id,
                        "pipeline_match": pipe_g_id,
                        "ground_truth_match": "NO_MATCH",
                        "reason": "Incorrectly matched unmatched bank entry"
                    })
                    fp += 1
                elif true_g_id != "NO_MATCH" and pipe_g_id == "NO_MATCH":
                    fn += 1
                    
        accuracy = (correct_predictions / total_evaluations * 100) if total_evaluations > 0 else 0.0
        precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0

        # --- DETAILED ERROR ANALYSIS ---
        # Build lookup maps for record details
        bank_map = original_bank.set_index("txn_id").to_dict(orient="index")
        gateway_map = original_gateway.set_index("order_id").to_dict(orient="index")

        def _compute_features(b_row, g_row):
            from src.exact_matcher import normalize_ref
            b_amt = float(b_row.get("amount", 0))
            g_amt = float(g_row.get("amount", 0))
            b_date = pd.to_datetime(b_row.get("date", ""))
            g_date = pd.to_datetime(g_row.get("settlement_date", ""))
            b_ref = normalize_ref(str(b_row.get("reference_note", "")))
            g_ref = normalize_ref(str(g_row.get("reference_note", "")))
            amt_diff_pct = round(abs(b_amt - g_amt) / b_amt * 100, 4) if b_amt > 0 else 0.0
            date_diff = abs((g_date - b_date).days)
            ref_sim = round(fuzz.ratio(b_ref, g_ref), 2)
            fee_pct = (b_amt - g_amt) / b_amt if b_amt > 0 else -1.0
            within_fee = 1.0 if 0.01 <= fee_pct <= 0.0305 else 0.0
            ref_exact = 1 if b_ref == g_ref else 0
            return {
                "amount_diff_pct": amt_diff_pct,
                "date_diff_days": int(date_diff),
                "reference_similarity": ref_sim,
                "amount_within_fee_range": bool(within_fee),
                "ref_exact_match": bool(ref_exact)
            }

        def _fp_explanation(b_row, g_row, features):
            """Generate plain-English explanation for a False Positive."""
            ref_sim = features["reference_similarity"]
            date_diff = features["date_diff_days"]
            amt_diff = features["amount_diff_pct"]
            if ref_sim > 80:
                return (f"Near-duplicate confusion: high reference similarity ({ref_sim:.0f}%) "
                        f"caused model to match to a wrong gateway entry (amount diff {amt_diff:.2f}%)")
            elif date_diff <= 3 and amt_diff < 3:
                return (f"Likely matched due to close date ({date_diff}d gap) and similar amount, "
                        f"but reference strings differ — model over-trusted numeric features")
            else:
                return (f"Model incorrectly matched despite {amt_diff:.2f}% amount gap and {date_diff}-day "
                        f"date gap — low-confidence match was accepted above threshold")

        def _fn_explanation(b_row, g_row, features):
            """Generate plain-English explanation for a False Negative."""
            ref_sim = features["reference_similarity"]
            date_diff = features["date_diff_days"]
            amt_diff = features["amount_diff_pct"]
            if date_diff > 3:
                return (f"Missed due to date gap ({date_diff} days) exceeding the 3-day settlement "
                        f"window — model correctly penalises but missed the true match")
            elif ref_sim < 70:
                return (f"Reference similarity too low ({ref_sim:.0f}%) — possible typo or format "
                        f"mismatch prevented the model from connecting the correct pair")
            elif amt_diff > 3:
                return (f"Amount difference ({amt_diff:.2f}%) beyond typical 1-3% fee range — "
                        f"model rejected what was actually a valid match with unusually high fee")
            else:
                return (f"True match was outcompeted by a higher-scoring but incorrect candidate "
                        f"(ref sim {ref_sim:.0f}%, amt diff {amt_diff:.2f}%, date gap {date_diff}d)")

        detailed_errors = []

        # False Positives: pipeline matched, GT says something different
        for fm in false_matches:
            b_id = fm["bank_txn_id"]
            pipe_g_id = fm["pipeline_match"]
            true_g_id = fm["ground_truth_match"]

            b_row = bank_map.get(b_id, {})
            pipe_g_row = gateway_map.get(pipe_g_id, {}) if pipe_g_id != "NO_MATCH" else {}

            features = _compute_features(b_row, pipe_g_row) if pipe_g_row else {}
            explanation = _fp_explanation(b_row, pipe_g_row, features) if pipe_g_row else "No gateway record for wrong match"

            detailed_errors.append({
                "error_type": "FALSE_POSITIVE",
                "bank_txn_id": b_id,
                "bank_date": b_row.get("date", ""),
                "bank_amount": b_row.get("amount", ""),
                "bank_reference": b_row.get("reference_note", ""),
                "pipeline_gateway_id": pipe_g_id,
                "pipeline_gateway_date": pipe_g_row.get("settlement_date", ""),
                "pipeline_gateway_amount": pipe_g_row.get("amount", ""),
                "pipeline_gateway_reference": pipe_g_row.get("reference_note", ""),
                "correct_gateway_id": true_g_id,
                "features": features,
                "failure_explanation": explanation
            })

        # False Negatives: pipeline left unmatched, but GT says there should be a match
        for b_id, true_g_id in gt_bank_to_gate.items():
            pipe_g_id = pipe_bank_to_gate.get(b_id, "NO_MATCH")
            if true_g_id != "NO_MATCH" and pipe_g_id == "NO_MATCH":
                b_row = bank_map.get(b_id, {})
                true_g_row = gateway_map.get(true_g_id, {})
                features = _compute_features(b_row, true_g_row) if true_g_row else {}
                explanation = _fn_explanation(b_row, true_g_row, features) if true_g_row else "No gateway record found for missed match"

                detailed_errors.append({
                    "error_type": "FALSE_NEGATIVE",
                    "bank_txn_id": b_id,
                    "bank_date": b_row.get("date", ""),
                    "bank_amount": b_row.get("amount", ""),
                    "bank_reference": b_row.get("reference_note", ""),
                    "pipeline_gateway_id": "NOT MATCHED",
                    "pipeline_gateway_date": "",
                    "pipeline_gateway_amount": "",
                    "pipeline_gateway_reference": "",
                    "correct_gateway_id": true_g_id,
                    "correct_gateway_date": true_g_row.get("settlement_date", ""),
                    "correct_gateway_amount": true_g_row.get("amount", ""),
                    "correct_gateway_reference": true_g_row.get("reference_note", ""),
                    "features": features,
                    "failure_explanation": explanation
                })

        with open(os.path.join(processed_dir, "model_errors.json"), "w", encoding="utf-8") as f:
            json.dump(detailed_errors, f, indent=4)

    # Save Ground Truth confusion matrix details for app to load
    gt_metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "correct_predictions": correct_predictions,
        "total_evaluations": total_evaluations
    }
    with open(os.path.join(processed_dir, "gt_metrics.json"), "w") as f:
        json.dump(gt_metrics, f, indent=4)

    # Build report text
    report_text = f"""# AI Finance Reconciliation Report
Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}

## Pipeline Summary
- **Total Bank Transactions**: {total_bank}
- **Total Gateway Settlement Records**: {total_gateway}

- **Exact Matches**: {exact_count} ({exact_count / total_bank * 100:.2f}%)
- **ML Matches**: {ml_count} ({ml_count / total_bank * 100:.2f}%) (Avg Confidence: {avg_ml_conf:.2f}%)
- **Unmatched Bank Transactions (Exceptions)**: {bank_exc_count} ({bank_exc_count / total_bank * 100:.2f}%)
- **Unmatched Gateway Transactions (Exceptions)**: {gate_exc_count} ({gate_exc_count / total_gateway * 100:.2f}%)

- **Overall Match Rate (Bank)**: {match_rate_bank:.2f}%

## Business Impact Report
- **Total Transaction Value Processed**: {format_inr(total_value_processed)}
- **Total Value Auto-Reconciled**: {format_inr(total_value_reconciled)} ({reconciled_pct:.2f}% of total value)
- **Total Value Flagged as Exceptions (Requires Audit)**: {format_inr(total_value_exceptions)}
- **Estimated Audit Time Saved**: {hours_saved:.2f} hours (based on ~3 min per auto-matched record)
"""

    if evaluation_available:
        report_text += f"""
## Validation Against Ground Truth
- **Reconciliation Accuracy**: {accuracy:.2f}%
- **Precision**: {precision:.2f}%
- **Recall**: {recall:.2f}%
- **Confusion Matrix**:
  - True Positives (Correct Matches): {tp}
  - True Negatives (Correct Unmatched): {tn}
  - False Positives (Incorrect Matches): {fp}
  - False Negatives (Missed Matches): {fn}
"""
        if false_matches:
            report_text += "\n### Mismatches / Errors:\n"
            for fm in false_matches:
                report_text += f"- Bank Txn `{fm['bank_txn_id']}`: Pipeline matched to `{fm['pipeline_match']}` but ground truth is `{fm['ground_truth_match']}` ({fm['reason']})\n"
        else:
            report_text += "\nNo matching errors detected! The pipeline matches the ground truth 100% correctly.\n"
            
    # Save Report
    report_path = os.path.join(processed_dir, "reconciliation_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
        
    # Generate and Save Chart
    categories = ['Exact Match', 'ML Match', 'Bank Exceptions', 'Gateway Exceptions']
    counts = [exact_count, ml_count, bank_exc_count, gate_exc_count]
    colors = ['#4CAF50', '#2196F3', '#F44336', '#E91E63']
    
    plt.figure(figsize=(8, 5))
    bars = plt.bar(categories, counts, color=colors, edgecolor='black', width=0.6)
    plt.title('Reconciliation Category Breakdown', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('Count', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 5, f"{int(yval)}", ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    chart_path = os.path.join(processed_dir, "reconciliation_breakdown.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    
    return report_text, chart_path
