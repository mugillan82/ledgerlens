import os
import sys
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json
import re

# Ensure root directory is on the path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_generator import generate_synthetic_data
from src.exact_matcher import run_exact_matcher
from src.ml_matcher import train_ml_classifier, run_ml_matcher
from src.exception_handler import run_exception_handler
from src.reporter import generate_report

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

# Set page config
st.set_page_config(
    page_title="LedgerLens - ML Reconciliation Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished look
st.markdown("""
<style>
    .metric-card {
        background-color: var(--secondary-background-color);
        border-radius: 10px;
        padding: 15px;
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
    .metric-card h2 {
        color: var(--text-color);
        margin: 0;
        padding-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- Main App Layout -----------------
st.title("🔍 LedgerLens — ML Reconciliation Agent")
st.markdown("Automated matching, explainable predictions, business impact analytics, and audit validation.")

# Sidebar Controls
st.sidebar.header("📁 Data Source Selection")
mode = st.sidebar.radio("Mode", ["Use Generated Synthetic Data", "Upload Custom Files"])

raw_dir = "data/raw"
processed_dir = "data/processed"

bank_df = None
gateway_df = None
custom_run = False

if mode == "Upload Custom Files":
    uploaded_bank = st.sidebar.file_uploader("Upload Bank Statement (CSV)", type="csv")
    uploaded_gateway = st.sidebar.file_uploader("Upload Gateway Settlement (CSV)", type="csv")
    
    processing_count = 0
    if uploaded_bank and uploaded_gateway:
        bank_df = pd.read_csv(uploaded_bank)
        gateway_df = pd.read_csv(uploaded_gateway)
        custom_run = True
        
        # Schema Detection & Mapping for Bank
        if all(col in bank_df.columns for col in ["Date", "Amount", "Reference", "Description"]):
            bank_df = bank_df.rename(columns={
                "Date": "date",
                "Amount": "amount",
                "Reference": "raw_reference",
                "Description": "description"
            })
            
            # Extract setl_ ID using regex
            def extract_setl(ref):
                match = re.search(r"(setl_[a-zA-Z0-9]+)", str(ref), re.IGNORECASE)
                return match.group(1) if match else str(ref)
                
            bank_df["reference_note"] = bank_df["raw_reference"].apply(extract_setl)
            
            # Ensure txn_id exists for matching output
            if "txn_id" not in bank_df.columns:
                bank_df["txn_id"] = "TXN-UP-" + bank_df.index.astype(str)
                
        # Schema Detection & Mapping for Gateway
        expected_gateway_cols = ["order_id", "payment_id", "settlement_id", "gross_amount", "razorpay_fee", "net_amount", "settlement_date", "status"]
        if all(col in gateway_df.columns for col in expected_gateway_cols):
            # Exclude processing settlements from the matching pool entirely
            processing_mask = gateway_df["status"].str.lower() == "processing"
            processing_count = processing_mask.sum()
            gateway_df = gateway_df[~processing_mask].copy()
            
            # Map columns to match internal matcher expectations
            gateway_df = gateway_df.rename(columns={
                "net_amount": "amount",
                "settlement_id": "reference_note",
            })
            
        st.sidebar.success("Files loaded successfully!")
    else:
        st.sidebar.info("Please upload both CSV files to start matching.")
else:
    # Use synthetic data
    bank_path = os.path.join(raw_dir, "bank_statement.csv")
    gateway_path = os.path.join(raw_dir, "gateway_settlement.csv")
    
    if not (os.path.exists(bank_path) and os.path.exists(gateway_path)):
        with st.spinner("Generating initial synthetic financial dataset..."):
            os.makedirs(raw_dir, exist_ok=True)
            generate_synthetic_data(raw_dir)
            
    bank_df = pd.read_csv(bank_path)
    gateway_df = pd.read_csv(gateway_path)

if bank_df is not None and gateway_df is not None:
    # Run matching
    with st.spinner("Reconciling transactions via ML model..."):
        # Run matches
        exact_matches, unmatched_bank, unmatched_gateway = run_exact_matcher(bank_df, gateway_df)
        
        # ML matching (fallback if model doesn't exist yet on disk)
        model_dir = "models"
        model_path = os.path.join(model_dir, "match_classifier.pkl")
        if not os.path.exists(model_path):
            with st.spinner("Training initial ML match classifier..."):
                train_ml_classifier(raw_dir, model_dir)
            
        ml_matches, still_unmatched_bank, still_unmatched_gateway = run_ml_matcher(
            unmatched_bank, unmatched_gateway, model_path=model_path, threshold=0.7
        )
        
        # Populate explanations for exact matches
        for m in exact_matches:
            m["explanation"] = "Exact 1:1 match on date, amount, and reference code"
            
        all_matches = exact_matches + ml_matches
        df_matches = pd.DataFrame(all_matches)
        
        # Exception handler
        df_exceptions = run_exception_handler(
            still_unmatched_bank, still_unmatched_gateway, bank_df, gateway_df, processed_dir
        )
        
        # Generate summary numbers
        total_bank = len(bank_df)
        total_gateway = len(gateway_df)
        exact_count = len(exact_matches)
        ml_count = len(ml_matches)
        matched_count = exact_count + ml_count
        bank_exc_count = len(still_unmatched_bank)
        gate_exc_count = len(still_unmatched_gateway)
        
        # Lay out core counts as cards
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown(f'<div class="metric-card"><h5 style="color:#888;">Total Bank</h5><h2>{total_bank}</h2></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><h5 style="color:#4CAF50;">Exact Matches</h5><h2>{exact_count}</h2></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><h5 style="color:#2196F3;">ML Matches</h5><h2>{ml_count}</h2></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-card"><h5 style="color:#F44336;">Bank Exceptions</h5><h2>{bank_exc_count}</h2></div>', unsafe_allow_html=True)
        with col5:
            st.markdown(f'<div class="metric-card"><h5 style="color:#E91E63;">Gateway Exceptions</h5><h2>{gate_exc_count}</h2></div>', unsafe_allow_html=True)
            
        st.markdown("---")
        
        if custom_run and 'processing_count' in locals() and processing_count > 0:
            st.info(f"ℹ️ **{processing_count} settlements** are still 'processing' and have been excluded from matching, as they are not yet expected in the bank statement.")
            
        # Layout Tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Analytics Dashboard", 
            "📈 Business Impact", 
            "🔍 Match List", 
            "⚠️ Exceptions", 
            "🎯 Ground Truth Accuracy",
            "🚨 Model Errors"
        ])
        
        with tab1:
            st.subheader("Match Rate Breakdown")
            c_exact_pct = (exact_count / total_bank * 100) if total_bank > 0 else 0
            c_ml_pct = (ml_count / total_bank * 100) if total_bank > 0 else 0
            c_exc_pct = (bank_exc_count / total_bank * 100) if total_bank > 0 else 0
            
            fig, ax = plt.subplots(figsize=(7, 3.5))
            fig.patch.set_facecolor('#0e1117')
            ax.set_facecolor('#1e222b')
            
            categories = ['Exact Match', 'ML Match', 'Exceptions']
            shares = [c_exact_pct, c_ml_pct, c_exc_pct]
            colors = ['#4CAF50', '#2196F3', '#F44336']
            
            bars = ax.barh(categories, shares, color=colors, height=0.5, edgecolor='#2d3139')
            ax.set_xlabel('Percentage (%)', color='white')
            ax.tick_params(colors='white')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#2d3139')
            ax.spines['bottom'].set_color('#2d3139')
            ax.set_title("Bank Reconciliation Distribution", color='white', pad=10)
            
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
                        va='center', ha='left', color='white', fontweight='bold')
            
            st.pyplot(fig)
            
        with tab2:
            st.subheader("Business Impact & Time Saved")
            
            # Calculations
            total_value_processed = float(bank_df["amount"].sum())
            matched_bank_ids = {m["bank_txn_id"] for m in all_matches}
            total_value_reconciled = float(bank_df[bank_df["txn_id"].isin(matched_bank_ids)]["amount"].sum())
            reconciled_pct = (total_value_reconciled / total_value_processed * 100) if total_value_processed > 0 else 0.0
            total_value_exceptions = float(still_unmatched_bank["amount"].sum())
            hours_saved = (matched_count * 3) / 60.0
            
            # Metrics cards
            col_b1, col_b2, col_b3, col_b4 = st.columns(4)
            with col_b1:
                st.markdown(f'<div class="metric-card"><h5 style="color:#888;">Total Processed</h5><h2>{format_inr(total_value_processed)}</h2></div>', unsafe_allow_html=True)
            with col_b2:
                st.markdown(f'<div class="metric-card"><h5 style="color:#4CAF50;">Auto-Reconciled</h5><h2>{format_inr(total_value_reconciled)}</h2><p style="color:#4CAF50;font-weight:bold;margin:0;">{reconciled_pct:.2f}% of total</p></div>', unsafe_allow_html=True)
            with col_b3:
                st.markdown(f'<div class="metric-card"><h5 style="color:#F44336;">Flagged exceptions</h5><h2>{format_inr(total_value_exceptions)}</h2></div>', unsafe_allow_html=True)
            with col_b4:
                st.markdown(f'<div class="metric-card"><h5 style="color:#FFC107;">Audit Hours Saved</h5><h2>{hours_saved:.2f} hrs</h2><p style="color:#888;margin:0;">at 3 min/record</p></div>', unsafe_allow_html=True)
                
            # Quick summary text
            st.markdown("### Summary Statement")
            summary_statement = (
                f"The automated reconciliation engine processed **{format_inr(total_value_processed)}** of transactional value. "
                f"By automatically matching **{matched_count}** out of **{total_bank}** records, it has auto-cleared "
                f"**{format_inr(total_value_reconciled)}** without human intervention, saving **{hours_saved:.2f} hours** of labor time. "
                f"Only **{format_inr(total_value_exceptions)}** ({bank_exc_count} transactions) requires physical validation."
            )
            st.markdown(summary_statement)
            
        with tab3:
            st.subheader("Match List (with AI Explanations)")
            if not df_matches.empty:
                # ── Ground-truth cross-check (evaluation mode only) ───────────────
                gt_path_tab3 = os.path.join(raw_dir, "ground_truth.csv")
                if not custom_run and os.path.exists(gt_path_tab3):
                    df_gt_tab3 = pd.read_csv(gt_path_tab3)
                    gt_map = {
                        row["bank_txn_id"]: row["gateway_order_id"]
                        for _, row in df_gt_tab3.iterrows()
                        if row["bank_txn_id"] != "NO_MATCH"
                    }

                    def _gt_badge(row):
                        b_id = row["bank_txn_id"]
                        p_id = row["gateway_order_id"]
                        true_g = gt_map.get(b_id, "NO_MATCH")
                        if true_g == "NO_MATCH":
                            return "⚠️ Incorrect — GT says NO_MATCH"
                        elif true_g == p_id:
                            return "✅ Verified"
                        else:
                            return f"⚠️ Wrong pair — GT expects {true_g}"

                    df_matches = df_matches.copy()
                    df_matches["gt_status"] = df_matches.apply(_gt_badge, axis=1)
                    gt_cross_available = True
                else:
                    gt_cross_available = False

                # ── Build fee_display: unified column that's never None ────────
                df_matches = df_matches.copy()
                if "fee_pct" not in df_matches.columns:
                    df_matches["fee_pct"] = 0.0
                if "fee" not in df_matches.columns:
                    df_matches["fee"] = 0.0
                df_matches["fee_pct"] = df_matches["fee_pct"].fillna(0.0)
                df_matches["fee"] = df_matches["fee"].fillna(0.0)
                df_matches["fee_display"] = df_matches.apply(
                    lambda r: f"{r['fee_pct']:.2f}% (₹{abs(r['fee']):,.2f})",
                    axis=1
                )

                # ── Build display columns ─────────────────────────────────────
                display_cols = ["bank_txn_id", "gateway_order_id", "match_type",
                                "confidence", "fee_display", "explanation"]
                if gt_cross_available:
                    display_cols = ["gt_status"] + display_cols

                if gt_cross_available:
                    bad_mask = df_matches["gt_status"].str.startswith("⚠️")
                    bad_count = bad_mask.sum()
                    good_count = (~bad_mask).sum()
                    st.markdown(
                        f"🟢 **{good_count} verified correct** &nbsp;|&nbsp; "
                        f"🔴 **{bad_count} flagged incorrect** (cross-checked against ground truth)",
                        unsafe_allow_html=True
                    )

                st.dataframe(
                    df_matches[display_cols].rename(columns={
                        "gt_status": "GT Verification",
                        "fee_display": "Gateway Fee"
                    }),
                    use_container_width=True
                )

                if gt_cross_available and bad_count > 0:
                    st.warning(
                        f"⚠️ {bad_count} match(es) above are flagged by ground truth as incorrect. "
                        "See the **🚨 Model Errors** tab for full details."
                    )
            else:
                st.info("No successful matches generated.")

                
        with tab4:
            st.subheader("Detailed Exception List")
            st.markdown("These entries could not be matched automatically. Audit reasons are assigned below:")
            if not df_exceptions.empty:
                st.dataframe(df_exceptions, use_container_width=True)
            else:
                st.success("Zero exceptions found! Outstanding balance fully reconciled.")
                
        with tab5:
            st.subheader("Ground Truth Validation (Evaluation Mode)")
            gt_path = os.path.join(raw_dir, "ground_truth.csv")
            if not custom_run and os.path.exists(gt_path):
                # Calculate GT metrics
                df_gt = pd.read_csv(gt_path)
                
                gt_bank_to_gate = {}
                for _, row in df_gt.iterrows():
                    b_id = row["bank_txn_id"]
                    g_id = row["gateway_order_id"]
                    if b_id != "NO_MATCH":
                        gt_bank_to_gate[b_id] = g_id
                
                pipe_bank_to_gate = {}
                for m in all_matches:
                    pipe_bank_to_gate[m["bank_txn_id"]] = m["gateway_order_id"]
                    
                tp = 0
                fp = 0
                fn = 0
                tn = 0
                correct = 0
                total = len(gt_bank_to_gate)
                
                for b_id, true_g_id in gt_bank_to_gate.items():
                    pipe_g_id = pipe_bank_to_gate.get(b_id, "NO_MATCH")
                    if true_g_id == pipe_g_id:
                        correct += 1
                        if true_g_id != "NO_MATCH":
                            tp += 1
                        else:
                            tn += 1
                    else:
                        if true_g_id != "NO_MATCH" and pipe_g_id != "NO_MATCH":
                            fp += 1
                            fn += 1
                        elif true_g_id == "NO_MATCH" and pipe_g_id != "NO_MATCH":
                            fp += 1
                        elif true_g_id != "NO_MATCH" and pipe_g_id == "NO_MATCH":
                            fn += 1
                            
                accuracy = (correct / total * 100) if total > 0 else 0
                precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0
                recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0
                
                col_g1, col_g2, col_g3 = st.columns(3)
                with col_g1:
                    st.metric("Model Match Accuracy", f"{accuracy:.2f}%")
                with col_g2:
                    st.metric("Model Precision", f"{precision:.2f}%")
                with col_g3:
                    st.metric("Model Recall", f"{recall:.2f}%")
                    
                st.markdown("### Confusion Matrix")
                cm_data = {
                    "Classification": ["True Positives (Correct Matches)", "True Negatives (Correct Exceptions)", "False Positives (Incorrect Matches)", "False Negatives (Missed Matches)"],
                    "Count": [tp, tn, fp, fn]
                }
                st.table(pd.DataFrame(cm_data))
                
                st.info("Evaluation details compare matched pairs against hidden true transaction records.")
            else:
                st.warning("Ground Truth validation is only available when running on generated synthetic data containing 'ground_truth.csv'.")

        with tab6:
            st.subheader("🚨 Model Error Analysis")
            st.markdown(
                "Detailed breakdown of **False Positives** (wrong matches) and **False Negatives** (missed matches) "
                "from the ground truth evaluation. Each error includes the records involved, feature values that "
                "likely caused the error, and a plain-English explanation of the failure mode."
            )

            errors_path = os.path.join(processed_dir, "model_errors.json")

            if custom_run:
                st.warning("Model error analysis is only available for generated synthetic data (requires ground_truth.csv).")
            else:
                gt_path_raw = os.path.join(raw_dir, "ground_truth.csv")
                if not os.path.exists(errors_path) and os.path.exists(gt_path_raw):
                    generate_report(
                        all_matches, still_unmatched_bank, still_unmatched_gateway,
                        bank_df, gateway_df, raw_dir, processed_dir
                    )
                    
                if not os.path.exists(errors_path):
                    st.info("No model_errors.json found. Ground truth comparison is not available.")
                else:
                    with open(errors_path, "r", encoding="utf-8") as f:
                        errors = json.load(f)

                fp_errors = [e for e in errors if e["error_type"] == "FALSE_POSITIVE"]
                fn_errors = [e for e in errors if e["error_type"] == "FALSE_NEGATIVE"]

                col_e1, col_e2 = st.columns(2)
                with col_e1:
                    st.markdown(f'<div class="metric-card"><h5 style="color:#FF5722;">False Positives</h5><h2>{len(fp_errors)}</h2><p style="color:#888;margin:0;">Wrong matches</p></div>', unsafe_allow_html=True)
                with col_e2:
                    st.markdown(f'<div class="metric-card"><h5 style="color:#FF9800;">False Negatives</h5><h2>{len(fn_errors)}</h2><p style="color:#888;margin:0;">Missed matches</p></div>', unsafe_allow_html=True)

                st.markdown("---")

                if fp_errors:
                    st.markdown("### 🔴 False Positives — Incorrectly Matched Records")
                    st.caption("The pipeline matched these bank transactions to the wrong gateway entry. Ground truth says they should be unmatched (NO_MATCH).")
                    for i, err in enumerate(fp_errors, 1):
                        feats = err.get("features", {})
                        with st.expander(f"FP #{i}: Bank `{err['bank_txn_id']}` → wrongly matched to `{err['pipeline_gateway_id']}`", expanded=(i == 1)):
                            c1, c2 = st.columns(2)
                            with c1:
                                st.markdown("**🏦 Bank Record**")
                                st.table(pd.DataFrame({
                                    "Field": ["Transaction ID", "Date", "Amount", "Reference"],
                                    "Value": [
                                        err["bank_txn_id"],
                                        err["bank_date"],
                                        format_inr(float(err["bank_amount"])) if err["bank_amount"] != "" else "—",
                                        err["bank_reference"]
                                    ]
                                }))
                            with c2:
                                st.markdown("**🏧 Wrong Gateway Match (Pipeline Prediction)**")
                                st.table(pd.DataFrame({
                                    "Field": ["Order ID", "Date", "Amount", "Reference"],
                                    "Value": [
                                        err["pipeline_gateway_id"],
                                        err["pipeline_gateway_date"],
                                        format_inr(float(err["pipeline_gateway_amount"])) if err["pipeline_gateway_amount"] != "" else "—",
                                        err["pipeline_gateway_reference"]
                                    ]
                                }))
                            st.markdown(f"**✅ Correct Ground Truth Answer:** `{err['correct_gateway_id']}`")
                            st.markdown("**📊 Feature Values at Time of Prediction:**")
                            feat_df = pd.DataFrame([{
                                "Feature": "amount_diff_pct (%)",
                                "Value": f"{feats.get('amount_diff_pct', '—')}",
                            }, {
                                "Feature": "date_diff_days",
                                "Value": f"{feats.get('date_diff_days', '—')}",
                            }, {
                                "Feature": "reference_similarity",
                                "Value": f"{feats.get('reference_similarity', '—')}",
                            }, {
                                "Feature": "amount_within_fee_range",
                                "Value": str(feats.get('amount_within_fee_range', '—')),
                            }])
                            st.table(feat_df)
                            st.error(f"💡 **Failure Reason:** {err['failure_explanation']}")

                if fn_errors:
                    st.markdown("### 🟠 False Negatives — Missed Matches")
                    st.caption("The pipeline left these bank transactions unmatched. Ground truth says they should have been matched to a gateway record.")
                    for i, err in enumerate(fn_errors, 1):
                        feats = err.get("features", {})
                        with st.expander(f"FN #{i}: Bank `{err['bank_txn_id']}` → should match `{err['correct_gateway_id']}`", expanded=(i == 1)):
                            c1, c2 = st.columns(2)
                            with c1:
                                st.markdown("**🏦 Bank Record**")
                                st.table(pd.DataFrame({
                                    "Field": ["Transaction ID", "Date", "Amount", "Reference"],
                                    "Value": [
                                        err["bank_txn_id"],
                                        err["bank_date"],
                                        format_inr(float(err["bank_amount"])) if err["bank_amount"] != "" else "—",
                                        err["bank_reference"]
                                    ]
                                }))
                            with c2:
                                st.markdown("**🏧 Correct Gateway Record (Ground Truth)**")
                                st.table(pd.DataFrame({
                                    "Field": ["Order ID", "Date", "Amount", "Reference"],
                                    "Value": [
                                        err["correct_gateway_id"],
                                        err.get("correct_gateway_date", "—"),
                                        format_inr(float(err["correct_gateway_amount"])) if err.get("correct_gateway_amount", "") != "" else "—",
                                        err.get("correct_gateway_reference", "—")
                                    ]
                                }))
                            st.markdown("**📊 Feature Values (Bank vs Correct Gateway):**")
                            feat_df = pd.DataFrame([{
                                "Feature": "amount_diff_pct (%)",
                                "Value": f"{feats.get('amount_diff_pct', '—')}",
                            }, {
                                "Feature": "date_diff_days",
                                "Value": f"{feats.get('date_diff_days', '—')}",
                            }, {
                                "Feature": "reference_similarity",
                                "Value": f"{feats.get('reference_similarity', '—')}",
                            }, {
                                "Feature": "amount_within_fee_range",
                                "Value": str(feats.get('amount_within_fee_range', '—')),
                            }])
                            st.table(feat_df)
                            st.warning(f"💡 **Failure Reason:** {err['failure_explanation']}")

                if not fp_errors and not fn_errors:
                    st.success("🎉 No model errors detected! The pipeline predicted all matches and exceptions perfectly.")
else:
    st.info("Upload custom files via the sidebar or run the generator to populate workspace data.")
