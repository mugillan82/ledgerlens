import os
import sys
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
import re

# Ensure root directory is on the path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_generator import generate_synthetic_data
from src.exact_matcher import run_exact_matcher
from src.ml_matcher import train_ml_classifier, run_ml_matcher
from src.model_comparison import compare_and_train_models
from src.audit_export import generate_excel_report, generate_pdf_report
from src.exception_handler import run_exception_handler
from src.reporter import generate_report
from src.razorpay_client import fetch_live_razorpay_data, create_test_orders

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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Metric cards ── */
    .metric-card {
        background: linear-gradient(135deg, #131929 0%, #0d1422 100%);
        border-radius: 12px;
        padding: 18px 20px 16px 20px;
        border: 1px solid rgba(0, 191, 165, 0.15);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.35);
        min-height: 90px;
        margin-bottom: 4px;
    }
    .metric-card h5 {
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin: 0 0 6px 0;
    }
    .metric-card h2 {
        color: #E8EDF5;
        font-size: 1.9rem;
        font-weight: 700;
        margin: 0;
        line-height: 1.1;
    }
    .metric-card p {
        font-size: 0.82rem;
        margin: 4px 0 0 0;
    }

    /* ── Color accents for metric labels ── */
    .mc-neutral  { color: #8A95A8; }
    .mc-teal     { color: #00BFA5; }
    .mc-blue     { color: #4FC3F7; }
    .mc-coral    { color: #FF6B6B; }
    .mc-rose     { color: #F48FB1; }
    .mc-amber    { color: #FFB74D; }
    .mc-orange   { color: #FF7043; }
</style>
""", unsafe_allow_html=True)

# ----------------- Main App Layout -----------------
st.title("🔍 LedgerLens — ML Reconciliation Agent")
st.markdown("Automated matching, explainable predictions, business impact analytics, and audit validation.")

# Sidebar Controls
st.sidebar.header("📁 Data Source Selection")
mode = st.sidebar.radio("Mode", ["Use Generated Synthetic Data", "Upload Custom Files", "Live Razorpay Test API"])

raw_dir = "data/raw"
processed_dir = "data/processed"

bank_df = None
gateway_df = None
custom_run = False
is_live_api = False
api_meta = {}

# Clear stale live-API session state when switching away from that mode,
# so cached live DataFrames never bleed into a synthetic or upload session.
if mode != "Live Razorpay Test API":
    for _key in ["rzp_bank_df", "rzp_gateway_df", "rzp_meta"]:
        st.session_state.pop(_key, None)

if mode == "Live Razorpay Test API":
    is_live_api = True
    st.sidebar.markdown("### ⚡ Live Razorpay Test API")
    st.sidebar.info("Connects to Razorpay using credentials in `.env` or environment secrets.")
    
    seed_new = st.sidebar.checkbox("Generate fresh test Orders on Razorpay", value=True, help="Creates real test orders via Razorpay Orders API")
    order_count = st.sidebar.slider("Number of test orders to generate", min_value=5, max_value=30, value=15) if seed_new else 0
    
    if st.sidebar.button("🚀 Fetch Live Data from Razorpay Test API", type="primary"):
        with st.spinner("Connecting to Razorpay API and fetching live records..."):
            try:
                b_df, g_df, meta = fetch_live_razorpay_data(create_new_sample=seed_new, sample_count=order_count)
                st.session_state["rzp_bank_df"] = b_df
                st.session_state["rzp_gateway_df"] = g_df
                st.session_state["rzp_meta"] = meta
                st.sidebar.success(f"Retrieved {meta['orders_count']} orders from Razorpay API!")
            except Exception as e:
                st.sidebar.error(f"API Error: {str(e)}")
                
    if "rzp_bank_df" in st.session_state and "rzp_gateway_df" in st.session_state:
        bank_df = st.session_state["rzp_bank_df"]
        gateway_df = st.session_state["rzp_gateway_df"]
        api_meta = st.session_state.get("rzp_meta", {})
    else:
        st.info("👈 Click **Fetch Live Data from Razorpay Test API** in the sidebar to retrieve live orders and run reconciliation.")

elif mode == "Upload Custom Files":
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
            with st.spinner("Training initial ML match classifiers..."):
                train_ml_classifier(raw_dir, model_dir)
            
        # Get all candidate ML matches with threshold >= 0.5 for interactive threshold tuning
        all_candidate_ml_matches, _, _ = run_ml_matcher(
            unmatched_bank, unmatched_gateway, model_path=model_path, threshold=0.5
        )
        
        # Default accepted ML matches at standard 70% threshold
        default_ml_threshold = 70.0
        ml_matches = [m for m in all_candidate_ml_matches if m["confidence"] >= default_ml_threshold]
        
        # Populate explanations for exact matches
        for m in exact_matches:
            m["explanation"] = "Exact 1:1 match on date, amount, and reference code"
            
        all_matches = exact_matches + ml_matches
        df_matches = pd.DataFrame(all_matches)
        
        # Determine unmatched sets based on standard matches
        matched_bank_ids = {m["bank_txn_id"] for m in all_matches}
        matched_gateway_ids = {m["gateway_order_id"] for m in all_matches}
        still_unmatched_bank = unmatched_bank[~unmatched_bank["txn_id"].isin(matched_bank_ids)].copy()
        still_unmatched_gateway = unmatched_gateway[~unmatched_gateway["order_id"].isin(matched_gateway_ids)].copy()
        
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
        
        if is_live_api:
            st.warning("""
### ⚡ Live Razorpay Test Mode Transparency Notice
* **✅ Orders are LIVE** — fetched directly from Razorpay's API in real time.
* **⚠️ Settlement & bank-side data is SIMULATED** using Razorpay's real fee structure (2% + GST), since Razorpay's test mode does not support programmatic payment capture or settlement generation without a live checkout session — this is a platform-wide constraint, not a limitation of this reconciliation engine.
""")

        # Lay out core counts as cards
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown(f'<div class="metric-card"><h5 class="mc-neutral">Total Bank</h5><h2>{total_bank}</h2></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><h5 class="mc-teal">Exact Matches</h5><h2>{exact_count}</h2></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><h5 class="mc-blue">ML Matches</h5><h2>{ml_count}</h2></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-card"><h5 class="mc-coral">Bank Exceptions</h5><h2>{bank_exc_count}</h2></div>', unsafe_allow_html=True)
        with col5:
            st.markdown(f'<div class="metric-card"><h5 class="mc-rose">Gateway Exceptions</h5><h2>{gate_exc_count}</h2></div>', unsafe_allow_html=True)
            
        st.markdown("---")
        
        if custom_run and 'processing_count' in locals() and processing_count > 0:
            st.info(f"ℹ️ **{processing_count} settlements** are still 'processing' and have been excluded from matching, as they are not yet expected in the bank statement.")
            
        # Layout Tabs
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "📊 Analytics Dashboard", 
            "📈 Business Impact", 
            "🔍 Match List", 
            "⚠️ Exceptions", 
            "🤖 Model Comparison",
            "🎯 Ground Truth Accuracy",
            "🚨 Model Errors",
            "📥 Download Reports"
        ])
        
        with tab1:
            st.subheader("Match Rate Breakdown")
            c_exact_pct = (exact_count / total_bank * 100) if total_bank > 0 else 0
            c_ml_pct = (ml_count / total_bank * 100) if total_bank > 0 else 0
            c_exc_pct = (bank_exc_count / total_bank * 100) if total_bank > 0 else 0

            categories = ['Exact Match', 'ML Match', 'Exceptions']
            shares = [c_exact_pct, c_ml_pct, c_exc_pct]
            bar_colors = ['#00BFA5', '#4FC3F7', '#FF6B6B']

            fig_tab1 = go.Figure(go.Bar(
                x=shares,
                y=categories,
                orientation='h',
                marker=dict(
                    color=bar_colors,
                    line=dict(color='rgba(0,0,0,0)', width=0)
                ),
                text=[f'{v:.1f}%' for v in shares],
                textposition='outside',
                textfont=dict(color='#E8EDF5', size=13, family='Inter'),
                hovertemplate='<b>%{y}</b><br>Share: %{x:.2f}%<extra></extra>',
            ))
            fig_tab1.update_layout(
                title=dict(text='Bank Reconciliation Distribution', font=dict(color='#E8EDF5', size=15, family='Inter')),
                plot_bgcolor='#131929',
                paper_bgcolor='#0A0F1E',
                font=dict(color='#E8EDF5', family='Inter'),
                xaxis=dict(
                    title='Percentage (%)',
                    color='#8A95A8',
                    gridcolor='rgba(255,255,255,0.06)',
                    range=[0, max(shares + [1]) * 1.2],
                    zeroline=False,
                ),
                yaxis=dict(color='#E8EDF5', gridcolor='rgba(0,0,0,0)'),
                margin=dict(l=20, r=30, t=50, b=30),
                height=260,
            )
            st.plotly_chart(fig_tab1, use_container_width=True)
            
        with tab2:
            st.subheader("Business Impact & Time Saved")
            
            # Calculations
            total_value_processed = float(bank_df["amount"].sum())
            total_value_reconciled = float(bank_df[bank_df["txn_id"].isin(matched_bank_ids)]["amount"].sum())
            reconciled_pct = (total_value_reconciled / total_value_processed * 100) if total_value_processed > 0 else 0.0
            total_value_exceptions = float(still_unmatched_bank["amount"].sum())
            hours_saved = (matched_count * 3) / 60.0
            
            # Metrics cards
            col_b1, col_b2, col_b3, col_b4 = st.columns(4)
            with col_b1:
                st.markdown(f'<div class="metric-card"><h5 class="mc-neutral">Total Processed</h5><h2>{format_inr(total_value_processed)}</h2></div>', unsafe_allow_html=True)
            with col_b2:
                st.markdown(f'<div class="metric-card"><h5 class="mc-teal">Auto-Reconciled</h5><h2>{format_inr(total_value_reconciled)}</h2><p class="mc-teal" style="font-weight:600;">{reconciled_pct:.2f}% of total</p></div>', unsafe_allow_html=True)
            with col_b3:
                st.markdown(f'<div class="metric-card"><h5 class="mc-coral">Flagged Exceptions</h5><h2>{format_inr(total_value_exceptions)}</h2></div>', unsafe_allow_html=True)
            with col_b4:
                st.markdown(f'<div class="metric-card"><h5 class="mc-amber">Audit Hours Saved</h5><h2>{hours_saved:.2f} hrs</h2><p class="mc-neutral">at 3 min/record</p></div>', unsafe_allow_html=True)
                
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
                # ── Ground-truth cross-check (evaluation mode only — synthetic data only) ───
                gt_path_tab3 = os.path.join(raw_dir, "ground_truth.csv")
                if not custom_run and not is_live_api and os.path.exists(gt_path_tab3):
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

        # ==================== TAB 5: MODEL COMPARISON ====================
        with tab5:
            st.subheader("🤖 Machine Learning Model Benchmarking & Comparison")
            st.markdown(
                "To select the most accurate and reliable matching algorithm, LedgerLens trains and benchmarks "
                "three distinct classifiers on a stratified held-out test split using the exact 5 reconciliation features."
            )
            
            comp_csv_path = os.path.join(model_dir, "model_comparison.csv")
            if not os.path.exists(comp_csv_path) and not custom_run:
                with st.spinner("Benchmarking classifiers..."):
                    compare_and_train_models(raw_dir, model_dir)
                    
            if os.path.exists(comp_csv_path):
                df_comp = pd.read_csv(comp_csv_path)
                
                # Identify winning model
                winner_row = df_comp[df_comp["Winner"].str.contains("WINNER|Best", na=False)]
                if winner_row.empty:
                    winner_row = df_comp.sort_values(by=["F1 Score", "Precision"], ascending=False).iloc[[0]]
                winner_name = winner_row.iloc[0]["Model"]
                winner_f1 = winner_row.iloc[0]["F1 Score"]
                winner_prec = winner_row.iloc[0]["Precision"]
                winner_rec = winner_row.iloc[0]["Recall"]
                winner_time = winner_row.iloc[0]["Training Time (ms)"]
                
                # Winner highlight banner
                st.success(
                    f"🏆 **Selected Model for Live Inference:** `{winner_name}` (F1 Score: **{winner_f1 * 100:.2f}%** | "
                    f"Precision: **{winner_prec * 100:.2f}%** | Recall: **{winner_rec * 100:.2f}%** | Training: **{winner_time:.2f} ms**)"
                )
                
                # Format dataframe for presentation
                df_display = df_comp.copy()
                df_display["Accuracy"] = df_display["Accuracy"].apply(lambda v: f"{v * 100:.2f}%")
                df_display["Precision"] = df_display["Precision"].apply(lambda v: f"{v * 100:.2f}%")
                df_display["Recall"] = df_display["Recall"].apply(lambda v: f"{v * 100:.2f}%")
                df_display["F1 Score"] = df_display["F1 Score"].apply(lambda v: f"{v * 100:.2f}%")
                df_display["Training Time"] = df_display["Training Time (ms)"].apply(lambda v: f"{v:.2f} ms")
                df_display["Status"] = df_display["Winner"].apply(lambda w: "⭐ WINNER (Deployed)" if "WINNER" in str(w) or "Best" in str(w) else "Evaluated")
                
                cols_to_show = ["Model", "Status", "F1 Score", "Precision", "Recall", "Accuracy", "Training Time"]
                st.dataframe(df_display[cols_to_show], use_container_width=True)
                
                # Model comparison bar chart (Plotly)
                models_list = df_comp["Model"].tolist()

                fig_comp = go.Figure()
                fig_comp.add_trace(go.Bar(
                    name='F1 Score',
                    x=models_list,
                    y=df_comp["F1 Score"] * 100,
                    marker_color='#00BFA5',
                    hovertemplate='<b>%{x}</b><br>F1 Score: %{y:.2f}%<extra></extra>',
                ))
                fig_comp.add_trace(go.Bar(
                    name='Precision',
                    x=models_list,
                    y=df_comp["Precision"] * 100,
                    marker_color='#4FC3F7',
                    hovertemplate='<b>%{x}</b><br>Precision: %{y:.2f}%<extra></extra>',
                ))
                fig_comp.add_trace(go.Bar(
                    name='Recall',
                    x=models_list,
                    y=df_comp["Recall"] * 100,
                    marker_color='#FFB74D',
                    hovertemplate='<b>%{x}</b><br>Recall: %{y:.2f}%<extra></extra>',
                ))
                fig_comp.update_layout(
                    title=dict(text='Model Performance Comparison (Held-Out Test Set)', font=dict(color='#E8EDF5', size=15, family='Inter')),
                    barmode='group',
                    plot_bgcolor='#131929',
                    paper_bgcolor='#0A0F1E',
                    font=dict(color='#E8EDF5', family='Inter'),
                    xaxis=dict(color='#E8EDF5', gridcolor='rgba(255,255,255,0.06)'),
                    yaxis=dict(title='Score (%)', color='#8A95A8', gridcolor='rgba(255,255,255,0.06)', range=[0, 115]),
                    legend=dict(bgcolor='rgba(19,25,41,0.8)', bordercolor='rgba(0,191,165,0.2)', borderwidth=1),
                    margin=dict(l=20, r=20, t=50, b=30),
                    height=340,
                )
                st.plotly_chart(fig_comp, use_container_width=True)
                
                # Selection criterion explanation
                st.markdown("### 💡 Why F1 Score is Used as the Selection Criterion")
                st.markdown("""
In financial ledger reconciliation, evaluation metrics carry asymmetric business consequences:
- **False Positives (Precision Loss)**: Inaccurately pairing two distinct transactions creates phantom settlements and ledger discrepancies.
- **False Negatives (Recall Loss)**: Missing a legitimate match sends valid transactions to the exception queue, creating unnecessary manual audit workload.

The **F1 Score** calculates the harmonic mean of Precision and Recall:
$$\\text{F1 Score} = 2 \\times \\frac{\\text{Precision} \\times \\text{Recall}}{\\text{Precision} + \\text{Recall}}$$

Unlike raw accuracy—which can appear deceptively high on imbalanced non-match datasets—**F1 Score penalizes both false positives and false negatives equally**, ensuring LedgerLens automatically selects the classifier that maximizes automated clearing without sacrificing matching integrity.
""")
            else:
                st.info("Model comparison results will be available once synthetic dataset training has executed.")

        # ==================== TAB 6: GROUND TRUTH & THRESHOLD SLIDER ====================
        with tab6:
            st.subheader("🎯 Ground Truth Validation & Confidence Threshold Tuning")
            gt_path = os.path.join(raw_dir, "ground_truth.csv")
            if not custom_run and not is_live_api and os.path.exists(gt_path):
                df_gt = pd.read_csv(gt_path)
                
                gt_bank_to_gate = {}
                for _, row in df_gt.iterrows():
                    b_id = row["bank_txn_id"]
                    g_id = row["gateway_order_id"]
                    if b_id != "NO_MATCH":
                        gt_bank_to_gate[b_id] = g_id
                
                # Build lookup maps
                exact_bank_map = {m["bank_txn_id"]: m["gateway_order_id"] for m in exact_matches}
                candidate_ml_map = {m["bank_txn_id"]: (m["gateway_order_id"], float(m["confidence"])) for m in all_candidate_ml_matches}
                
                st.markdown("#### ⚙️ Interactive ML Match Confidence Threshold")
                slider_val = st.slider(
                    "ML Match Confidence Threshold (%)",
                    min_value=50,
                    max_value=99,
                    value=70,
                    step=1,
                    help="Adjust the minimum confidence probability required to accept a predictive ML match."
                )
                
                st.caption("ℹ️ **Tradeoff Rule:** Raising the threshold reduces false positives (fewer wrong matches) but increases false negatives (more transactions requiring manual review). Lowering it does the opposite.")
                
                # Dynamic evaluation function at any threshold
                def evaluate_at_threshold(thresh):
                    tp_val, fp_val, fn_val, tn_val = 0, 0, 0, 0
                    correct_val = 0
                    for b_id, true_g_id in gt_bank_to_gate.items():
                        if b_id in exact_bank_map:
                            p_gid = exact_bank_map[b_id]
                        elif b_id in candidate_ml_map:
                            cand_gid, conf = candidate_ml_map[b_id]
                            p_gid = cand_gid if conf >= thresh else "NO_MATCH"
                        else:
                            p_gid = "NO_MATCH"
                            
                        if true_g_id == p_gid:
                            correct_val += 1
                            if true_g_id != "NO_MATCH":
                                tp_val += 1
                            else:
                                tn_val += 1
                        else:
                            if true_g_id != "NO_MATCH" and p_gid != "NO_MATCH":
                                fp_val += 1
                                fn_val += 1
                            elif true_g_id == "NO_MATCH" and p_gid != "NO_MATCH":
                                fp_val += 1
                            elif true_g_id != "NO_MATCH" and p_gid == "NO_MATCH":
                                fn_val += 1
                                
                    tot = len(gt_bank_to_gate)
                    acc = (correct_val / tot * 100) if tot > 0 else 0.0
                    prec = (tp_val / (tp_val + fp_val) * 100) if (tp_val + fp_val) > 0 else 0.0
                    rec = (tp_val / (tp_val + fn_val) * 100) if (tp_val + fn_val) > 0 else 0.0
                    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
                    return acc, prec, rec, f1, tp_val, tn_val, fp_val, fn_val
                
                # Compute live metrics for selected slider threshold
                curr_acc, curr_prec, curr_rec, curr_f1, curr_tp, curr_tn, curr_fp, curr_fn = evaluate_at_threshold(slider_val)
                
                # Display 4 live metric cards
                col_g1, col_g2, col_g3, col_g4 = st.columns(4)
                with col_g1:
                    st.metric("Model Match Accuracy", f"{curr_acc:.2f}%")
                with col_g2:
                    st.metric("Model Precision", f"{curr_prec:.2f}%")
                with col_g3:
                    st.metric("Model Recall", f"{curr_rec:.2f}%")
                with col_g4:
                    st.metric("Model F1 Score", f"{curr_f1:.2f}%")
                    
                st.markdown(f"### Live Confusion Matrix (Cutoff = {slider_val}%)")
                cm_data = {
                    "Classification Category": [
                        "True Positives (Correct Matches Cleared)", 
                        "True Negatives (Correct Unmatched Exceptions)", 
                        "False Positives (Incorrectly Paired Matches)", 
                        "False Negatives (Missed Valid Matches)"
                    ],
                    "Count": [curr_tp, curr_tn, curr_fp, curr_fn]
                }
                st.table(pd.DataFrame(cm_data))
                
                # Tradeoff Visualization across all thresholds [50...99]
                st.markdown("### 📈 Precision-Recall-F1 Tradeoff Curve")
                threshold_range = list(range(50, 100))
                curve_data = [evaluate_at_threshold(t) for t in threshold_range]
                
                prec_curve = [c[1] for c in curve_data]
                rec_curve = [c[2] for c in curve_data]
                f1_curve = [c[3] for c in curve_data]
                
                fig_trade = go.Figure()
                fig_trade.add_trace(go.Scatter(
                    x=threshold_range, y=prec_curve,
                    mode='lines', name='Precision (%)',
                    line=dict(color='#4FC3F7', width=2.5),
                    hovertemplate='Threshold: %{x}%<br>Precision: %{y:.1f}%<extra></extra>',
                ))
                fig_trade.add_trace(go.Scatter(
                    x=threshold_range, y=rec_curve,
                    mode='lines', name='Recall (%)',
                    line=dict(color='#FFB74D', width=2.5),
                    hovertemplate='Threshold: %{x}%<br>Recall: %{y:.1f}%<extra></extra>',
                ))
                fig_trade.add_trace(go.Scatter(
                    x=threshold_range, y=f1_curve,
                    mode='lines', name='F1 Score (%)',
                    line=dict(color='#00BFA5', width=2.5, dash='dash'),
                    hovertemplate='Threshold: %{x}%<br>F1: %{y:.1f}%<extra></extra>',
                ))
                # Vertical line at current slider value
                fig_trade.add_vline(
                    x=slider_val,
                    line=dict(color='#F48FB1', width=2, dash='dot'),
                    annotation_text=f'Threshold: {slider_val}%',
                    annotation_font_color='#F48FB1',
                    annotation_position='top right',
                )
                fig_trade.update_layout(
                    title=dict(text='Precision / Recall / F1 vs. Confidence Cutoff', font=dict(color='#E8EDF5', size=15, family='Inter')),
                    plot_bgcolor='#131929',
                    paper_bgcolor='#0A0F1E',
                    font=dict(color='#E8EDF5', family='Inter'),
                    xaxis=dict(title='Confidence Threshold (%)', color='#8A95A8', gridcolor='rgba(255,255,255,0.06)'),
                    yaxis=dict(title='Metric (%)', color='#8A95A8', gridcolor='rgba(255,255,255,0.06)', range=[0, 105]),
                    legend=dict(bgcolor='rgba(19,25,41,0.8)', bordercolor='rgba(0,191,165,0.2)', borderwidth=1),
                    margin=dict(l=20, r=20, t=50, b=30),
                    height=340,
                )
                st.plotly_chart(fig_trade, use_container_width=True)
            else:
                st.info(
                    "⚡ **Ground Truth Accuracy is not available in Live Razorpay Test API mode.** "
                    "Because the live session uses real Razorpay order IDs, there is no pre-built "
                    "ground truth file to validate against. Switch to **Use Generated Synthetic Data** "
                    "to use full GT validation and the confidence threshold tuner."
                )

        # ==================== TAB 7: MODEL ERROR ANALYSIS ====================
        with tab7:
            st.subheader("🚨 Model Error Analysis")
            st.markdown(
                "Detailed breakdown of **False Positives** (wrong matches) and **False Negatives** (missed matches) "
                "from the ground truth evaluation. Each error includes the records involved, feature values that "
                "likely caused the error, and a plain-English explanation of the failure mode."
            )

            errors_path = os.path.join(processed_dir, "model_errors.json")

            if custom_run or is_live_api:
                st.info(
                    "⚡ **Model Error Analysis is not available in this mode.** "
                    "Ground truth comparison requires the generated synthetic dataset. "
                    "Switch to **Use Generated Synthetic Data** to see False Positive / "
                    "False Negative breakdowns."
                )
            else:
                gt_path_raw = os.path.join(raw_dir, "ground_truth.csv")
                if not os.path.exists(errors_path) and os.path.exists(gt_path_raw):
                    generate_report(
                        all_matches, still_unmatched_bank, still_unmatched_gateway,
                        bank_df, gateway_df, raw_dir, processed_dir
                    )
                    
                if not os.path.exists(errors_path):
                    st.info("No model_errors.json found. Ground truth comparison is not available.")
                    fp_errors = []
                    fn_errors = []
                else:
                    with open(errors_path, "r", encoding="utf-8") as f:
                        errors = json.load(f)
                    fp_errors = [e for e in errors if e["error_type"] == "FALSE_POSITIVE"]
                    fn_errors = [e for e in errors if e["error_type"] == "FALSE_NEGATIVE"]

                col_e1, col_e2 = st.columns(2)
                with col_e1:
                    st.markdown(f'<div class="metric-card"><h5 class="mc-coral">False Positives</h5><h2>{len(fp_errors)}</h2><p class="mc-neutral">Wrong matches</p></div>', unsafe_allow_html=True)
                with col_e2:
                    st.markdown(f'<div class="metric-card"><h5 class="mc-orange">False Negatives</h5><h2>{len(fn_errors)}</h2><p class="mc-neutral">Missed matches</p></div>', unsafe_allow_html=True)

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

        # ==================== TAB 8: DOWNLOAD REPORTS ====================
        with tab8:
            st.subheader("📥 Download Reconciliation Audit Reports")
            st.markdown(
                "Export complete reconciliation audits in enterprise-standard formats. "
                "Download formatted multi-sheet Excel workbooks or executive-ready PDF audit packages."
            )
            
            # Prepare summary data dictionary for export generators
            total_value_proc = float(bank_df["amount"].sum())
            total_value_rec = float(bank_df[bank_df["txn_id"].isin(matched_bank_ids)]["amount"].sum())
            total_value_exc = float(still_unmatched_bank["amount"].sum())
            hrs_sav = (matched_count * 3) / 60.0
            
            gt_accuracy_val = 100.0
            gt_precision_val = 100.0
            gt_recall_val = 100.0
            
            # Check if gt metrics exist
            gt_metrics_path = os.path.join(processed_dir, "gt_metrics.json")
            if os.path.exists(gt_metrics_path):
                try:
                    with open(gt_metrics_path, "r", encoding="utf-8") as f:
                        saved_gt = json.load(f)
                        gt_accuracy_val = saved_gt.get("accuracy", 100.0)
                        gt_precision_val = saved_gt.get("precision", 100.0)
                        gt_recall_val = saved_gt.get("recall", 100.0)
                except Exception:
                    pass

            summary_report_data = {
                "total_bank": total_bank,
                "total_gateway": total_gateway,
                "exact_count": exact_count,
                "ml_count": ml_count,
                "bank_exc_count": bank_exc_count,
                "gate_exc_count": gate_exc_count,
                "matched_count": matched_count,
                "total_value_processed": total_value_proc,
                "total_value_reconciled": total_value_rec,
                "total_value_exceptions": total_value_exc,
                "hours_saved": hrs_sav,
                "accuracy": gt_accuracy_val,
                "precision": gt_precision_val,
                "recall": gt_recall_val,
            }
            
            # Load error list if available
            df_errors_for_export = None
            if os.path.exists(errors_path):
                try:
                    with open(errors_path, "r", encoding="utf-8") as f:
                        err_data = json.load(f)
                        if err_data:
                            df_errors_for_export = pd.DataFrame(err_data)
                except Exception:
                    pass

            # Report preview cards
            col_rep1, col_rep2 = st.columns(2)
            with col_rep1:
                st.markdown("""
<div class="metric-card">
    <h4 style="color:#2196F3;margin-bottom:8px;">📊 Formatted Excel Workbook (.xlsx)</h4>
    <p style="color:#888;font-size:0.9em;margin-bottom:12px;">Includes 4 comprehensive tabs:</p>
    <ul style="color:#CCC;font-size:0.85em;margin-bottom:15px;">
        <li><b>Summary:</b> KPIs, match distribution & business impact</li>
        <li><b>All Matches:</b> Every matched pair with confidence & AI rationale</li>
        <li><b>Exceptions:</b> Unmatched transactions with assigned reasons</li>
        <li><b>Model Errors:</b> FP/FN diagnostic details (if ground truth active)</li>
    </ul>
</div>
""", unsafe_allow_html=True)
                # Generate Excel
                excel_bytes = generate_excel_report(
                    summary_report_data,
                    df_matches,
                    df_exceptions,
                    df_errors_for_export
                )
                st.download_button(
                    label="📥 Download Excel Report (.xlsx)",
                    data=excel_bytes,
                    file_name="LedgerLens_Reconciliation_Report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    type="primary"
                )
                
            with col_rep2:
                st.markdown("""
<div class="metric-card">
    <h4 style="color:#E91E63;margin-bottom:8px;">📄 Executive Audit PDF (.pdf)</h4>
    <p style="color:#888;font-size:0.9em;margin-bottom:12px;">Formatted for finance auditors & controllers:</p>
    <ul style="color:#CCC;font-size:0.85em;margin-bottom:15px;">
        <li><b>Title & Metadata:</b> Timestamped reconciliation certificate</li>
        <li><b>Business Impact:</b> ₹ cleared volume & labor hours saved</li>
        <li><b>Auditor Action Log:</b> Complete exception table with audit actions</li>
        <li><b>Clean Layout:</b> Professional document designed for archiving</li>
    </ul>
</div>
""", unsafe_allow_html=True)
                # Generate PDF
                pdf_bytes = generate_pdf_report(
                    summary_report_data,
                    df_exceptions
                )
                st.download_button(
                    label="📄 Download PDF Report (.pdf)",
                    data=pdf_bytes,
                    file_name="LedgerLens_Audit_Report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
                
            st.markdown("---")
            st.caption("🔒 All exported reports are generated locally from active ledger datasets.")
else:
    st.info("Upload custom files via the sidebar or run the generator to populate workspace data.")

