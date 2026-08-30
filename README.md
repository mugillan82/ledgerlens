# LedgerLens — ML Reconciliation Agent

## Overview
LedgerLens is an end-to-end reconciliation pipeline that reconciles bank statements against payment gateway settlements using deterministic exact matching combined with a trained Machine Learning classifier for ambiguous cases. Built for finance and operations teams who spend hours performing manual line-by-line reconciliation, it automates settlement matching, highlights edge cases with clear reason codes, and provides full auditability and explainability.

## Problem Statement
In real-world payment operations, bank statement credits rarely match payment gateway settlement records one-to-one due to processing fee deductions (1–3%), settlement time lags (1–3 days), and reference note truncation or typos introduced across banking rails. Unhandled duplicate records and edge-case exceptions often force finance teams to manually compare mismatched spreadsheets row by row. This manual process is time-consuming, prone to human error, and lacks an audit trail explaining why matches or mismatches occurred.

## How It Works
The pipeline follows a structured multi-stage workflow:
1. **Exact Matching (Rule-Based, High Confidence)**: Cleans and normalizes reference notes, comparing bank records against gateway settlements for identical references, amounts, and dates with 100% confidence.
2. **ML-Based Matching for Ambiguous Leftovers**: Unmatched records are evaluated by a trained Logistic Regression classifier using 5 extracted features: `amount_diff_pct`, `date_diff_days`, `reference_similarity` (rapidfuzz), `amount_within_fee_range` (1–3% gateway fee heuristics), and `ref_exact_match`. Each ML match includes dynamic, human-readable explanations of feature contributions.
3. **Exception Handling with Reason Codes**: Remaining unmatched transactions from both sides are assigned explicit diagnosis codes (e.g., *No corresponding gateway entry found*, *Date beyond settlement window*, or *Amount mismatch beyond tolerance*) and exported to `exceptions.csv`.
4. **Ground-Truth Validation**: Evaluates pipeline predictions against labelled benchmark data (`ground_truth.csv`), computing precision, recall, and a confusion matrix with full visibility into false positives and false negatives along with root-cause explanations.
5. **Business Impact Translation**: Translates technical matching metrics into financial metrics, computing the total monetary value auto-reconciled, un-reconciled value requiring review, and estimated manual audit hours saved.

## Tech Stack
- **Python** (Core runtime)
- **pandas** & **numpy** (Data processing & transformations)
- **scikit-learn** (Logistic Regression model training & inference)
- **rapidfuzz** (Fuzzy string similarity scoring)
- **Streamlit** (Interactive monitoring dashboard & analytics)
- **matplotlib** (Visual breakdown & metrics plotting)
- **Faker** (Realistic synthetic transaction & settlement generation)

## Results (From Actual Last Run)
- **Total records processed**: 488 bank transactions, 490 gateway records
- **Exact matches**: 338 (69.26%)
- **ML matches**: 136 (27.87%) (Avg Confidence: 98.20%)
- **Exceptions**: Bank: 14 (2.87%), Gateway: 16 (3.27%)
- **Ground truth accuracy**: 97.54%
- **Precision**: 98.31%
- **Recall**: 98.73%
- **False positives**: 8, **False negatives**: 6
- **Total value auto-reconciled**: ₹7,09,16,792.66 (96.79% of total value processed: ₹7,32,71,127.65)
- **Estimated audit hours saved**: 23.70 hrs (based on ~3 min per auto-matched record)

## What Makes This More Than Rule-Based Automation
Unlike traditional reconciliation tools that depend on brittle hardcoded if-else thresholds, this system utilizes a trained statistical Logistic Regression classifier to evaluate multi-dimensional decision boundaries across date lags, fee percentage structures, and text similarities. Crucially, the system offers complete transparency by surfacing its own false positives and false negatives with per-error root-cause diagnostics, and provides per-prediction plain-English explanations identifying which feature values drove each match decision.

## Project Structure
```
ledgerlens/
├── dashboard/
│   └── app.py                     # Streamlit interactive dashboard application
├── data/
│   ├── raw/
│   │   ├── bank_statement.csv     # Synthetic raw bank statement feed
│   │   ├── gateway_settlement.csv # Synthetic raw payment gateway settlement records
│   │   └── ground_truth.csv       # Labelled ground truth matches for evaluation
│   └── processed/
│       ├── business_metrics.json  # Exported business impact and financial metrics
│       ├── exceptions.csv         # Flagged exceptions with diagnostic reason codes
│       ├── gt_metrics.json        # Model evaluation metrics (precision, recall, accuracy)
│       ├── matches.csv            # Combined reconciliation matches with explanations
│       ├── model_errors.json      # Root-cause breakdown of model FP/FN errors
│       ├── reconciliation_breakdown.png # Generated visual breakdown chart
│       └── reconciliation_report.md     # Comprehensive human-readable summary report
├── models/
│   └── match_classifier.pkl       # Serialized trained Logistic Regression classifier
├── src/
│   ├── __init__.py                # Package initialization
│   ├── data_generator.py          # Synthetic financial dataset generator (Faker)
│   ├── exact_matcher.py           # Stage 1 deterministic exact matching logic
│   ├── exception_handler.py       # Stage 3 exception categorization & reason coding
│   ├── fuzzy_matcher.py           # Legacy fuzzy matching logic (pre-ML module)
│   ├── ml_matcher.py              # Stage 2 ML feature extraction, training & inference
│   └── reporter.py                # Reporting, validation against GT & business impact calculation
├── diagnose.py                    # Diagnosis script for pipeline debugging
├── diagnose2.py                   # Secondary validation script for metric inspection
├── main.py                        # Main pipeline execution entry point
├── requirements.txt               # Project Python package dependencies
├── verify_final.py                # Final verification & consistency test script
└── README.md                      # Project documentation and guide
```

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/mugillan82/ledgerlens.git
cd ledgerlens
```

### 2. Create and activate virtual environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Install requirements
```bash
pip install -r requirements.txt
```

### 4. Run the reconciliation pipeline
```bash
python main.py
```
*This will generate synthetic data (if not present), train the ML classifier, run exact & ML matchers, generate exception logs, validate against ground truth, and write outputs to `data/processed/`.*

### 5. Launch the Streamlit dashboard
```bash
streamlit run dashboard/app.py
```

## Limitations & Future Work
- **Synthetic Data**: The project currently relies on synthetic transaction sets. While designed around Razorpay patterns, it has not been validated against live production settlement schemas with complex banking nuances.
- **Fixed Heuristic Feature Windows**: Fee band tolerances (1–3.05%) and settlement date deltas (0–3 days) are currently heuristic priors informing the ML model rather than dynamically learned per merchant.
- **Single-Match Architecture**: The pipeline currently matches 1-to-1 transactions (one bank transaction to one gateway record). Future iterations could support 1-to-many or many-to-many batch settlement splits.
