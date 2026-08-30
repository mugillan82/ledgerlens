import os
import pandas as pd
from src.data_generator import generate_synthetic_data
from src.exact_matcher import run_exact_matcher
from src.ml_matcher import train_ml_classifier, run_ml_matcher
from src.exception_handler import run_exception_handler
from src.reporter import generate_report

def main():
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        
    print("==================================================")
    print("      AI FINANCE RECONCILIATION PIPELINE (ML)")
    print("==================================================")
    
    raw_dir = "data/raw"
    processed_dir = "data/processed"
    
    bank_path = os.path.join(raw_dir, "bank_statement.csv")
    gateway_path = os.path.join(raw_dir, "gateway_settlement.csv")
    
    # 1. Generate synthetic data if not exists
    if not os.path.exists(bank_path) or not os.path.exists(gateway_path):
        print("\n[Step 1] Synthetic data not found. Generating synthetic files...")
        generate_synthetic_data(raw_dir)
    else:
        print("\n[Step 1] Loading existing raw transaction files...")
        
    df_bank = pd.read_csv(bank_path)
    df_gateway = pd.read_csv(gateway_path)
    
    # 2. Offline Training Phase
    print("\n[Step 2] Launching offline ML model training...")
    train_ml_classifier(raw_dir, "models")
    
    # 3. Exact Matcher
    print("\n[Step 3] Running Exact Matcher...")
    exact_matches, unmatched_bank, unmatched_gateway = run_exact_matcher(df_bank, df_gateway)
    print(f"-> Found {len(exact_matches)} exact matches.")
    print(f"-> Leftovers: {len(unmatched_bank)} bank, {len(unmatched_gateway)} gateway.")
    
    # 4. ML Matcher (replacing fuzzy matcher)
    print("\n[Step 4] Running ML Matcher on leftovers...")
    ml_matches, still_unmatched_bank, still_unmatched_gateway = run_ml_matcher(
        unmatched_bank, unmatched_gateway, model_path="models/match_classifier.pkl", threshold=0.7
    )
    print(f"-> Found {len(ml_matches)} ML matches.")
    print(f"-> Leftovers: {len(still_unmatched_bank)} bank, {len(still_unmatched_gateway)} gateway.")
    
    # Combine matches
    all_matches = exact_matches + ml_matches
    os.makedirs(processed_dir, exist_ok=True)
    df_matches = pd.DataFrame(all_matches)
    df_matches.to_csv(os.path.join(processed_dir, "matches.csv"), index=False)
    print(f"-> Matches saved to {os.path.join(processed_dir, 'matches.csv')}")
    
    # 5. Exception Handler
    print("\n[Step 5] Running Exception Handler...")
    # Clean matches formatting for exception handler if explanation is present
    df_exceptions = run_exception_handler(
        still_unmatched_bank, still_unmatched_gateway, df_bank, df_gateway, processed_dir
    )
    print(f"-> Processed {len(df_exceptions)} exceptions, saved to {os.path.join(processed_dir, 'exceptions.csv')}")
    
    # 6. Reporter
    print("\n[Step 6] Generating Summary, Business Impact & Validation Reports...")
    report_text, chart_path = generate_report(
        all_matches, still_unmatched_bank, still_unmatched_gateway, df_bank, df_gateway, raw_dir, processed_dir
    )
    
    print("\n" + report_text)
    print(f"Matplotlib chart saved to: {chart_path}")
    print("==================================================")
    print("              RECONCILIATION COMPLETE")
    print("==================================================")

if __name__ == "__main__":
    main()
