import os
import time
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.exact_matcher import normalize_ref
from src.ml_matcher import compute_features

def extract_dataset(raw_dir="data/raw"):
    """
    Builds the feature matrix X and label vector y from ground_truth.csv and raw statements.
    Uses the exact 5 features:
    [amount_diff_pct, date_diff_days, reference_similarity, amount_within_fee_range, ref_exact_match]
    """
    df_bank = pd.read_csv(os.path.join(raw_dir, "bank_statement.csv"))
    df_gateway = pd.read_csv(os.path.join(raw_dir, "gateway_settlement.csv"))
    df_gt = pd.read_csv(os.path.join(raw_dir, "ground_truth.csv"))
    
    bank_map = df_bank.set_index("txn_id").to_dict(orient="index")
    gateway_map = df_gateway.set_index("order_id").to_dict(orient="index")
    
    X = []
    y = []
    
    # 1. Positive match samples
    true_pairs = []
    for _, row in df_gt.iterrows():
        b_id = row["bank_txn_id"]
        g_id = row["gateway_order_id"]
        if b_id != "NO_MATCH" and g_id != "NO_MATCH":
            true_pairs.append((b_id, g_id))
            
    for b_id, g_id in true_pairs:
        if b_id in bank_map and g_id in gateway_map:
            features = compute_features(bank_map[b_id], gateway_map[g_id])
            X.append(features)
            y.append(1)
            
    # 2. Negative match samples (synthesized hard/random negatives)
    np.random.seed(42)
    bank_ids = list(bank_map.keys())
    gate_ids = list(gateway_map.keys())
    true_set = set(true_pairs)
    
    for b_id in bank_ids:
        b_row = bank_map[b_id]
        for _ in range(3):
            random_g_id = np.random.choice(gate_ids)
            if (b_id, random_g_id) not in true_set:
                features = compute_features(b_row, gateway_map[random_g_id])
                X.append(features)
                y.append(0)
                
    return np.array(X), np.array(y)

def compare_and_train_models(raw_dir="data/raw", model_dir="models", processed_dir="data/processed"):
    """
    Trains Logistic Regression, Random Forest, and Gradient Boosting on a stratified train/test split.
    Evaluates accuracy, precision, recall, F1 score, and training time.
    Saves a comparison table and the winning model (highest F1 score) to models/match_classifier.pkl.
    """
    print("\n==================================================")
    print("      ML MODEL BENCHMARKING & COMPARISON")
    print("==================================================")
    
    X, y = extract_dataset(raw_dir)
    print(f"Dataset extracted: {len(X)} total samples ({sum(y == 1)} positive, {sum(y == 0)} negative)")
    
    # Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training samples: {len(X_train)} | Held-out test samples: {len(X_test)}")
    
    candidate_models = {
        "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    results = []
    trained_models = {}
    
    for name, clf in candidate_models.items():
        # Measure training time
        start_time = time.perf_counter()
        clf.fit(X_train, y_train)
        train_time_ms = (time.perf_counter() - start_time) * 1000.0
        
        # Evaluate on held-out test split
        y_pred = clf.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        trained_models[name] = clf
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "Training Time (ms)": round(train_time_ms, 2)
        })
        
    df_results = pd.DataFrame(results)
    
    # Determine best model by F1 Score (tie-breaker: precision, then recall)
    df_sorted = df_results.sort_values(by=["F1 Score", "Precision", "Recall"], ascending=False).reset_index(drop=True)
    best_model_name = df_sorted.iloc[0]["Model"]
    best_f1 = df_sorted.iloc[0]["F1 Score"]
    
    # Mark winner in dataframe
    df_results["Winner"] = df_results["Model"].apply(lambda m: "[WINNER]" if m == best_model_name else "")
    
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass
            
    print("\n--- Model Benchmark Results (Held-Out Test Set) ---")
    print(df_results.to_string(index=False))
    print(f"\nWinning Model: {best_model_name} (F1 Score: {best_f1:.4f})")
    
    # Save comparison table as CSV to both models/ and data/processed/
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    csv_path_models = os.path.join(model_dir, "model_comparison.csv")
    csv_path_processed = os.path.join(processed_dir, "model_comparison.csv")
    df_results.to_csv(csv_path_models, index=False)
    df_results.to_csv(csv_path_processed, index=False)
    
    # Save the winning model
    best_model = trained_models[best_model_name]
    model_save_path = os.path.join(model_dir, "match_classifier.pkl")
    
    joblib.dump(best_model, model_save_path)
    
    # Also save metadata for dashboard display
    metadata = {
        "best_model_name": best_model_name,
        "best_f1": best_f1,
        "models_evaluated": list(candidate_models.keys()),
        "results": results
    }
    joblib.dump(metadata, os.path.join(model_dir, "model_metadata.pkl"))
    
    print(f"Saved winning model ({best_model_name}) to: {model_save_path}")
    print(f"Saved comparison report to: {csv_path_models}")
    print("==================================================\n")
    
    return df_results, best_model_name

if __name__ == "__main__":
    compare_and_train_models()
