import os
import pandas as pd
from .exact_matcher import normalize_ref

def run_exception_handler(still_unmatched_bank, still_unmatched_gateway, original_bank, original_gateway, output_dir="data/processed"):
    """
    Analyzes unmatched records and assigns detailed reasons for reconciliation failure.
    Saves the final exceptions to data/processed/exceptions.csv.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    exceptions = []
    
    # Pre-normalize reference notes for matching checks
    orig_bank = original_bank.copy()
    orig_gateway = original_gateway.copy()
    orig_bank["norm_ref"] = orig_bank["reference_note"].apply(normalize_ref)
    orig_gateway["norm_ref"] = orig_gateway["reference_note"].apply(normalize_ref)
    
    # Bank Exceptions
    for _, row in still_unmatched_bank.iterrows():
        b_ref = normalize_ref(row["reference_note"])
        b_amount = float(row["amount"])
        b_date = pd.to_datetime(row["date"])
        
        # Check if the reference note exists anywhere in the gateway
        matching_gateways = orig_gateway[orig_gateway["norm_ref"] == b_ref]
        
        if matching_gateways.empty:
            reason = "No corresponding gateway entry found"
        else:
            # Let's inspect the potential matches to see why they failed
            g_dates = pd.to_datetime(matching_gateways["settlement_date"])
            g_amounts = matching_gateways["amount"].astype(float)
            
            # Check date window: settlement should be 0 to 3 days after bank
            date_ok = ((g_dates - b_date).dt.days >= 0) & ((g_dates - b_date).dt.days <= 3)
            # Check amount window: bank - gateway should be 0% to 3% fee
            amt_diff = b_amount - g_amounts
            fee_pct = amt_diff / b_amount
            amount_ok = (fee_pct >= 0.0) & (fee_pct <= 0.0305)
            
            if not date_ok.any() and amount_ok.any():
                reason = "Date beyond settlement window"
            elif not amount_ok.any() and date_ok.any():
                reason = "Amount mismatch beyond tolerance"
            else:
                # Both or neither
                if not date_ok.any():
                    reason = "Date beyond settlement window"
                else:
                    reason = "Amount mismatch beyond tolerance"
                    
        exceptions.append({
            "source": "Bank Statement",
            "record_id": row["txn_id"],
            "date": row["date"],
            "amount": row["amount"],
            "reference_note": row["reference_note"],
            "reason": reason
        })
        
    # Gateway Exceptions
    for _, row in still_unmatched_gateway.iterrows():
        g_ref = normalize_ref(row["reference_note"])
        g_amount = float(row["amount"])
        g_date = pd.to_datetime(row["settlement_date"])
        
        matching_banks = orig_bank[orig_bank["norm_ref"] == g_ref]
        
        if matching_banks.empty:
            reason = "No corresponding bank entry found"
        else:
            b_dates = pd.to_datetime(matching_banks["date"])
            b_amounts = matching_banks["amount"].astype(float)
            
            # Check date window
            date_ok = ((g_date - b_dates).dt.days >= 0) & ((g_date - b_dates).dt.days <= 3)
            # Check amount window
            amt_diff = b_amounts - g_amount
            fee_pct = amt_diff / b_amounts
            amount_ok = (fee_pct >= 0.0) & (fee_pct <= 0.0305)
            
            if not date_ok.any() and amount_ok.any():
                reason = "Date beyond settlement window"
            elif not amount_ok.any() and date_ok.any():
                reason = "Amount mismatch beyond tolerance"
            else:
                if not date_ok.any():
                    reason = "Date beyond settlement window"
                else:
                    reason = "Amount mismatch beyond tolerance"
                    
        exceptions.append({
            "source": "Gateway Settlement",
            "record_id": row["order_id"],
            "date": row["settlement_date"],
            "amount": row["amount"],
            "reference_note": row["reference_note"],
            "reason": reason
        })
        
    df_exceptions = pd.DataFrame(exceptions)
    output_path = os.path.join(output_dir, "exceptions.csv")
    df_exceptions.to_csv(output_path, index=False)
    
    return df_exceptions
