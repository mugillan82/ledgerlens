import os
import random
import pandas as pd
from datetime import datetime, timedelta
from faker import Faker

def generate_synthetic_data(data_dir="data/raw", seed=42):
    random.seed(seed)
    fake = Faker()
    Faker.seed(seed)
    
    os.makedirs(data_dir, exist_ok=True)
    
    relations = []
    total_txns = 500
    
    # 5% of 500 is 25 records. Let's designate specific counts:
    confusable_count = 6       # 3 sets of 2 near-duplicates (6 records)
    typo_count = 10            # 10 records with reference typos
    split_count = 3            # 3 bank records split into 6 gateway records
    
    # Adjust other counts so total base matches remain around 481
    # Let's subtract 19 matching records from the exact matches
    exact_count = 331
    fuzzy_amount_count = 75
    fuzzy_date_count = 50
    exception_count = 25
    
    bank_exc_count = exception_count // 2 + (exception_count % 2) # 13
    gateway_exc_count = exception_count // 2 # 12
    
    start_date = datetime(2026, 8, 1)
    txn_seq = 1000
    
    def get_base_attributes():
        nonlocal txn_seq
        txn_seq += 1
        date = start_date + timedelta(days=random.randint(0, 25))
        amount = round(random.uniform(500.0, 300000.0), 2)
        ref_num = f"TXN{txn_seq}"
        desc = fake.sentence(nb_words=4)
        return date, amount, ref_num, desc

    # 1. Exact matches
    for _ in range(exact_count):
        date, amount, ref_num, desc = get_base_attributes()
        bank_rec = {"date": date.strftime("%Y-%m-%d"), "amount": amount, "reference_note": ref_num, "description": desc}
        gate_rec = {"settlement_date": date.strftime("%Y-%m-%d"), "amount": amount, "reference_note": f"REF-{ref_num}", "fee": 0.00, "status": "Settled"}
        relations.append({"bank": bank_rec, "gateway": gate_rec, "match_type": "EXACT"})
        
    # 2. Fuzzy Amount matches
    for _ in range(fuzzy_amount_count):
        date, amount, ref_num, desc = get_base_attributes()
        fee_pct = round(random.uniform(0.01, 0.03), 4)
        fee = round(amount * fee_pct, 2)
        gateway_amount = round(amount - fee, 2)
        bank_rec = {"date": date.strftime("%Y-%m-%d"), "amount": amount, "reference_note": ref_num, "description": desc}
        gate_rec = {"settlement_date": date.strftime("%Y-%m-%d"), "amount": gateway_amount, "reference_note": f"REF-{ref_num}", "fee": fee, "status": "Settled"}
        relations.append({"bank": bank_rec, "gateway": gate_rec, "match_type": "FUZZY_AMOUNT"})
        
    # 3. Fuzzy Date matches
    for _ in range(fuzzy_date_count):
        date, amount, ref_num, desc = get_base_attributes()
        delay = random.randint(1, 3)
        settlement_date = date + timedelta(days=delay)
        bank_rec = {"date": date.strftime("%Y-%m-%d"), "amount": amount, "reference_note": ref_num, "description": desc}
        gate_rec = {"settlement_date": settlement_date.strftime("%Y-%m-%d"), "amount": amount, "reference_note": f"REF-{ref_num}", "fee": 0.00, "status": "Settled"}
        relations.append({"bank": bank_rec, "gateway": gate_rec, "match_type": "FUZZY_DATE"})

    # 4. Hard cases: Near-duplicate confusables
    # 3 sets of 2. Each set shares exact same date and amount, but different references.
    for set_idx in range(3):
        date = start_date + timedelta(days=random.randint(0, 25))
        amount = round(random.uniform(5000.0, 150000.0), 2)
        for _ in range(2):
            _, _, ref_num, desc = get_base_attributes()
            bank_rec = {"date": date.strftime("%Y-%m-%d"), "amount": amount, "reference_note": ref_num, "description": desc}
            gate_rec = {"settlement_date": date.strftime("%Y-%m-%d"), "amount": amount, "reference_note": f"REF-{ref_num}", "fee": 0.00, "status": "Settled"}
            relations.append({"bank": bank_rec, "gateway": gate_rec, "match_type": "CONFUSABLE"})

    # 5. Hard cases: Noisy reference typos
    for _ in range(typo_count):
        date, amount, ref_num, desc = get_base_attributes()
        # ref_num is like "TXN1469". Split into prefix + digits.
        # Apply the transposition only inside the DIGIT portion so that after
        # normalize_ref both sides share the same numeric core with a small edit.
        prefix = "TXN"
        digits = ref_num[len(prefix):]  # e.g. "1469"
        digit_list = list(digits)
        # Transpose the last two digits to create a realistic typo
        if len(digit_list) >= 2:
            digit_list[-1], digit_list[-2] = digit_list[-2], digit_list[-1]
        typo_digits = "".join(digit_list)
        typo_ref = prefix + typo_digits  # e.g. "TXN1496"

        bank_rec = {"date": date.strftime("%Y-%m-%d"), "amount": amount, "reference_note": ref_num, "description": desc}
        gate_rec = {"settlement_date": date.strftime("%Y-%m-%d"), "amount": amount, "reference_note": f"REF-{typo_ref}", "fee": 0.00, "status": "Settled"}
        relations.append({"bank": bank_rec, "gateway": gate_rec, "match_type": "TYPO"})


    # 6. Hard cases: Split settlement (1 bank txn maps to 2 gateway settlements)
    for _ in range(split_count):
        date, amount, ref_num, desc = get_base_attributes()
        amt1 = round(amount * 0.6, 2)
        amt2 = round(amount - amt1, 2)
        
        bank_rec = {"date": date.strftime("%Y-%m-%d"), "amount": amount, "reference_note": ref_num, "description": desc}
        gate_rec1 = {"settlement_date": date.strftime("%Y-%m-%d"), "amount": amt1, "reference_note": f"REF-{ref_num}-A", "fee": 0.00, "status": "Settled"}
        gate_rec2 = {"settlement_date": date.strftime("%Y-%m-%d"), "amount": amt2, "reference_note": f"REF-{ref_num}-B", "fee": 0.00, "status": "Settled"}
        
        # In a 1:1 ground truth file, splits cannot be represented cleanly as matches.
        # We will record them as unmatched bank/gate exceptions, but marked as "SPLIT_SETTLEMENT"
        relations.append({"bank": bank_rec, "gateway": None, "match_type": "SPLIT_SETTLEMENT"})
        relations.append({"bank": None, "gateway": gate_rec1, "match_type": "SPLIT_SETTLEMENT"})
        relations.append({"bank": None, "gateway": gate_rec2, "match_type": "SPLIT_SETTLEMENT"})

    # 7. Standard Bank exceptions
    for _ in range(bank_exc_count):
        date, amount, ref_num, desc = get_base_attributes()
        bank_rec = {"date": date.strftime("%Y-%m-%d"), "amount": amount, "reference_note": ref_num, "description": desc}
        relations.append({"bank": bank_rec, "gateway": None, "match_type": "BANK_EXCEPTION"})
        
    # 8. Standard Gateway exceptions
    for _ in range(gateway_exc_count):
        date, amount, ref_num, desc = get_base_attributes()
        fee_pct = round(random.uniform(0.01, 0.03), 4)
        fee = round(amount * fee_pct, 2)
        gateway_amount = round(amount - fee, 2)
        gate_rec = {"settlement_date": date.strftime("%Y-%m-%d"), "amount": gateway_amount, "reference_note": f"REF-{ref_num}", "fee": fee, "status": "Settled"}
        relations.append({"bank": None, "gateway": gate_rec, "match_type": "GATEWAY_EXCEPTION"})
        
    # Separate records
    bank_records = [r["bank"] for r in relations if r["bank"] is not None]
    gateway_records = [r["gateway"] for r in relations if r["gateway"] is not None]
    
    # Shuffle completely before ID assignment
    random.shuffle(bank_records)
    random.shuffle(gateway_records)
    
    for idx, b_rec in enumerate(bank_records):
        b_rec["txn_id"] = f"TXN-{idx+1:05d}"
        
    for idx, g_rec in enumerate(gateway_records):
        g_rec["order_id"] = f"ORD-{idx+1:05d}"
        
    # Build Ground Truth
    ground_truth_records = []
    for rel in relations:
        b_id = rel["bank"]["txn_id"] if rel["bank"] is not None else "NO_MATCH"
        g_id = rel["gateway"]["order_id"] if rel["gateway"] is not None else "NO_MATCH"
        ground_truth_records.append({
            "bank_txn_id": b_id,
            "gateway_order_id": g_id,
            "match_type": rel["match_type"]
        })
        
    df_bank = pd.DataFrame(bank_records)
    df_gateway = pd.DataFrame(gateway_records)
    df_ground_truth = pd.DataFrame(ground_truth_records)
    
    df_bank = df_bank[["txn_id", "date", "amount", "reference_note", "description"]]
    df_gateway = df_gateway[["order_id", "settlement_date", "amount", "reference_note", "fee", "status"]]
    
    df_bank.to_csv(os.path.join(data_dir, "bank_statement.csv"), index=False)
    df_gateway.to_csv(os.path.join(data_dir, "gateway_settlement.csv"), index=False)
    df_ground_truth.to_csv(os.path.join(data_dir, "ground_truth.csv"), index=False)
    
    print(f"Generated {len(df_bank)} bank statements, {len(df_gateway)} gateway settlements.")
    print(f"Ground truth saved with {len(df_ground_truth)} records.")

if __name__ == "__main__":
    generate_synthetic_data()
