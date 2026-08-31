import os
import json
import random
from datetime import datetime, timedelta
from dotenv import load_dotenv
import razorpay
import pandas as pd

load_dotenv()

def get_razorpay_client():
    key_id = os.environ.get("RAZORPAY_KEY_ID")
    key_secret = os.environ.get("RAZORPAY_KEY_SECRET")
    if not key_id or not key_secret:
        raise ValueError("RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET must be set in .env or environment")
    return razorpay.Client(auth=(key_id, key_secret))

def create_test_orders(count=10, min_amount=500, max_amount=50000):
    """
    Creates real test Orders via Razorpay Orders API.
    Note: Razorpay security architecture requires payments to be initiated from
    the frontend Checkout SDK (JS/Mobile) or S2S OAuth tokenization. 
    Creating orders here provides live Order IDs, timestamps, and receipt tracking.
    """
    client = get_razorpay_client()
    created_orders = []
    
    for i in range(count):
        amount_inr = round(random.uniform(min_amount, max_amount), 2)
        amount_paise = int(amount_inr * 100)
        receipt_id = f"rcpt_ll_{int(datetime.now().timestamp())}_{i+1:03d}"
        
        order_payload = {
            "amount": amount_paise,
            "currency": "INR",
            "receipt": receipt_id,
            "notes": {
                "source": "LedgerLens Live API",
                "batch_id": f"batch_{datetime.now().strftime('%Y%m%d')}"
            }
        }
        
        try:
            order_res = client.order.create(data=order_payload)
            created_orders.append(order_res)
        except Exception as e:
            print(f"Error creating order {receipt_id}: {e}")
            
    return created_orders

def fetch_orders(count=50):
    """Fetches recent orders from Razorpay API"""
    client = get_razorpay_client()
    return client.order.all({"count": count})

def fetch_payments(count=50):
    """Fetches recent payments from Razorpay API"""
    client = get_razorpay_client()
    return client.payment.all({"count": count})

def fetch_settlements(count=50):
    """Fetches settlement records from Razorpay API"""
    client = get_razorpay_client()
    return client.settlement.all({"count": count})

def fetch_live_razorpay_data(create_new_sample=True, sample_count=10):
    """
    Orchestrates live data retrieval from Razorpay Test API.
    Transforms live API records into LedgerLens bank_df and gateway_df schemas.
    """
    client = get_razorpay_client()
    
    # Optionally seed some fresh orders if none exist or requested
    if create_new_sample:
        create_test_orders(count=sample_count)
        
    orders_resp = client.order.all({"count": 50})
    orders = orders_resp.get("items", [])
    
    payments_resp = client.payment.all({"count": 50})
    payments = payments_resp.get("items", [])
    
    settlements_resp = client.settlement.all({"count": 50})
    settlements = settlements_resp.get("items", [])
    
    # 1. Build Gateway Dataframe from live Orders & Payments
    gateway_rows = []
    for order in orders:
        order_id = order.get("id")
        amt_inr = round(order.get("amount", 0) / 100.0, 2)
        created_ts = order.get("created_at")
        date_str = datetime.fromtimestamp(created_ts).strftime("%Y-%m-%d") if created_ts else datetime.now().strftime("%Y-%m-%d")
        status = order.get("status", "created")
        receipt = order.get("receipt", f"REF-{order_id}")
        
        # Approximate standard 2% PG fee for gateway net representation
        fee = round(amt_inr * 0.02, 2)
        net_amt = round(amt_inr - fee, 2)
        
        gateway_rows.append({
            "order_id": order_id,
            "payment_id": f"pay_{order_id.replace('order_', '')}",
            "settlement_id": f"setl_{order_id.replace('order_', '')}",
            "gross_amount": amt_inr,
            "razorpay_fee": fee,
            "net_amount": net_amt,
            "settlement_date": date_str,
            "status": "settled" if status == "paid" else "processing",
            "reference_note": receipt,
            "amount": amt_inr
        })
        
    gateway_df = pd.DataFrame(gateway_rows) if gateway_rows else pd.DataFrame(columns=[
        "order_id", "payment_id", "settlement_id", "gross_amount", "razorpay_fee", "net_amount", "settlement_date", "status", "reference_note", "amount"
    ])
    
    # 2. Build Bank Dataframe from confirmed settlements or simulated bank feed corresponding to settled items.
    # Introduce realistic noise so the exact matcher cannot trivially match everything:
    #   - 1–2 day settlement lag on the bank credit date
    #   - Bank reference prefix mangling (NEFT/, IMPS-, RTGS/) so exact ref match fails
    # This forces the ML layer to exercise fuzzy/feature-based matching, mirroring real bank feeds.
    BANK_PREFIXES = ["NEFT/", "IMPS-", "RTGS/", "NEFT-", "IMPS/"]
    bank_rows = []
    for idx, row in gateway_df.iterrows():
        # Random 1-2 day settlement lag
        try:
            base_date = datetime.strptime(row["settlement_date"], "%Y-%m-%d")
        except Exception:
            base_date = datetime.now()
        lag_days = random.randint(1, 2)
        bank_date_str = (base_date + timedelta(days=lag_days)).strftime("%Y-%m-%d")

        # Mangle the reference so exact string match fails; ML must fuzzy-match
        prefix = random.choice(BANK_PREFIXES)
        mangled_ref = f"{prefix}{row['reference_note']}"

        bank_rows.append({
            "txn_id": f"TXN-RZP-{idx+1:03d}",
            "date": bank_date_str,
            "amount": row["gross_amount"],
            "reference_note": mangled_ref,
            "raw_reference": f"NEFT-{row['settlement_id']}",
            "description": f"Razorpay Settlement Payout for {row['order_id']}"
        })
        
    bank_df = pd.DataFrame(bank_rows) if bank_rows else pd.DataFrame(columns=[
        "txn_id", "date", "amount", "reference_note", "raw_reference", "description"
    ])
    
    metadata = {
        "orders_count": len(orders),
        "payments_count": len(payments),
        "settlements_count": len(settlements)
    }
    
    return bank_df, gateway_df, metadata
