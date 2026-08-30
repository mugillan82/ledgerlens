import os
import json
from dotenv import load_dotenv
import razorpay

load_dotenv()

key_id = os.environ.get("RAZORPAY_KEY_ID")
key_secret = os.environ.get("RAZORPAY_KEY_SECRET")
client = razorpay.Client(auth=(key_id, key_secret))

print("1. Testing createPaymentJson:")
try:
    res = client.payment.createPaymentJson({
        "amount": 10000,
        "currency": "INR",
        "email": "test@example.com",
        "contact": "9999999999",
        "method": "card"
    })
    print("createPaymentJson response:", res)
except Exception as e:
    print(f"createPaymentJson failed: {e}")

print("\n2. Testing createUpi:")
try:
    res = client.payment.createUpi({
        "amount": 10000,
        "currency": "INR",
        "email": "test@example.com",
        "contact": "9999999999",
        "vpa": "test@razorpay"
    })
    print("createUpi response:", res)
except Exception as e:
    print(f"createUpi failed: {e}")

print("\n3. Testing Payment Link creation:")
try:
    pl_res = client.payment_link.create({
        "amount": 100000, # 1000 INR
        "currency": "INR",
        "accept_partial": False,
        "description": "LedgerLens Test Payment Link",
        "customer": {
            "name": "Test Merchant",
            "email": "test@example.com",
            "contact": "+919876543210"
        },
        "notify": {"sms": False, "email": False},
        "reminder_enable": False,
        "notes": {"policy_name": "LedgerLens"}
    })
    print("Payment link created:", json.dumps(pl_res, indent=2))
except Exception as e:
    print(f"Payment link creation failed: {e}")
