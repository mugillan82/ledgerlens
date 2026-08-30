import os
import json
from dotenv import load_dotenv
import razorpay

load_dotenv()

key_id = os.environ.get("RAZORPAY_KEY_ID")
key_secret = os.environ.get("RAZORPAY_KEY_SECRET")

print(f"Key ID configured: {bool(key_id)} ({key_id[:8]}... if key_id else 'None')")
print(f"Key Secret configured: {bool(key_secret)}")

client = razorpay.Client(auth=(key_id, key_secret))

print("\n--- Available resources in razorpay Client ---")
print([attr for attr in dir(client) if not attr.startswith("_")])

print("\n--- 1. Testing Order Creation ---")
try:
    order_data = {
        "amount": 50000,  # in paise: 500.00 INR
        "currency": "INR",
        "receipt": "rcpt_test_001",
        "notes": {"source": "LedgerLens Test"}
    }
    order_res = client.order.create(data=order_data)
    print("Order Create Response:")
    print(json.dumps(order_res, indent=2))
except Exception as e:
    print(f"Order creation failed: {e}")

print("\n--- 2. Testing Payment Creation / Capture methods ---")
print("client.payment methods:", [m for m in dir(client.payment) if not m.startswith("_")])

# Test creating a payment or checking if server-side payment creation is supported
try:
    # Let's inspect methods or check client.payment
    pass
except Exception as e:
    print(f"Payment test failed: {e}")

print("\n--- 3. Testing Settlement API ---")
print("client.settlement methods:", [m for m in dir(client.settlement) if not m.startswith("_")])
try:
    settlements = client.settlement.all()
    print("Settlements all() Response:")
    print(json.dumps(settlements, indent=2))
except Exception as e:
    print(f"Settlements fetch failed: {e}")

print("\n--- 4. Testing Payments list ---")
try:
    payments = client.payment.all()
    print(f"Payments count: {len(payments.get('items', []))}")
    print("Recent Payments:")
    print(json.dumps(payments, indent=2))
except Exception as e:
    print(f"Payments fetch failed: {e}")
