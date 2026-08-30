import os
import json
from dotenv import load_dotenv
import razorpay

load_dotenv()

key_id = os.environ.get("RAZORPAY_KEY_ID")
key_secret = os.environ.get("RAZORPAY_KEY_SECRET")
client = razorpay.Client(auth=(key_id, key_secret))

# 1. Test create recurring / upi / payment json or orders
print("--- Testing client.payment methods ---")
try:
    print("Testing client.order.all():")
    orders = client.order.all({"count": 5})
    print(f"Orders count: {orders.get('count')}")
    for o in orders.get("items", []):
        print(f"Order: {o['id']} | Amount: {o['amount']/100} | Status: {o['status']} | Receipt: {o['receipt']}")
except Exception as e:
    print(f"client.order.all failed: {e}")

# 2. Test client.payment.createPaymentJson or createRecurring or direct payments
try:
    print("\nTesting client.payment.createPaymentJson:")
    # Check signature or doc
    import inspect
    print("createPaymentJson args:", inspect.signature(client.payment.createPaymentJson))
except Exception as e:
    print(f"createPaymentJson inspection failed: {e}")

try:
    print("\nTesting client.payment.createUpi:")
    import inspect
    print("createUpi args:", inspect.signature(client.payment.createUpi))
except Exception as e:
    print(f"createUpi inspection failed: {e}")

# 3. Test client.payment_link
print("\nTesting Payment Links:")
try:
    pl = client.payment_link.all({"count": 5})
    print(f"Payment Links count: {pl.get('count')}")
except Exception as e:
    print(f"Payment link fetch failed: {e}")

# 4. Test settlements detail / on-demand
try:
    print("\nTesting client.settlement.report:")
    # report is available
    print("settlement report params:", inspect.signature(client.settlement.report))
except Exception as e:
    print(f"settlement report inspection failed: {e}")
