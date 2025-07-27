from iqoptionapi.stable_api import IQ_Option
import logging
import time
from secret import secret
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s %(message)s')
# Credentials
EMAIL = secret.get('email')
PASSWORD = secret.get('password')
Iq=IQ_Option(EMAIL,PASSWORD)

# Connect and check login
Iq.connect()
if Iq.check_connect():
    print("Successfully connected!")
else:
    print("Failed to connect. Check your credentials or internet connection.")
    exit(1)

goal="GBPUSD-OTC"
print("get candles")
try:
    candles = Iq.get_candles(goal,60,111,time.time())
    print(candles)
except Exception as e:
    print(f"Error getting candles: {e}")
    exit(1)
Money=1
ACTIVES="GBPUSD-OTC"
ACTION="call"#or "put"
expirations_mode=1

check, id = Iq.buy_digital_spot_v2(ACTIVES, Money, ACTION, expirations_mode)
if check:
    print("!buy! Digital Spot Trade ID:", id)
else:
    print("buy fail")
