from iqoptionapi.stable_api import IQ_Option
from secrets import secrets

email = secrets.get('email')
password = secrets.get('password')

Iq=IQ_Option(email,password)
print("Connecting...")
check,reason=Iq.connect()#connect to iqoption
print(check, reason)
print(Iq.get_balance())
# print(Iq.reset_practice_balance())
print(Iq.get_balance()) 

print(Iq.get_all_ACTIVES_OPCODE())