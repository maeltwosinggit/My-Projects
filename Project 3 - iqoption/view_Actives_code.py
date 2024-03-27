from iqoptionapi.stable_api import IQ_Option
Iq=IQ_Option("ungkuadrian@gmail.com","Nanana123")
print("Connecting...")
check,reason=Iq.connect()#connect to iqoption
print(check, reason)
print(Iq.get_balance())
# print(Iq.reset_practice_balance())
print(Iq.get_balance()) 

print(Iq.get_all_ACTIVES_OPCODE())