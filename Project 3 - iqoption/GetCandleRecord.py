import pandas as pd
import time
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

end_from_time=time.time()
ANS=[]
for i in range(10800):
    data=Iq.get_candles("EURUSD", 60, 1, end_from_time)
    ANS =data+ANS
    end_from_time=int(data[0]["from"])-1
print(ANS)

df = pd.DataFrame(ANS)

xlsx_file = 'outputweek.xlsx'

df.to_excel(xlsx_file, index=False)

print(f'Data saved to {xlsx_file}')
