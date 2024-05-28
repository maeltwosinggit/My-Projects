import pandas as pd
import time
from iqoptionapi.stable_api import IQ_Option
from secret import secret

email = secret.get('email')
password = secret.get('password')

Iq=IQ_Option(email,password)
print("Connecting...")
check,reason=Iq.connect()#connect to iqoption
print(check, reason)
print(Iq.get_balance())
# print(Iq.reset_practice_balance())
print(Iq.get_balance()) 

# end_from_time=time.time()
# ANS=[]
# for i in range(10800):
#     data=Iq.get_candles("EURUSD=OTC", 60, 1, end_from_time)
#     ANS =data+ANS
#     end_from_time=int(data[0]["from"])-1
#     print(ANS)
# print(ANS)

end_from_time = time.time()
ANS = []

for i in range(10800):
    try:
        data = Iq.get_candles("EURUSD-OTC", 60, 1, end_from_time)
        if data:
            ANS.append(data[0])
            end_from_time = int(data[0]["from"]) - 1
        else:
            print("No data received, retrying...")
            time.sleep(1)  # Retry after a short delay
    except Exception as e:
        print(f"Error fetching data at iteration {i}: {e}")
        time.sleep(5)  # Longer delay before retrying after an error

        # Attempt to reconnect if necessary
        if "need reconnect" in str(e).lower():
            print("Reconnecting...")
            Iq=IQ_Option(email,password)
            check,reason=Iq.connect()#connect to iqoption
            if check != "True":
                print("Reconnection failed, exiting.")
                break

    # if i % 100 == 0:
    print(f"Collected {len(ANS)} candles so far")

    # Save progress intermittently
    df = pd.DataFrame(ANS)
    df.to_excel('intermediate_output.xlsx', index=False)
    print("Intermediate data saved to 'intermediate_output.xlsx'")

# Save the final collected data
if ANS:
    df = pd.DataFrame(ANS)
    xlsx_file = 'outputweek.xlsx'
    df.to_excel(xlsx_file, index=False)
    print(f"Data saved to {xlsx_file}")
else:
    print("No data collected.")