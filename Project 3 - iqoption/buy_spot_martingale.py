from iqoptionapi.stable_api import IQ_Option
# import logging
# import random
# import time

import pandas as pd
from datetime import datetime
from secret import secret

#logging.basicConfig(level=logging.DEBUG,format='%(asctime)s %(message)s')

email = secret.get('email')
password = secret.get('password')

Iq=IQ_Option(email,password)
Iq.connect()#connect to iqoption
ACTIVES="EURUSD" #EURUSD-OTC
duration=1#minute 1 or 5
amount=5
action="call"#put
coef = 2.18
cycle = 0

win = 1
max_range = 60
count = -1
record = []

try:
    while count < max_range:
        print("\n\nCheck Connection:",Iq.check_connect())
        #Iq.connect()#connect to iqoption #letak sementara untuk test connection, maybe akan issue delay time
        # count+=1    #comment out for infinite loop
        cycle+=1
        print("Cycle:", cycle)    
        print("iteration",count)
        print("current max range:", max_range)
        
        print("Current Balance Before Trade:",Iq.get_balance())

        
        
        #will break once not enough amount
        if Iq.get_balance()<amount:
            print("Balance", Iq.get_balance(),"is lesser than required amount", amount)
            break
        
        else:
            #martingale
            if win<0:
                amount = amount*coef
                print("Amount: ", amount)
                _,id=(Iq.buy_digital_spot(ACTIVES,amount,action,duration))
                print(_)
                print(id)
                print("Time Placed =", datetime.fromtimestamp(Iq.get_server_timestamp()))
                if id !="error":
                    while True:
                        check,win=Iq.check_win_digital_v2(id)
                        if check==True:
                            break
                    if win<0:
                        print("you loss "+str(win)+"$")
                        print("Current Balance After Trade:",Iq.get_balance())
                    else:
                        print("you win "+str(win)+"$")
                        print("Current Balance After Trade:",Iq.get_balance())
                else:
                    print("please try again")
            
            else:
                amount = 5
                print("Amount: ", amount)
                _,id=(Iq.buy_digital_spot(ACTIVES,amount,action,duration))
                print(_)
                print(id)
                print("Time Placed =", datetime.fromtimestamp(Iq.get_server_timestamp()))
                if id !="error":
                    while True:
                        check,win=Iq.check_win_digital_v2(id)
                        if check==True:
                            break
                    if win<0:
                        print("you loss "+str(win)+"$")
                        print("Current Balance After Trade:",Iq.get_balance())
                    else:
                        print("you win "+str(win)+"$")
                        print("Current Balance After Trade:",Iq.get_balance())
                else:
                    print("please try again")
        
        
        #prevent close at loss for last iteration
        if count == max_range:
            if win<0:
                max_range+=1

        record.append([cycle,count,max_range,Iq.get_balance(),_,id,amount,datetime.fromtimestamp(Iq.get_server_timestamp()),win])
        # record = recordappend+record


except KeyboardInterrupt:
    print('Loop interrupted')

df = pd.DataFrame(record, columns=["Cycle", "Count", "Max Range","Current Balance", "Placeholder", "ID","Amount Placed", "Timestamp", "Win"])
current_time = datetime.now()
current_time_str = current_time.strftime("%Y-%m-%d %H-%M")

xlsx_file = f'Log/Saved Log_{current_time_str}.xlsx'

df.to_excel(xlsx_file, index=False)

print(f'Data saved to {xlsx_file}')
print(df)
