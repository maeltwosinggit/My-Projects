from iqoptionapi.stable_api import IQ_Option

from secret import secret

#logging.basicConfig(level=logging.DEBUG,format='%(asctime)s %(message)s')

email = secret.get('email')
password = secret.get('password')

Iq=IQ_Option(email,password)
Iq.connect()#connect to iqoption
balance_type="PRACTICE"
Iq.change_balance(balance_type)
print("Current Balance:",Iq.get_balance())