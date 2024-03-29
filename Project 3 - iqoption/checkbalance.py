from iqoptionapi.stable_api import IQ_Option

from secret import secret

#logging.basicConfig(level=logging.DEBUG,format='%(asctime)s %(message)s')

email = secret.get('email')
password = secret.get('password')

Iq=IQ_Option(email,password)
Iq.connect()#connect to iqoption
print("Current Balance:",Iq.get_balance())