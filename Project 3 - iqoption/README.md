## Dependencies
pip install -U https://github.com/iqoptionapi/iqoptionapi/archive/refs/heads/master.zip OR
pip install -U git+git://github.com/iqoptionapi/iqoptionapi.git (never tried this version yet)

## Reference Links
- https://github.com/iqoptionapi/iqoptionapi/tree/master
- https://lu-yi-hsun.github.io/iqoptionapi/
- https://iqoptionapi.github.io/iqoptionapi/en/

## Project Log
29/3/2024
- pushed my project to github
- seperated credentials to secrets.py
- renamed secrets.py to secret.py due to conflict with numpy

3/11/2023
- tried to visualize data in Power BI (no luck haha)
- objective candle data: to study when is the occurence of the most streak (i want to avoid that hour)
- added manual code break function (achievable by pressing: ctrl+c)
- added function to save the order log once break
-* try to add - when keyboard interrupt (ctrl+c) is pressed, check current order if win<0, trade again until win (things to consider for this fx: may backfire if we want to force stop (still can simply close the python process manually easily))

2/11/2023
- pulled candle data for 1000 minutes, and 10800 mins (1 week)
- facing issue plotting time series

31/10/2023
- testing without Iq.connect() in while loop - successful cycle 266 before not enough balance, however, it didnt trigger the not enough balance if statement. maybe because of the if statement position in the code.
-* try to get candle info - study on candle pattern - get also info on very small candle.


30/10/2023
- after set sleep screen to never, successfully run 31 cycles by using infinite while loop
- reason it breaks is due to multiple loss until not enough balance: initial balance during the time is 500$
- currently retesting with 1000$ balance
- Log Saved #4 - error place order at cycle 54, (note: ada usik minor things kat apps, which i dont think it will affect the script but who knows right?)
- 54 cycle final balance = 1127.99$ yahoo!
- retry cycle using EURUSD pair - result stuck at cycle 18
- need to try this line - Iq.get_all_init() - study the output. Why? see below note
- notice issue where placed order is fixed at 89% profit only eventhough apps shows 92%
- tiba2 order at 22:06 dapat pula 92% win (pelik haha) lepastu order at 22:09 jadi 89% balik lol
- retry cycle using EURUSD pair - result stuck at cycle 12 - rasanya sebab internet tak stable ni (my phone main ml (same network) pun ping tinggi dunno whether its related or not :P)
- retry survived until cycle 34
- above test contains Iq.connect() at every iteration with an average of 5-6 seconds of order placement after new candle

29/10/2023
- issue increment max range dah solve, check max range = count kena letak belah bawah
- current issue, connection issue: run script lepas iteration ke-6 dia sangkut, check iq option apps kata timeout.
- cuba untuk letak Iq.connect() kat bahagian while loop, tapi still sangkut kat iteration ke-6
- retest solution atas, dgn letak indicator print connection status
- cuba identify solution baru, set sleep screen to never (haha)


28/10/2023
- implement and run iqoption script
- partial ready. ada few bugs kena cater
- takleh guna for loop, sebab tak increment max range bila kita ada loss kat last iteration
- boleh guna while loop, solve issue atas

27/10/2023
- study on iqoption api for python