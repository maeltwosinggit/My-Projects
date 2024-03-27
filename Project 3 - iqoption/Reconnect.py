from iqoptionapi.stable_api import IQ_Option
import time
error_password="""{"code":"invalid_credentials","message":"You entered the wrong credentials. Please check that the login/password is correct."}"""
iqoption = IQ_Option("ungkuadrian@gmail.com", "Nanana123")
check,reason=iqoption.connect()

#macam tak jadi, lepas tutup wifi still tak masuk loop if iqoption.check_connect()==False

if check:
    print("Start your robot")
    #if see this you can close network for test
    while True:
        print("\nrecheck connection.")
        time.sleep(1)
        print("recheck connection...")
        if iqoption.check_connect()==False:#detect the websocket is close
            print("try reconnect")
            check,reason=iqoption.connect()
            if check:
                print("Reconnect successfully")
            else:   
                if reason==error_password:
                    print("Error Password")
                else:
                    print("No Network")

else:

    if reason=="[Errno -2] Name or service not known":
        print("No Network")
    elif reason==error_password:
        print("Error Password")