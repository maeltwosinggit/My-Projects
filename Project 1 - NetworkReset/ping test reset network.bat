@echo off
    cls
    setlocal enabledelayedexpansion
    REM Configuration:
    set SERVER=google.com
    set TIMEOUT_AFTER_PING_FAIL_SECONDS=5
    set TIMEOUT_AFTER_PING_SUCCEED_SECONDS=3
    set TIMEOUT_AFTER_LINK_DOWN_SECONDS=15
    set DECLARE_LINK_DOWN_FAILS=5

    set CONSECUTIVE_FAIL_COUNT=0
    :Start
    set PING_RESULT=Failure
    for /f "delims=" %%X in ('ping /n 1 %SERVER%') do (
       set TEMPVAR=%%X
       if "Reply from"=="!TEMPVAR:~0,10!" set PING_RESULT=Success
       )
    goto:!PING_RESULT!

    :Success
    echo Ping Succeeded
    set CONSECUTIVE_FAIL_COUNT=0
    call:Sleep %TIMEOUT_AFTER_PING_SUCCEED_SECONDS%
    goto:Start

    :Failure
    set /A CONSECUTIVE_FAIL_COUNT+=1
    echo Ping Failed !CONSECUTIVE_FAIL_COUNT! Time(s)
    netsh interface set interface "Wi-Fi" disabled
    timeout /t 2 /nobreak
    netsh interface set interface "Wi-Fi" enabled
    if !CONSECUTIVE_FAIL_COUNT!==%DECLARE_LINK_DOWN_FAILS% (call:LinkDownHandler&goto:Start)
    call:Sleep %TIMEOUT_AFTER_PING_FAIL_SECONDS%
    goto:Start

    :Sleep
    REM See http://stackoverflow.com/questions/4317020/windows-batch-sleep
    setlocal
    set /A ITERATIONS=%1+1
    ping -n %ITERATIONS% 127.0.0.1 >nul
    goto:eof

    :LinkDownHandler
    echo Link is Down
    set CONSECUTIVE_FAIL_COUNT=0
    REM Add additional link-down handler actions here
    call:Sleep %TIMEOUT_AFTER_LINK_DOWN_SECONDS%
    goto:eof