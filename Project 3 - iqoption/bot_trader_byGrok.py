import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
import talib
from iqoptionapi.stable_api import IQ_Option
from datetime import datetime
import time
import logging
import os
from secret import secret
# from dotenv import load_dotenv  # Uncomment for secure credentials

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Check for talib
try:
    import talib
except ImportError:
    logging.error("TA-Lib not installed. Run: pip install TA-Lib and ensure the C library is installed.")
    exit()

# Credentials
EMAIL = secret.get('email')
PASSWORD = secret.get('password')

# Initialize IQ Option API
try:
    Iq = IQ_Option(EMAIL, PASSWORD)
    Iq.connect()
except Exception as e:
    logging.error(f"Failed to connect to IQ Option: {e}")
    exit()

# Switch to practice account
Iq.change_balance("PRACTICE")
if Iq.get_balance_mode() == "PRACTICE":
    logging.info("Connected to IQ Option PRACTICE account")
else:
    logging.error("Failed to switch to PRACTICE account")
    exit()

# Trading parameters
ACTIVES = "EURUSD-OTC"  # Primary asset for weekend trading
FALLBACK_ACTIVES = "EURUSD"  # Fallback for weekdays or if OTC fails
TIMEFRAME = "1m"  # 1-minute candles
DURATION = 1  # 1-minute trade duration
BASE_AMOUNT = 5  # Initial trade amount ($)
COEF = 2.18  # Martingale coefficient
USE_MARTINGALE = True  # Toggle Martingale (True) or fixed sizing (False)
MAX_RANGE = 30  # Maximum iterations
PIP_SIZE = 0.0001  # Pip size for EURUSD-OTC
MIN_PIP_GAIN = 19  # Minimum target gain in pips (for monitoring)
STOP_LOSS_PIPS = 10  # Stop-loss in pips
TAKE_PROFIT_PIPS = 19  # Take-profit in pips
MAX_DRAWDOWN = 0.2  # 20% max drawdown
MAX_RETRIES = 3  # Retry attempts for trade execution
record = []  # Store trade records

# Check available assets (only digital spot trading)
def check_available_assets():
    try:
        assets = Iq.get_all_open_time()
        logging.info(f"Raw API response for open assets: {assets}")
        digital_assets = assets.get('digital', {})
        if not digital_assets:
            logging.warning("No digital assets available in API response")
        if ACTIVES in digital_assets and digital_assets[ACTIVES].get('open', False):
            logging.info(f"{ACTIVES} is available for digital spot trading")
            return ACTIVES
        elif FALLBACK_ACTIVES in digital_assets and digital_assets[FALLBACK_ACTIVES].get('open', False):
            logging.warning(f"{ACTIVES} unavailable, falling back to {FALLBACK_ACTIVES} for digital spot trading")
            return FALLBACK_ACTIVES
        logging.error("Neither EURUSD-OTC nor EURUSD is available for digital spot trading")
        return None
    except Exception as e:
        logging.error(f"Failed to check available assets: {e}")
        logging.info("Attempting candle-based availability check")
        try:
            candles = Iq.get_candles(ACTIVES, 60, 1, time.time())
            if candles:
                logging.info(f"{ACTIVES} appears available based on candle data")
                return ACTIVES
        except Exception as e:
            logging.error(f"Candle check failed: {e}")
        return None

# Fetch historical data from IQ Option API
def fetch_historical_data(active):
    try:
        candles = Iq.get_candles(active, 60, 1440, time.time())  # 1440 minutes = 1 day
        data = pd.DataFrame(candles)
        data = data.rename(columns={'close': 'Close', 'open': 'Open', 'max': 'High', 'min': 'Low', 'volume': 'Volume'})
        data['Datetime'] = pd.to_datetime(data['from'], unit='s')
        data.set_index('Datetime', inplace=True)
        data = data[['Close', 'Open', 'High', 'Low', 'Volume']].dropna()
        if len(data) < 100:
            logging.error(f"Insufficient historical data for {active}: less than 100 candles")
            return None
        return data
    except Exception as e:
        logging.error(f"Failed to fetch historical data for {active}: {e}")
        return None

# Calculate technical indicators using talib
def calculate_indicators(data):
    data['SMA_short'] = talib.SMA(data['Close'], timeperiod=10)
    data['SMA_long'] = talib.SMA(data['Close'], timeperiod=50)
    data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
    macd, macd_signal, _ = talib.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    data['MACD'] = macd
    data['MACD_signal'] = macd_signal
    data['BB_upper'], data['BB_middle'], data['BB_lower'] = talib.BBANDS(data['Close'], timeperiod=20)
    return data

# Prepare features and labels for ML model
def prepare_ml_data(data):
    data = calculate_indicators(data)
    data['Price_Diff'] = data['Close'].shift(-1) - data['Close']
    data['Target'] = np.where(data['Price_Diff'] >= PIP_SIZE * MIN_PIP_GAIN, 1,  # Buy
                              np.where(data['Price_Diff'] <= -PIP_SIZE * MIN_PIP_GAIN, -1, 0))  # Sell
    features = ['SMA_short', 'SMA_long', 'RSI', 'MACD', 'MACD_signal', 'BB_upper', 'BB_middle', 'BB_lower']
    X = data[features].dropna()
    y = data['Target'].loc[X.index]
    if len(X) < 50:
        logging.error("Insufficient data for ML training: less than 50 samples")
        return None, None
    return X, y

# Train ML model
def train_model(X, y):
    try:
        tscv = TimeSeriesSplit(n_splits=5)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        accuracies = []
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracies.append(accuracy_score(y_test, y_pred))
        avg_accuracy = np.mean(accuracies)
        logging.info(f"Cross-validated accuracy: {avg_accuracy}")
        if avg_accuracy >= 0.99:
            logging.warning("High cross-validated accuracy (≥0.99) may indicate overfitting")
        model.fit(X, y)  # Train on full dataset for final model
        return model
    except Exception as e:
        logging.error(f"Failed to train model: {e}")
        return None

# Fetch real-time data
def fetch_realtime_data(active):
    try:
        candles = Iq.get_candles(active, 60, 100, time.time())
        data = pd.DataFrame(candles)
        data = data.rename(columns={'close': 'Close', 'open': 'Open', 'max': 'High', 'min': 'Low', 'volume': 'Volume'})
        data['Datetime'] = pd.to_datetime(data['from'], unit='s')
        data.set_index('Datetime', inplace=True)
        data = data[['Close', 'Open', 'High', 'Low', 'Volume']].dropna()
        if data.empty:
            logging.error(f"Real-time data for {active} is empty")
            return None
        return data
    except Exception as e:
        logging.error(f"Failed to fetch real-time data for {active}: {e}")
        return None

# Generate trading signal
def generate_signal(model, data):
    try:
        data = calculate_indicators(data)
        if data.empty or len(data) < 50:
            logging.error("Insufficient data for indicator calculation")
            return 0
        latest = data.iloc[-1]
        logging.info(f"Latest indicators: SMA_short={latest['SMA_short']}, SMA_long={latest['SMA_long']}, RSI={latest['RSI']}, MACD={latest['MACD']}, MACD_signal={latest['MACD_signal']}")

        # Relaxed rule-based conditions
        if (latest['SMA_short'] > latest['SMA_long'] and latest['RSI'] < 80):  # Relaxed RSI threshold for buy
            logging.info("Rule-based buy signal triggered")
            return 1
        elif (latest['SMA_short'] < latest['SMA_long'] and latest['RSI'] > 20):  # Relaxed RSI threshold for sell
            logging.info("Rule-based sell signal triggered")
            return -1
        else:
            logging.info("Rule-based conditions not met, falling back to ML model")
            features = ['SMA_short', 'SMA_long', 'RSI', 'MACD', 'MACD_signal', 'BB_upper', 'BB_middle', 'BB_lower']
            X = data[features].dropna()
            if X.empty:
                logging.warning("No valid features for ML prediction after dropna")
                # Fallback to trend-based signal if ML features are unavailable
                if latest['SMA_short'] > latest['SMA_long']:
                    logging.info("Fallback to trend-based buy signal")
                    return 1
                elif latest['SMA_short'] < latest['SMA_long']:
                    logging.info("Fallback to trend-based sell signal")
                    return -1
                logging.warning("No valid trend-based signal, returning neutral")
                return 0
            prediction = model.predict(X.iloc[-1:])
            logging.info(f"ML model prediction: {prediction[0]}")
            return prediction[0]
    except Exception as e:
        logging.error(f"Error generating signal: {e}")
        return 0

# Execute trade with retry logic (only digital spot trading)
def execute_trade(signal, amount, cycle, count, active):
    if signal == 1:
        action = "call"
    elif signal == -1:
        action = "put"
    else:
        logging.info(f"No trade signal ({signal}), skipping cycle")
        return None, None, None, None

    logging.info(f"Placing {action} digital spot trade with amount: {amount} on {active}")
    for attempt in range(MAX_RETRIES):
        try:
            result, id = Iq.buy_digital_spot_v2(active, amount, action, DURATION)
            if result:
                timestamp = datetime.fromtimestamp(Iq.get_server_timestamp())
                logging.info(f"Trade ID: {id}, Time Placed: {timestamp}")
                return action, id, amount, timestamp
            else:
                logging.error(f"Digital spot trade failed on attempt {attempt + 1}: API returned False, ID: {id}")
        except Exception as e:
            logging.error(f"Trade failed on attempt {attempt + 1} for {active}: {e}")
        time.sleep(2)  # Wait before retry
    logging.error(f"Trade failed after {MAX_RETRIES} attempts, skipping cycle")
    return None, None, None, None

# Monitor and close trade
def monitor_trade(id, action, entry_price, active):
    if not id or id == "error":
        logging.warning("Invalid trade ID, returning 0 profit")
        return 0
    target_price = entry_price + PIP_SIZE * TAKE_PROFIT_PIPS if action == "call" else entry_price - PIP_SIZE * TAKE_PROFIT_PIPS
    stop_loss = entry_price - PIP_SIZE * STOP_LOSS_PIPS if action == "call" else entry_price + PIP_SIZE * STOP_LOSS_PIPS
    start_time = time.time()
    max_attempts = 3

    while time.time() - start_time < 60:  # Monitor for 1 minute
        for attempt in range(max_attempts):
            try:
                candles = Iq.get_candles(active, 60, 1, time.time())
                current_price = candles[-1]['close']
                if action == "call":
                    if current_price >= target_price or current_price <= stop_loss:
                        check, win = Iq.check_win_digital_v2(id)
                        if check:
                            logging.info(f"Trade closed at {current_price}, Profit: {win}")
                            return win if win is not None else 0
                        else:
                            logging.warning(f"Trade check failed for ID {id}, assuming no profit")
                            return 0
                elif action == "put":
                    if current_price <= target_price or current_price >= stop_loss:
                        check, win = Iq.check_win_digital_v2(id)
                        if check:
                            logging.info(f"Trade closed at {current_price}, Profit: {win}")
                            return win if win is not None else 0
                        else:
                            logging.warning(f"Trade check failed for ID {id}, assuming no profit")
                            return 0
                time.sleep(1)
                break  # Exit attempt loop if successful
            except Exception as e:
                logging.error(f"Error monitoring trade on attempt {attempt + 1}: {e}")
                if attempt < max_attempts - 1:
                    logging.info("Attempting to reconnect to IQ Option API")
                    try:
                        Iq.connect()
                        time.sleep(2)
                    except Exception as reconn_err:
                        logging.error(f"Reconnection failed: {reconn_err}")
                time.sleep(2)

    # Final check after timeout
    for attempt in range(max_attempts):
        try:
            check, win = Iq.check_win_digital_v2(id)
            if check:
                logging.info(f"Trade closed due to timeout, Profit: {win}")
                return win if win is not None else 0
            else:
                logging.warning(f"Final trade check failed for ID {id}, assuming no profit")
                return 0
        except Exception as e:
            logging.error(f"Error checking trade result on attempt {attempt + 1}: {e}")
            if attempt < max_attempts - 1:
                logging.info("Attempting to reconnect to IQ Option API")
                try:
                    Iq.connect()
                    time.sleep(2)
                except Exception as reconn_err:
                    logging.error(f"Reconnection failed: {reconn_err}")
            time.sleep(2)
    logging.error(f"Failed to retrieve trade result for ID {id} after all attempts")
    return 0

# Main trading loop
def main():
    global MAX_RANGE  # Declare MAX_RANGE as global

    # Check available assets
    active = ACTIVES
    # active = check_available_assets()
    # if active is None:
    #     logging.error("Exiting due to unavailable assets")
    #     return

    # Train ML model
    data = fetch_historical_data(active)
    if data is None or data.empty:
        logging.error("Exiting due to data fetch failure or empty dataset")
        return
    X, y = prepare_ml_data(data)
    if X is None or y is None:
        logging.error("Exiting due to insufficient data for ML training")
        return
    model = train_model(X, y)
    if model is None:
        logging.error("Exiting due to model training failure")
        return

    # Initialize variables
    amount = BASE_AMOUNT
    cycle = 0
    count = -1
    initial_balance = Iq.get_balance()
    logging.info(f"Initial balance: {initial_balance}")

    try:
        while count < MAX_RANGE:
            count += 1
            cycle += 1
            logging.info(f"\nCycle: {cycle}, Iteration: {count}, Max Range: {MAX_RANGE}")
            logging.info(f"Current Balance Before Trade: {Iq.get_balance()}")

            # Check balance
            if Iq.get_balance() < amount:
                logging.warning(f"Balance {Iq.get_balance()} is less than required amount {amount}")
                break

            # Validate trade amount
            if amount < 1 or amount > Iq.get_balance():
                logging.error(f"Invalid trade amount: {amount}. Must be between 1 and balance ({Iq.get_balance()})")
                break

            # Fetch real-time data and generate signal
            data = fetch_realtime_data(active)
            if data is None or data.empty:
                logging.warning("Skipping cycle due to data fetch failure or empty dataset")
                time.sleep(60)
                continue
            current_price = data['Close'].iloc[-1]
            signal = generate_signal(model, data)
            logging.info(f"Signal is = {signal}")

            # Execute trade
            action, id, trade_amount, timestamp = execute_trade(signal, amount, cycle, count, active)
            if id and id != "error":
                while True:
                    # win = monitor_trade(id, action, current_price, active)
                    check,win=Iq.check_win_digital_v2(id)
                    if check==True:
                            break
                if USE_MARTINGALE:
                    if win < 0:
                        logging.info(f"Loss: {win}")
                        amount *= COEF  # Martingale: increase amount
                    else:
                        logging.info(f"Win: {win}")
                        amount = BASE_AMOUNT  # Reset amount after win
                else:
                    logging.info(f"Profit: {win}")
                    amount = BASE_AMOUNT  # Fixed sizing
            else:
                logging.error("Trade failed, skipping cycle")
                win = 0
                # # Try fallback asset
                # if active == ACTIVES:
                #     active = check_available_assets()
                #     if active is None:
                #         logging.error("Exiting due to unavailable assets")
                #         break
                #     logging.info(f"Switching to asset: {active}")
                #     data = fetch_historical_data(active)
                #     if data is None or data.empty:
                #         logging.error("Exiting due to fallback data fetch failure")
                #         break
                #     X, y = prepare_ml_data(data)
                #     if X is None or y is None:
                #         logging.error("Exiting due to insufficient data for ML training")
                #         break
                #     model = train_model(X, y)
                #     if model is None:
                #         logging.error("Exiting due to model training failure")
                #         break

            # Record trade
            record.append([cycle, count, MAX_RANGE, Iq.get_balance(), action, id, trade_amount, timestamp, win])

            # Extend max_range if last iteration is a loss (Martingale only)
            if USE_MARTINGALE and count == MAX_RANGE and win < 0:
                MAX_RANGE += 1
                logging.info("Extending max_range due to loss")

            # Check drawdown
            drawdown = (initial_balance - Iq.get_balance()) / initial_balance
            if drawdown > MAX_DRAWDOWN:
                logging.warning("Maximum drawdown reached. Stopping bot.")
                break

            # time.sleep(60)  # Wait for next candle

    except KeyboardInterrupt:
        logging.info("Loop interrupted by user")

    # Save trade log to Excel
    df = pd.DataFrame(record, columns=["Cycle", "Count", "Max Range", "Current Balance", "Action", "ID", "Amount Placed", "Timestamp", "Win"])
    current_time = datetime.now()
    current_time_str = current_time.strftime("%Y-%m-%d %H-%M")
    xlsx_file = f'Log/Saved_Log_{current_time_str}.xlsx'
    os.makedirs('Log', exist_ok=True)
    df.to_excel(xlsx_file, index=False)
    logging.info(f"Data saved to {xlsx_file}")
    print(df)

if __name__ == "__main__":
    main()