import time
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
import talib
from iqoptionapi.stable_api import IQ_Option
import logging
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime
from secret import secret

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Check for talib
try:
    import talib
except ImportError:
    logging.error("TA-Lib not installed. Run: pip install TA-Lib and ensure the C library is installed.")
    exit()

# Check for imblearn
try:
    from imblearn.combine import SMOTEENN
    from imblearn.over_sampling import SMOTE
except ImportError:
    logging.error("imblearn not installed. Run: pip install imbalanced-learn")
    exit()

# Check for xgboost
try:
    from xgboost import XGBClassifier
except ImportError:
    logging.error("xgboost not installed. Run: pip install xgboost")
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

# Parameters
ACTIVES = "EURUSD-OTC"
PIP_SIZE = 0.0001
MIN_PIP_GAIN = 3
OUTPUT_DIR = "Model_Visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Fetch historical data
def fetch_historical_data(active):
    try:
        candles = Iq.get_candles(active, 60, 4320, time.time())  # 3 days of 1-minute candles
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

# Calculate technical indicators
def calculate_indicators(data):
    data['SMA_short'] = talib.SMA(data['Close'], timeperiod=10)
    data['SMA_long'] = talib.SMA(data['Close'], timeperiod=50)
    data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
    macd, macd_signal, _ = talib.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    data['MACD'] = macd
    data['MACD_signal'] = macd_signal
    data['BB_upper'], data['BB_middle'], data['BB_lower'] = talib.BBANDS(data['Close'], timeperiod=20)
    data['ATR'] = talib.ATR(data['High'], data['Low'], data['Close'], timeperiod=14)
    data['STOCH_k'], data['STOCH_d'] = talib.STOCH(data['High'], data['Low'], data['Close'],
                                                    fastk_period=14, slowk_period=3, slowd_period=3)
    data['MOM'] = talib.MOM(data['Close'], timeperiod=10)
    data['ADX'] = talib.ADX(data['High'], data['Low'], data['Close'], timeperiod=14)
    data['OBV'] = talib.OBV(data['Close'], data['Volume'])
    data['CCI'] = talib.CCI(data['High'], data['Low'], data['Close'], timeperiod=14)
    data['Close_lag1'] = data['Close'].shift(1)
    return data

# Prepare features and labels
def prepare_ml_data(data):
    data = calculate_indicators(data)
    data['Price_Diff'] = data['Close'].shift(-1) - data['Close']
    data['Target'] = np.where(data['Price_Diff'] >= PIP_SIZE * MIN_PIP_GAIN, 1,
                              np.where(data['Price_Diff'] <= -PIP_SIZE * MIN_PIP_GAIN, -1, 0))
    features = ['SMA_short', 'SMA_long', 'RSI', 'MACD', 'MACD_signal', 'BB_upper', 'BB_middle', 'BB_lower', 'ATR', 'STOCH_k', 'STOCH_d', 'MOM', 'ADX', 'OBV', 'CCI', 'Close_lag1']
    X = data[features].dropna()
    y = data['Target'].loc[X.index]
    if len(X) < 50:
        logging.error("Insufficient data for ML training: less than 50 samples")
        return None, None, None, None
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    logging.info(f"Class counts before resampling:\n{pd.Series(y).value_counts()}")
    return X_scaled, y, data.loc[X.index], scaler

# Train and tune model
def train_model(X, y):
    try:
        tscv = TimeSeriesSplit(n_splits=5)
        # Map labels to 0, 1, 2 for XGBoost
        label_mapping = {-1: 0, 0: 1, 1: 2}
        y_mapped = y.map(label_mapping)
        if y_mapped.isna().any():
            logging.error("NaN values found in y_mapped")
            return None, None, None
        
        # Calculate class weights
        class_counts = y_mapped.value_counts()
        total_samples = len(y_mapped)
        class_weights = {i: 2 * total_samples / (len(class_counts) * class_counts.get(i, 1)) for i in [0, 1, 2]}
        
        model = XGBClassifier(
            random_state=42,
            objective='multi:softprob',
            eval_metric='mlogloss',
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1
        )
        
        # Apply RFE for feature selection
        rfe = RFE(estimator=XGBClassifier(random_state=42), n_features_to_select=6)
        X_rfe = rfe.fit_transform(X, y_mapped)
        selected_features = X.columns[rfe.support_].tolist()
        feature_ranking = pd.DataFrame({'Feature': X.columns, 'Ranking': rfe.ranking_}).sort_values('Ranking')
        logging.info(f"Selected features: {selected_features}")
        logging.info(f"Feature ranking:\n{feature_ranking.to_string(index=False)}")
        X_selected = X[selected_features]
        logging.info(f"Samples after feature selection: {len(X_selected)}")
        
        # Custom cross-validation
        best_model = None
        best_f1_score = 0
        
        for i, (train_index, test_index) in enumerate(tscv.split(X_selected)):
            X_train, X_test = X_selected.iloc[train_index], X_selected.iloc[test_index]
            y_train, y_test = y_mapped.iloc[train_index], y_mapped.iloc[test_index]
            
            # Check for NaN in y_train and y_test
            if y_train.isna().any() or y_test.isna().any():
                logging.error(f"Fold {i+1}: NaN values found in y_train or y_test")
                return None, None, None
            
            # Log class counts
            logging.info(f"Fold {i+1} class counts:\n{pd.Series(y_train).map({0: 'Sell', 1: 'Neutral', 2: 'Buy'}).value_counts()}")
            
            # Dynamically adjust k_neighbors and try SMOTEENN
            min_samples = min(y_train.value_counts())
            k_neighbors = min(3, max(1, min_samples - 1)) if min_samples > 1 else 1
            majority_samples = max(y_train.value_counts())
            sampling_strategy = {0: majority_samples, 1: majority_samples, 2: majority_samples}
            smoteenn = SMOTEENN(
                random_state=42,
                smote=SMOTE(k_neighbors=k_neighbors, random_state=42, sampling_strategy=sampling_strategy)
            )
            
            try:
                X_train_resampled, y_train_resampled = smoteenn.fit_resample(X_train, y_train)
                logging.info(f"Fold {i+1}: Applied SMOTEENN, samples: {len(X_train_resampled)}, new class distribution:\n{pd.Series(y_train_resampled).map({0: 'Sell', 1: 'Neutral', 2: 'Buy'}).value_counts(normalize=True)}")
            except ValueError as e:
                logging.warning(f"SMOTEENN failed for fold {i+1}: {e}, trying SMOTE")
                smote = SMOTE(k_neighbors=k_neighbors, random_state=42, sampling_strategy=sampling_strategy)
                try:
                    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
                    logging.info(f"Fold {i+1}: Applied SMOTE, samples: {len(X_train_resampled)}, new class distribution:\n{pd.Series(y_train_resampled).map({0: 'Sell', 1: 'Neutral', 2: 'Buy'}).value_counts(normalize=True)}")
                except ValueError as e:
                    logging.warning(f"SMOTE failed for fold {i+1}: {e}, proceeding without resampling")
                    X_train_resampled, y_train_resampled = X_train, y_train
            
            # Train model
            model = XGBClassifier(
                random_state=42,
                objective='multi:softprob',
                eval_metric='mlogloss',
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1
            )
            sample_weights = [class_weights[y] for y in y_train_resampled]
            model.fit(X_train_resampled, y_train_resampled, sample_weight=sample_weights)
            
            # Evaluate with custom threshold
            probs = model.predict_proba(X_test)
            custom_threshold = 0.25
            y_pred = np.argmax(np.where(probs >= custom_threshold, probs, -np.inf), axis=1)
            y_pred = np.where(np.max(probs, axis=1) < custom_threshold, 1, y_pred)  # Default to Neutral
            logging.info(f"Fold {i+1} probability stats: Sell={np.mean(probs[:, 0]):.4f}, Neutral={np.mean(probs[:, 1]):.4f}, Buy={np.mean(probs[:, 2]):.4f}")
            
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
            logging.info(f"Fold {i+1} metrics:")
            for label, p, r, f in zip(['Sell', 'Neutral', 'Buy'], precision, recall, f1):
                logging.info(f"  {label}: Precision={p:.4f}, Recall={r:.4f}, F1={f:.4f}")
            
            # Log confusion matrix and prediction counts
            cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
            logging.info(f"Fold {i+1} confusion matrix:\n{cm}")
            pred_counts = pd.Series(y_pred).map({0: 'Sell', 1: 'Neutral', 2: 'Buy'}).value_counts()
            logging.info(f"Fold {i+1} prediction counts:\n{pred_counts}")
            
            # Update best model
            weighted_f1 = np.mean(f1)
            if weighted_f1 > best_f1_score:
                best_f1_score = weighted_f1
                best_model = model
        
        logging.info(f"Best cross-validated F1-score: {best_f1_score:.4f}")
        
        # Train final model with SMOTEENN
        min_samples = min(y_mapped.value_counts())
        k_neighbors = min(3, max(1, min_samples - 1)) if min_samples > 1 else 1
        majority_samples = max(y_mapped.value_counts())
        sampling_strategy = {0: majority_samples, 1: majority_samples, 2: majority_samples}
        smoteenn = SMOTEENN(
            random_state=42,
            smote=SMOTE(k_neighbors=k_neighbors, random_state=42, sampling_strategy=sampling_strategy)
        )
        try:
            X_resampled, y_resampled = smoteenn.fit_resample(X_selected, y_mapped)
            logging.info(f"Final model: Applied SMOTEENN, samples: {len(X_resampled)}, new class distribution:\n{pd.Series(y_resampled).map({0: 'Sell', 1: 'Neutral', 2: 'Buy'}).value_counts(normalize=True)}")
        except ValueError as e:
            logging.warning(f"SMOTEENN failed for final model: {e}, trying SMOTE")
            smote = SMOTE(k_neighbors=k_neighbors, random_state=42, sampling_strategy=sampling_strategy)
            try:
                X_resampled, y_resampled = smote.fit_resample(X_selected, y_mapped)
                logging.info(f"Final model: Applied SMOTE, samples: {len(X_resampled)}, new class distribution:\n{pd.Series(y_resampled).map({0: 'Sell', 1: 'Neutral', 2: 'Buy'}).value_counts(normalize=True)}")
            except ValueError as e:
                logging.warning(f"SMOTE failed for final model: {e}, training without resampling")
                X_resampled, y_resampled = X_selected, y_mapped
        
        sample_weights = [class_weights[y] for y in y_resampled]
        best_model.fit(X_resampled, y_resampled, sample_weight=sample_weights)
        
        # Log feature importance
        feature_importance = pd.DataFrame({
            'Feature': selected_features,
            'Importance': best_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        logging.info(f"Feature importance:\n{feature_importance.to_string(index=False)}")
        
        return best_model, label_mapping, selected_features
    except Exception as e:
        logging.error(f"Failed to train model: {e}")
        return None, None, None

# Visualize model performance
def visualize_performance(model, X_train, y_train, X_test, y_test, data_test, y_full, label_mapping, selected_features):
    try:
        # Check for NaN in inputs
        logging.info(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
        logging.info(f"y_train NaN count: {y_train.isna().sum()}, y_test NaN count: {y_test.isna().sum()}")
        logging.info(f"y_full NaN count: {y_full.isna().sum()}")
        
        # Map labels for visualization
        reverse_mapping = {0: -1, 1: 0, 2: 1}
        y_train_mapped = y_train.map(reverse_mapping)
        y_test_mapped = y_test.map(reverse_mapping)
        y_full_mapped = y_full.map(reverse_mapping)
        
        if y_train_mapped.isna().any() or y_test_mapped.isna().any() or y_full_mapped.isna().any():
            logging.error("NaN values found after label mapping")
            return
        
        # Adjust X_train and X_test for selected features
        X_train = X_train[selected_features]
        X_test = X_test[selected_features]
        
        # Train and test predictions with custom threshold
        probs_train = model.predict_proba(X_train)
        probs_test = model.predict_proba(X_test)
        custom_threshold = 0.25
        y_train_pred = np.argmax(np.where(probs_train >= custom_threshold, probs_train, -np.inf), axis=1)
        y_train_pred = np.where(np.max(probs_train, axis=1) < custom_threshold, 1, y_train_pred)
        y_test_pred = np.argmax(np.where(probs_test >= custom_threshold, probs_test, -np.inf), axis=1)
        y_test_pred = np.where(np.max(probs_test, axis=1) < custom_threshold, 1, y_test_pred)
        y_train_pred_mapped = pd.Series(y_train_pred).map(reverse_mapping)
        y_test_pred_mapped = pd.Series(y_test_pred).map(reverse_mapping)
        
        if y_train_pred_mapped.isna().any() or y_test_pred_mapped.isna().any():
            logging.error("NaN values found in predicted labels")
            return
        
        # Log prediction counts
        train_pred_counts = y_train_pred_mapped.value_counts()
        test_pred_counts = y_test_pred_mapped.value_counts()
        logging.info(f"Train prediction counts:\n{train_pred_counts}")
        logging.info(f"Test prediction counts:\n{test_pred_counts}")
        
        # Class distribution plot
        class_dist = pd.Series(y_full_mapped).value_counts(normalize=True).reset_index()
        class_dist = class_dist.rename(columns={'index': 'Target', 'proportion': 'Proportion'})
        class_dist['Target'] = class_dist['Target'].map({-1: 'Sell', 0: 'Neutral', 1: 'Buy'})
        fig_dist = px.bar(class_dist, x='Target', y='Proportion', title='Target Class Distribution',
                          labels={'Target': 'Class', 'Proportion': 'Proportion'},
                          color='Proportion', color_continuous_scale='Blues')
        
        # Confusion matrix
        cm = confusion_matrix(y_test_mapped, y_test_pred_mapped, labels=[-1, 0, 1])
        fig_cm = px.imshow(cm, x=['Sell', 'Neutral', 'Buy'], y=['Sell', 'Neutral', 'Buy'],
                           title='Confusion Matrix (Test Set)', text_auto=True,
                           color_continuous_scale='Blues')
        fig_cm.update_layout(xaxis_title='Predicted', yaxis_title='Actual')
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': selected_features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        fig_fi = px.bar(feature_importance, x='Feature', y='Importance',
                        title='Feature Importance', color='Importance')
        
        # Classification metrics
        precision, recall, f1, _ = precision_recall_fscore_support(y_test_mapped, y_test_pred_mapped, average=None, labels=[-1, 0, 1], zero_division=0)
        metrics_df = pd.DataFrame({
            'Class': ['Sell', 'Neutral', 'Buy'],
            'Precision': precision,
            'Recall': recall,
            'F1': f1
        })
        fig_metrics = make_subplots(rows=1, cols=3, subplot_titles=['Precision', 'Recall', 'F1-Score'])
        for i, metric in enumerate(['Precision', 'Recall', 'F1']):
            fig_metrics.add_trace(go.Bar(x=metrics_df['Class'], y=metrics_df[metric], name=metric),
                                 row=1, col=i+1)
        fig_metrics.update_layout(title_text='Classification Metrics (Test Set)')
        
        # Time-series signal plot with probabilities
        signals_df = pd.DataFrame({
            'Datetime': data_test.index,
            'Actual': y_test_mapped,
            'Predicted': y_test_pred_mapped,
            'Prob_Sell': probs_test[:, 0],
            'Prob_Neutral': probs_test[:, 1],
            'Prob_Buy': probs_test[:, 2]
        })
        fig_signals = go.Figure()
        fig_signals.add_trace(go.Scatter(x=signals_df['Datetime'], y=signals_df['Actual'],
                                        mode='lines+markers', name='Actual Signal',
                                        line=dict(color='blue')))
        fig_signals.add_trace(go.Scatter(x=signals_df['Datetime'], y=signals_df['Predicted'],
                                        mode='lines+markers', name='Predicted Signal',
                                        line=dict(color='red', dash='dash')))
        fig_signals.add_trace(go.Scatter(x=signals_df['Datetime'], y=signals_df['Prob_Buy'],
                                        mode='lines', name='Buy Probability',
                                        line=dict(color='green', dash='dot')))
        fig_signals.add_trace(go.Scatter(x=signals_df['Datetime'], y=signals_df['Prob_Sell'],
                                        mode='lines', name='Sell Probability',
                                        line=dict(color='purple', dash='dot')))
        fig_signals.update_layout(title='Actual vs Predicted Signals and Probabilities Over Time',
                                 xaxis_title='Time', yaxis_title='Signal / Probability',
                                 yaxis=dict(tickvals=[-1, 0, 1], ticktext=['Sell', 'Neutral', 'Buy']))
        
        # Save plots
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        fig_dist.write_html(os.path.join(OUTPUT_DIR, f'class_distribution_{timestamp}.html'))
        fig_cm.write_html(os.path.join(OUTPUT_DIR, f'confusion_matrix_{timestamp}.html'))
        fig_fi.write_html(os.path.join(OUTPUT_DIR, f'feature_importance_{timestamp}.html'))
        fig_metrics.write_html(os.path.join(OUTPUT_DIR, f'metrics_{timestamp}.html'))
        fig_signals.write_html(os.path.join(OUTPUT_DIR, f'signals_{timestamp}.html'))
        
        logging.info(f"Visualizations saved in {OUTPUT_DIR}")
        
        # Log test set accuracy
        test_accuracy = accuracy_score(y_test_mapped, y_test_pred_mapped)
        logging.info(f"Test set accuracy: {test_accuracy:.4f}")
        
    except Exception as e:
        logging.error(f"Failed to visualize performance: {e}")

# Main function
def main():
    # Fetch and prepare data
    data = fetch_historical_data(ACTIVES)
    if data is None:
        return
    X, y, data_aligned, scaler = prepare_ml_data(data)
    if X is None or y is None:
        return
    
    # Log class distribution
    logging.info(f"Target distribution:\n{pd.Series(y).value_counts(normalize=True)}")
    
    # Time-series train-test split (80% train, 20% test)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    data_test = data_aligned.iloc[train_size:]
    
    # Train model
    model, label_mapping, selected_features = train_model(X_train, y_train)
    if model is None:
        return
    
    # Visualize performance
    visualize_performance(model, X_train, y_train, X_test, y_test, data_test, y, label_mapping, selected_features)

if __name__ == "__main__":
    main()