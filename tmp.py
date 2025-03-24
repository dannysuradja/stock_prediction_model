import numpy as np
import pandas as pd
import yfinance as yf
import talib
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from utils import get_stock_data, add_technical_indicators, prepare_data, hyperparameter_tuning, train_and_evaluate
from get_finbert_sentiment import get_finbert_score, get_sentiment_table


# 1. Fetch Stock Data
def get_stock_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data

# 2. Feature Engineering

sentiment_score = pd.read_csv("sentiment_score.csv")

def add_technical_indicators(df, sentiment_score):
    # Ensure the stock data has a Date column
    df = df.reset_index()  # 'Date' becomes a column
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Compute technical indicators as before:
    close = df['Close'].values.ravel()
    df['SMA_10'] = talib.SMA(close, timeperiod=10)
    df['SMA_50'] = talib.SMA(close, timeperiod=50)
    df['RSI'] = talib.RSI(close, timeperiod=14)
    df['MACD'], df['MACD_signal'], _ = talib.MACD(close)
    df['Volatility'] = df['Close'].rolling(10).std()
    df['Returns'] = df['Close'].pct_change()
    df['Target'] = (df['Returns'] >= 0).astype(int)

    df.reset_index(inplace=True)
    df.columns = df.columns.get_level_values(0)

    # Convert 'Date' column in stock data to datetime

    df['Date'] = pd.to_datetime(df['Date'])
    sentiment_score["Date"] = pd.to_datetime(sentiment_score["Date"])
    #merge
    df = pd.merge(df, sentiment_score, on="Date", how="right")
    
    df["rolling_mean_score"] = df["rolling_mean_score"].fillna(0)


    return df.dropna()

# 3. Data Preparation
def prepare_data(df):
    features = ['SMA_10', 'SMA_50', 'RSI', 'MACD', 'MACD_signal', 'Volatility',"rolling_mean_score"]
    X = df[features]
    y = df['Target']
    # Using a time-aware train/test split (no shuffling)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

# 4. Hyperparameter Tuning
def hyperparameter_tuning(X_train, y_train):
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Grid for RandomForestClassifier
    rf_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10]
    }
    
    rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), 
                           param_grid=rf_param_grid, 
                           cv=tscv, 
                           scoring='accuracy',
                           n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    print("Best parameters for RandomForest:", rf_grid.best_params_)
    print("Best cross-validated accuracy (RandomForest):", rf_grid.best_score_)
    
    # Grid for XGBClassifier
    xgb_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2]
    }
    
    xgb_grid = GridSearchCV(XGBClassifier(eval_metric='logloss', random_state=42), 
                            param_grid=xgb_param_grid, 
                            cv=tscv, 
                            scoring='accuracy',
                            n_jobs=-1)
    xgb_grid.fit(X_train, y_train)
    print("Best parameters for XGBoost:", xgb_grid.best_params_)
    print("Best cross-validated accuracy (XGBoost):", xgb_grid.best_score_)
    
    return rf_grid.best_estimator_, xgb_grid.best_estimator_

# 5. Model Training & Evaluation (using tuned hyperparameters)
def train_and_evaluate(X_train, X_test, y_train, y_test, rf_model, xgb_model):
    models = {
        'RandomForest': rf_model,
        'XGBoost': xgb_model
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f'\n{name} Model Performance:')
        print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
        print(classification_report(y_test, y_pred))