import numpy as np
import pandas as pd
import yfinance as yf
import talib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from get_finbert_sentiment import get_finbert_score

def get_stock_data(ticker, start, end):
    """
    Fetches historical stock data for a specified ticker symbol over a given date range using the yfinance library.

    Parameters:
        ticker (str): The stock symbol for which to retrieve data (e.g., "AAPL" for Apple Inc.).
        start (str): The start date for the historical data in the format 'YYYY-MM-DD'.
        end (str): The end date for the historical data in the format 'YYYY-MM-DD'.

    Returns:
        pandas.DataFrame: A DataFrame containing historical stock data including open, high, low, close, volume,
                          and adjusted close prices over the specified date range.

    Example:
        >>> data = get_stock_data("AAPL", "2020-01-01", "2020-12-31")
        >>> print(data.head())
    """
    data = yf.download(ticker, start=start, end=end)
    return data

# 2. Feature Engineerings
def add_technical_indicators(df, sentiment_score):
    df = df.reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df['Date'] = pd.to_datetime(df['Date'])
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
    features = ['SMA_10', 'SMA_50', 'RSI', 'MACD', 'MACD_signal', 'Volatility', "rolling_mean_score"]
    X = df[features]
    y = df['Target']
    # Using a time-aware train/test split (no shuffling)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test



def hyperparameter_tuning(X_train, y_train, model_grid_dict, scoring='accuracy', n_jobs=-1):
    """
    Performs hyperparameter tuning for multiple models using GridSearchCV.

    This function iterates over a dictionary of models and their associated parameter grids,
    performs grid search with cross-validation for each model, and prints the best parameters
    and best cross-validated accuracy. It returns a dictionary of the best estimators for each model.

    Parameters:
        X_train (array-like or pandas.DataFrame): Training data features.
        y_train (array-like or pandas.Series): Training data target.
        model_grid_dict (dict): A dictionary where keys are model names (str) and values are tuples 
                                of the form (model_instance, param_grid), where:
                                - model_instance: an untrained model instance (e.g., RandomForestClassifier()).
                                - param_grid: a dictionary specifying the parameter grid for that model.
        scoring (str, optional): A string representing the scoring metric to use (default is 'accuracy').
        n_jobs (int, optional): Number of jobs to run in parallel (default is -1, using all processors).

    Returns:
        dict: A dictionary where the keys are model names and the values are the best estimator objects
              found via grid search.
    
    Example:
        >>> from sklearn.model_selection import TimeSeriesSplit
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from xgboost import XGBClassifier
        >>>
        >>> model_grid = {
        ...     'RandomForest': (
        ...         RandomForestClassifier(random_state=42), 
        ...         {'n_estimators': [50, 100, 200],
        ...          'max_depth': [None, 5, 10],
        ...          'min_samples_split': [2, 5, 10]}
        ...     ),
        ...     'XGBoost': (
        ...         XGBClassifier(eval_metric='logloss', random_state=42), 
        ...         {'n_estimators': [50, 100, 200],
        ...          'max_depth': [3, 5, 7],
        ...          'learning_rate': [0.01, 0.1, 0.2]}
        ...     )
        ... }
        >>>
        >>> best_estimators = hyperparameter_tuning(X_train, y_train, model_grid)
        >>> print(best_estimators)
    """
    # Use TimeSeriesSplit if no cv provided
    cv = TimeSeriesSplit(n_splits=5)

    best_estimators = {}

    for name, (model, param_grid) in model_grid_dict.items():
        grid_search = GridSearchCV(model, param_grid=param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs)
        grid_search.fit(X_train, y_train)
        print(f"Best parameters for {name}: {grid_search.best_params_}")
        print(f"Best cross-validated {scoring} for {name}: {grid_search.best_score_:.4f}")
        best_estimators[name] = grid_search.best_estimator_

    return best_estimators



def train_and_evaluate(X_train, X_test, y_train, y_test, models):
    """
    Trains and evaluates multiple classification models on provided training and testing data.

    This function iterates over a dictionary of models, fits each model using the training data,
    predicts the target values on the test data, and then prints the model's performance metrics.
    The performance metrics include accuracy and a detailed classification report.

    Parameters:
        X_train (array-like or pandas.DataFrame): The features for training the models.
        X_test (array-like or pandas.DataFrame): The features for testing the models.
        y_train (array-like or pandas.Series): The target variable for training.
        y_test (array-like or pandas.Series): The target variable for testing.
        models (dict): A dictionary where keys are model names (str) and values are untrained model objects
                       that implement the `fit` and `predict` methods.

    Returns:
        None: This function prints the performance metrics for each model and does not return any value.
    """        
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f'\n{name} Model Performance:')
        print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
        print(classification_report(y_test, y_pred))