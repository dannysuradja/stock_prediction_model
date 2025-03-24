import numpy as np
import pandas as pd
import yfinance as yf
import talib
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from models import run_classical_pipeline, run_pytorch_pipeline, run_tensorflow_pipeline


if __name__ == '__main__':
    print("Select a pipeline to run:")
    print("1. Classical ML Pipeline")
    print("2. TensorFlow Pipeline")
    print("3. PyTorch Pipeline")
    
    choice = input("Enter your choice (1/2/3): ")
    
    if choice == '1':
        run_classical_pipeline()
    elif choice == '2':
        run_tensorflow_pipeline()
    elif choice == '3':
        run_pytorch_pipeline()
    else:
        print("Invalid choice. Please enter 1, 2, or 3.")