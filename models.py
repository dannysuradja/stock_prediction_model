import numpy as np
import pandas as pd
import yfinance as yf
import talib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Import our utility functions from utils.py
from utils import get_stock_data, add_technical_indicators, prepare_data, hyperparameter_tuning, train_and_evaluate

# --------------------------------------------------
# Classical Machine Learning Pipeline
# --------------------------------------------------
sentiment_score = pd.read_csv("Stock_Price_Prediction\sentiment_score.csv")
def run_classical_pipeline():
    print("Running Classical Machine Learning Pipeline")
    data = get_stock_data('AAPL', '2014-01-01', '2023-12-01')
    data = add_technical_indicators(data, sentiment_score)
    
    # prepare_data() returns features:
    # ['SMA_10', 'SMA_50', 'RSI', 'MACD', 'MACD_signal', 'Volatility', 'rolling_mean_score']
    X_train, X_test, y_train, y_test = prepare_data(data)
    
    # Define classical ML models
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    models = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42)
    }
    
    train_and_evaluate(X_train, X_test, y_train, y_test, models)

# --------------------------------------------------
# TensorFlow (Keras) Pipeline
# --------------------------------------------------
def run_tensorflow_pipeline():
    print("Running TensorFlow Pipeline")
    
    # Load and preprocess data using utils
    data = get_stock_data('AAPL', '2014-01-01', '2023-12-01')
    df = add_technical_indicators(data, sentiment_score)
    
    # Ensure numeric returns and define target (Returns > 0)
    df = df[pd.to_numeric(df['Returns'], errors='coerce').notnull()]
    df['Returns'] = df['Returns'].astype(float)
    df['Target'] = (df['Returns'] > 0).astype(int)
    
    # Select features and fill missing values. Note that here we include 'Returns'
    features = ['SMA_10', 'SMA_50', 'RSI', 'MACD', 'Volatility', 'Returns','rolling_mean_score']
    X = df[features].fillna(0)
    y = df['Target']
    
    # Split and standardize
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    #Build Tensorflow Keras model
    model = Sequential([
        Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the model
    history = model.fit(X_train_scaled, y_train, epochs=20, batch_size=32, validation_split=0.2)
    
    # Evaluate the model
    loss, accuracy = model.evaluate(X_test_scaled, y_test)
    print("TensorFlow Test Accuracy:", accuracy)
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], marker='o', label='Train Loss')
    plt.plot(history.history['val_loss'], marker='o', label='Validation Loss')
    plt.title('TensorFlow Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Binary Crossentropy Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], marker='o', label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], marker='o', label='Validation Accuracy')
    plt.title('TensorFlow Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# --------------------------------------------------
# PyTorch Pipeline
# --------------------------------------------------
def run_pytorch_pipeline():
    print("Running PyTorch Pipeline")
    
    # Load and preprocess data using utils
    data = get_stock_data('AAPL', '2014-01-01', '2023-12-01')
    df = add_technical_indicators(data, sentiment_score)
    
    # Ensure numeric returns and define target (Returns > 0)
    df = df[pd.to_numeric(df['Returns'], errors='coerce').notnull()]
    df['Returns'] = df['Returns'].astype(float)
    df['Target'] = (df['Returns'] > 0).astype(int)
    
    # Select features and fill missing values
    features = ['SMA_10', 'SMA_50', 'RSI', 'MACD', 'Volatility', 'Returns','rolling_mean_score']
    X = df[features].fillna(0).values
    y = df['Target'].values
    
    # Split and standardize
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    
    # Create DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Define the PyTorch model
    class StockClassifier(nn.Module):
        def __init__(self, input_dim):
            super(StockClassifier, self).__init__()
            self.fc1 = nn.Linear(input_dim, 32)
            self.fc2 = nn.Linear(32, 16)
            self.fc3 = nn.Linear(16, 1)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
        
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.sigmoid(self.fc3(x))
            return x
    
    model = StockClassifier(X_train.shape[1])
    
    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 20
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    
    # Training loop with validation
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * X_batch.size(0)
            predicted = (outputs > 0.5).float()
            correct += (predicted.eq(y_batch).sum().item())
            total += y_batch.size(0)
        
        train_loss = running_loss / total
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_running_loss += loss.item() * X_batch.size(0)
                predicted = (outputs > 0.5).float()
                val_correct += (predicted.eq(y_batch).sum().item())
                val_total += y_batch.size(0)
        val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs+1), train_losses, marker='o', label='Train Loss')
    plt.plot(range(1, num_epochs+1), val_losses, marker='o', label='Validation Loss')
    plt.title('PyTorch Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Binary Cross Entropy Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs+1), train_accuracies, marker='o', label='Train Accuracy')
    plt.plot(range(1, num_epochs+1), val_accuracies, marker='o', label='Validation Accuracy')
    plt.title('PyTorch Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()