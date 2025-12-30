"""
LSTM Stock Price Forecasting Tool
Main implementation file
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pickle
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class StockPredictor:
    """LSTM-based stock price prediction model"""
    
    def __init__(self, sequence_length=60, lstm_units=50, dropout_rate=0.2):
        """
        Initialize the stock predictor
        
        Args:
            sequence_length: Number of time steps to look back
            lstm_units: Number of LSTM units in each layer
            dropout_rate: Dropout rate for regularization
        """
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.history = None
        
    def fetch_data(self, ticker, start_date, end_date):
        """
        Fetch historical stock data from Yahoo Finance
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with stock data
        """
        print(f"Fetching data for {ticker}...")
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if data.empty:
                raise ValueError(f"No data found for {ticker}")
            print(f"Downloaded {len(data)} data points")
            return data
        except Exception as e:
            print(f"Error fetching data: {e}")
            raise
    
    def prepare_data(self, data, train_split=0.8):
        """
        Prepare data for LSTM training
        
        Args:
            data: DataFrame with stock data
            train_split: Ratio of training data
            
        Returns:
            X_train, y_train, X_test, y_test, scaled_data
        """
        # Use Close price
        prices = data['Close'].values.reshape(-1, 1)
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(prices)
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        
        # Reshape for LSTM [samples, time steps, features]
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        # Split into train and test
        split_idx = int(len(X) * train_split)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        return X_train, y_train, X_test, y_test, scaled_data
    
    def build_model(self):
        """Build LSTM model architecture"""
        model = Sequential([
            LSTM(units=self.lstm_units, return_sequences=True, 
                 input_shape=(self.sequence_length, 1)),
            Dropout(self.dropout_rate),
            
            LSTM(units=self.lstm_units, return_sequences=True),
            Dropout(self.dropout_rate),
            
            LSTM(units=self.lstm_units, return_sequences=False),
            Dropout(self.dropout_rate),
            
            Dense(units=25),
            Dense(units=1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error', 
                     metrics=['mae'])
        
        self.model = model
        print("Model built successfully")
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """
        Train the LSTM model
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
        
        # Callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=10, 
                                   restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, 
                                     patience=5, min_lr=0.00001)
        
        print("Training model...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        return self.history
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        predictions = self.model.predict(X, verbose=0)
        return self.scaler.inverse_transform(predictions)
    
    def forecast_future(self, data, days_ahead=30):
        """
        Forecast future stock prices
        
        Args:
            data: Historical data
            days_ahead: Number of days to forecast
            
        Returns:
            Array of forecasted prices
        """
        # Get last sequence_length days
        last_sequence = data['Close'].values[-self.sequence_length:]
        last_sequence_scaled = self.scaler.transform(last_sequence.reshape(-1, 1))
        
        forecasts = []
        current_sequence = last_sequence_scaled.copy()
        
        for _ in range(days_ahead):
            # Reshape for prediction
            X_input = current_sequence.reshape(1, self.sequence_length, 1)
            
            # Predict next value
            next_pred = self.model.predict(X_input, verbose=0)
            forecasts.append(next_pred[0, 0])
            
            # Update sequence
            current_sequence = np.append(current_sequence[1:], next_pred)
        
        # Inverse transform
        forecasts = np.array(forecasts).reshape(-1, 1)
        forecasts = self.scaler.inverse_transform(forecasts)
        
        return forecasts.flatten()
    
    def evaluate(self, y_true, y_pred):
        """Calculate evaluation metrics"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        }
    
    def save_model(self, filepath='stock_model.h5'):
        """Save model and scaler"""
        if self.model is None:
            raise ValueError("No model to save")
        
        self.model.save(filepath)
        with open(filepath.replace('.h5', '_scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='stock_model.h5'):
        """Load model and scaler"""
        self.model = keras.models.load_model(filepath)
        with open(filepath.replace('.h5', '_scaler.pkl'), 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"Model loaded from {filepath}")


def main():
    """Main execution function"""
    # Configuration
    TICKER = 'AAPL'
    START_DATE = '2020-01-01'
    END_DATE = datetime.now().strftime('%Y-%m-%d')
    SEQUENCE_LENGTH = 60
    EPOCHS = 50
    BATCH_SIZE = 32
    
    # Initialize predictor
    predictor = StockPredictor(sequence_length=SEQUENCE_LENGTH)
    
    # Fetch data
    data = predictor.fetch_data(TICKER, START_DATE, END_DATE)
    
    # Prepare data
    X_train, y_train, X_test, y_test, scaled_data = predictor.prepare_data(data)
    
    # Train model
    history = predictor.train(X_train, y_train, X_test, y_test, 
                             epochs=EPOCHS, batch_size=BATCH_SIZE)
    
    # Make predictions on test set
    test_predictions = predictor.predict(X_test)
    y_test_actual = predictor.scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Evaluate
    metrics = predictor.evaluate(y_test_actual, test_predictions)
    print("\nTest Set Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Forecast future
    future_predictions = predictor.forecast_future(data, days_ahead=30)
    print(f"\nForecasted prices for next 30 days:")
    print(future_predictions)
    
    # Save model
    predictor.save_model(f'{TICKER}_model.h5')
    
    # Save predictions for dashboard
    results = {
        'ticker': TICKER,
        'historical_data': data,
        'test_predictions': test_predictions.flatten(),
        'test_actual': y_test_actual.flatten(),
        'test_dates': data.index[-len(y_test):],
        'future_predictions': future_predictions,
        'metrics': metrics,
        'sequence_length': SEQUENCE_LENGTH
    }
    
    with open('predictions.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("\nTraining complete! Run dashboard.py to visualize results.")


if __name__ == '__main__':
    main()
