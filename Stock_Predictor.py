import sys
print(sys.executable)
import tensorflow as tf
print(tf.__version__)
import urllib.request
import certifi
import ssl
from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import json
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

def fetch_stock_data(ticker, api_key):
    # api_key = 'LQD7WYTN73RBIKTB'  # Replace with your Alpha Vantage API key
    url_string = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&outputsize=full&apikey={api_key}"
    
    context = ssl.create_default_context(cafile=certifi.where())
    
    try:
        with urllib.request.urlopen(url_string, context=context) as url:
            data = json.loads(url.read().decode())
        
        time_series = data.get('Time Series (Daily)')
        if not time_series:
            raise ValueError(f"No time series data found for ticker '{ticker}'.")

        df = pd.DataFrame.from_dict(time_series, orient='index')
        df = df.rename(columns={
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. volume': 'Volume'
        })
        df = df.astype(float)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df

    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

def prepare_data(df, sequence_length=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['Close']])
    
    sequences = []
    labels = []
    
    for i in range(sequence_length, len(scaled_data)):
        sequences.append(scaled_data[i-sequence_length:i, 0])
        labels.append(scaled_data[i, 0])
    
    sequences = np.array(sequences)
    labels = np.array(labels)
    
    return sequences.reshape(-1, sequence_length, 1), labels, scaler

def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(units=100))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def predict_next_day(model, sequences, scaler):
    last_sequence = sequences[-1].reshape(1, sequences.shape[1], 1)
    prediction = model.predict(last_sequence)
    return scaler.inverse_transform(prediction)[0, 0]

if __name__ == "__main__":
    ticker = 'AAPL'
    df = fetch_stock_data(ticker)
    sequences, labels, scaler = prepare_data(df)
    model = build_model((sequences.shape[1], 1))
    model.fit(sequences, labels, epochs=5, batch_size=32, verbose=1)
    prediction = predict_next_day(model, sequences, scaler)
    print(f"Predicted next day close price for {ticker}: {prediction}")