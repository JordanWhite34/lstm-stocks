import sys
print(sys.executable)
import tensorflow as tf
print(tf.__version__)
import urllib.request
import certifi
import ssl
import pandas as pd
import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

def fetch_stock_data(ticker, api_key):
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
    if df is None or df.empty:
        raise ValueError("No data available for preparation.")
    
    print(f"Preparing data. DataFrame shape: {df.shape}")
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['Close']])
    
    sequences = []
    labels = []
    
    for i in range(len(scaled_data) - sequence_length):
        sequences.append(scaled_data[i:i+sequence_length])
        labels.append(scaled_data[i+sequence_length])
    
    sequences = np.array(sequences)
    labels = np.array(labels)
    
    print(f"Prepared sequences shape: {sequences.shape}")
    print(f"Prepared labels shape: {labels.shape}")
    
    if len(sequences) == 0 or len(labels) == 0:
        raise ValueError("Not enough data to create sequences and labels.")
    
    return sequences, labels, scaler

def build_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def predict_next_day(model, sequences, scaler):
    if sequences.shape[0] == 0:
        raise ValueError("No sequences available for prediction.")
    
    last_sequence = sequences[-1].reshape(1, sequences.shape[1], sequences.shape[2])
    prediction = model.predict(last_sequence)
    return scaler.inverse_transform(prediction)[0, 0]

if __name__ == "__main__":
    ticker = 'AAPL'
    api_key = 'YOUR_API_KEY_HERE'  # Replace with your actual API key
    df = fetch_stock_data(ticker, api_key)
    
    if df is not None and not df.empty:
        sequences, labels, scaler = prepare_data(df)
        model = build_model((sequences.shape[1], sequences.shape[2]))
        model.fit(sequences, labels, epochs=5, batch_size=32, verbose=1)
        prediction = predict_next_day(model, sequences, scaler)
        print(f"Predicted next day close price for {ticker}: {prediction}")
    else:
        print("Failed to fetch or prepare data. Please check your API key and internet connection.")