import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QProgressBar, QLabel, QLineEdit, QPushButton, QTextEdit, 
                             QComboBox, QStatusBar)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from datetime import datetime, timedelta
from Stock_Predictor import fetch_stock_data, prepare_data, build_model, predict_next_day

class PredictionThread(QThread):
    finished = pyqtSignal(float)
    progress = pyqtSignal(int)  # Emit an integer for progress

    def __init__(self, ticker, df, sequences, labels, scaler):
        QThread.__init__(self)
        self.ticker = ticker
        self.df = df
        self.sequences = sequences
        self.labels = labels
        self.scaler = scaler

    def run(self):
        self.progress.emit(10)
        model = build_model((self.sequences.shape[1], 1))
        self.progress.emit(30)
        model.fit(self.sequences, self.labels, epochs=25, batch_size=32, verbose=1)
        self.progress.emit(80)
        prediction = predict_next_day(model, self.sequences, self.scaler)
        self.progress.emit(100)
        self.finished.emit(prediction)

class StockPredictionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Stock Price Prediction')
        self.setGeometry(100, 100, 800, 650)

        # Add API key input
        self.api_key_input = QLineEdit(self)
        self.api_key_input.setPlaceholderText('Enter Alpha Vantage API Key')
        self.api_key_input.setEchoMode(QLineEdit.Password)  # Hide the API key

        self.ticker_input = QLineEdit(self)
        self.ticker_input.setPlaceholderText('Enter Stock Ticker')
        
        self.predict_button = QPushButton('Predict', self)
        self.predict_button.clicked.connect(self.fetch_data_and_plot)
        
        self.time_frame = QComboBox(self)
        self.time_frame.addItems(['1 Month', '3 Month', '6 Month', '1 Year'])

        self.results_text = QTextEdit(self)
        self.results_text.setReadOnly(True)
        
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)  # Initially hidden

        layout = QVBoxLayout()
        layout.addWidget(self.api_key_input)
        layout.addWidget(self.ticker_input)
        layout.addWidget(self.time_frame)
        layout.addWidget(self.predict_button)
        layout.addWidget(self.results_text)
        layout.addWidget(self.progress_bar)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Graph section
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.df = None
        self.sequences = None
        self.scaler = None
        self.prediction = None

    def fetch_data_and_plot(self):
        ticker = self.ticker_input.text().upper()
        api_key = self.api_key_input.text()

        if not ticker or not api_key:
            self.results_text.setText("Please enter a valid ticker symbol and API key.")
            return

        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)

            # Fetch and prepare data
            self.status_bar.showMessage("Fetching stock data...")
            self.df = fetch_stock_data(ticker)
            
            if self.df is None:
                raise ValueError(f"No data could be fetched for ticker '{ticker}'. Please check the ticker symbol or API key.")

            self.status_bar.showMessage("Preparing data...")
            self.sequences, self.labels, self.scaler = prepare_data(self.df, sequence_length=60)

            # Plot current data based on selected time frame
            self.status_bar.showMessage("Plotting current data...")
            time_frame = self.time_frame.currentText()
            if time_frame == "1 Month":
                date_cutoff = datetime.now() - timedelta(days=30)
            elif time_frame == "3 Month":
                date_cutoff = datetime.now() - timedelta(days=90)
            elif time_frame == "6 Month":
                date_cutoff = datetime.now() - timedelta(days=180)
            elif time_frame == "1 Year":
                date_cutoff = datetime.now() - timedelta(days=365)
            else:
                date_cutoff = datetime.now() - timedelta(days=90)  # Default to 3 months
            
            self.df = self.df[self.df.index >= date_cutoff]

            self.update_plot()

            # Start prediction in a separate thread
            self.status_bar.showMessage("Starting prediction process...")
            self.prediction_thread = PredictionThread(ticker, self.df, self.sequences, self.labels, self.scaler)
            self.prediction_thread.finished.connect(self.update_prediction)
            self.prediction_thread.progress.connect(self.update_progress)
            self.prediction_thread.start()

            self.results_text.setText("Calculating prediction...")

        except ValueError as ve:
            self.results_text.setText(f"An error occurred: {str(ve)}")
            self.status_bar.showMessage("Error occurred")
            self.progress_bar.setVisible(False)

        except Exception as e:
            self.results_text.setText(f"An error occurred: {str(e)}")
            self.status_bar.showMessage("Error occurred")
            self.progress_bar.setVisible(False)

    def update_plot(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Plotting the actual close prices
        ax.plot(self.df.index, self.df['Close'], label='Actual Close Price', color='b', marker='o', markersize=4, linestyle='-')
        
        # Plotting the predicted price
        if self.prediction is not None:
            last_date = self.df.index[-1]
            next_date = last_date + pd.Timedelta(days=1)
            ax.plot([last_date, next_date], [self.df['Close'].iloc[-1], self.prediction], label='Predicted Price', color='orange', marker='o', markersize=4, linestyle='-')
        
        ax.set_title(f'Stock Price - {self.ticker_input.text().upper()}', fontsize=14)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price ($)', fontsize=12)
        
        # Enhancing the x-axis and y-axis details
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))  # Show major ticks every week
        # ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))  # Show minor ticks every day
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))  # Shorter date format
        ax.xaxis.set_minor_formatter(mdates.DateFormatter('%d'))
        
        ax.grid(which='both', linestyle='--', linewidth=0.5)
        ax.tick_params(axis='x', rotation=45, labelsize=10)  # Rotate date labels for better readability
        ax.tick_params(axis='y', labelsize=10)
        
        # Adjust the layout to provide more space for labels
        self.figure.tight_layout(pad=2.0)
        
        # Adding a legend
        ax.legend(fontsize=12)
        
        self.canvas.draw()

    def update_status(self, message):
        self.status_bar.showMessage(message)

    def update_progress(self, value):
        self.progress_bar.setValue(value)
        if value == 100:
            self.progress_bar.setVisible(False)

    def update_prediction(self, next_day_price):
        self.prediction = next_day_price
        self.results_text.setText(f"Predicted next day price: {next_day_price:.2f}")
        self.status_bar.showMessage("Prediction complete.")
        self.update_plot()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = StockPredictionApp()
    window.show()
    sys.exit(app.exec_())