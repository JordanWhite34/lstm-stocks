import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QProgressBar, QLabel, QLineEdit, QPushButton, QTextEdit, 
                             QSlider, QStatusBar, QMessageBox, QFrame)
from PyQt5.QtGui import QFont, QIcon, QColor, QPalette
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPropertyAnimation, QEasingCurve, QRect
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from datetime import datetime, timedelta
from tensorflow import keras
from Stock_Predictor import fetch_stock_data, prepare_data, build_model, predict_next_day

class AnimatedWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.animation = QPropertyAnimation(self, b"geometry")
        self.animation.setEasingCurve(QEasingCurve.OutBounce)
        self.animation.setDuration(1000)

    def animate_in(self):
        start_rect = self.geometry()
        start_rect.setHeight(0)
        end_rect = self.geometry()
        self.animation.setStartValue(start_rect)
        self.animation.setEndValue(end_rect)
        self.animation.start()

class PredictionThread(QThread):
    finished = pyqtSignal(float)
    progress = pyqtSignal(int, int, float)
    error = pyqtSignal(str)
    log = pyqtSignal(str)

    def __init__(self, ticker, df, epochs):
        QThread.__init__(self)
        self.ticker = ticker
        self.df = df
        self.total_epochs = epochs

    def run(self):
        try:
            self.log.emit(f"Starting prediction for {self.ticker} with {self.total_epochs} epochs")
            self.log.emit(f"Data shape: {self.df.shape}")
            
            sequences, labels, scaler = prepare_data(self.df)
            self.log.emit(f"Prepared sequences shape: {sequences.shape}")
            self.log.emit(f"Prepared labels shape: {labels.shape}")
            
            model = build_model((sequences.shape[1], sequences.shape[2]))
            
            class ProgressCallback(keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    current_loss = logs.get('loss', 0)
                    self.model.thread.progress.emit(epoch + 1, self.model.thread.total_epochs, current_loss)
                    self.model.thread.log.emit(f"Epoch {epoch + 1}/{self.model.thread.total_epochs}, Loss: {current_loss:.6f}")

            model.thread = self
            
            history = model.fit(
                sequences, 
                labels, 
                epochs=self.total_epochs, 
                batch_size=32, 
                verbose=0,
                callbacks=[ProgressCallback()]
            )
            
            self.log.emit(f"Training completed. Final loss: {history.history['loss'][-1]:.6f}")
            
            prediction = predict_next_day(model, sequences, scaler)
            self.log.emit(f"Prediction: {prediction:.2f}")
            self.finished.emit(prediction)
        except Exception as e:
            self.error.emit(str(e))

class ApiKeyWindow(AnimatedWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        
        title_label = QLabel('Welcome to Stock Price Predictor!', self)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #2c3e50;")
        layout.addWidget(title_label)
        
        subtitle_label = QLabel('Please enter your Alpha Vantage API Key to begin:', self)
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setStyleSheet("font-size: 16px; color: #34495e;")
        layout.addWidget(subtitle_label)
        
        self.api_key_input = QLineEdit(self)
        self.api_key_input.setPlaceholderText('Enter Alpha Vantage API Key')
        self.api_key_input.setEchoMode(QLineEdit.Password)
        self.api_key_input.setStyleSheet("""
            QLineEdit {
                padding: 10px;
                font-size: 16px;
                border: 2px solid #3498db;
                border-radius: 5px;
            }
        """)
        layout.addWidget(self.api_key_input)
        
        self.submit_button = QPushButton('Submit', self)
        self.submit_button.clicked.connect(self.submit_api_key)
        self.submit_button.setStyleSheet("""
            QPushButton {
                padding: 10px;
                font-size: 16px;
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        layout.addWidget(self.submit_button)
        
        self.setLayout(layout)
        self.setStyleSheet("""
            QWidget {
                background-color: #ecf0f1;
            }
        """)

    def submit_api_key(self):
        api_key = self.api_key_input.text()
        if api_key:
            self.parent.set_api_key(api_key)
            self.parent.show_main_window()
        else:
            QMessageBox.warning(self, 'Invalid Input', 'Please enter a valid API key.')

class StockPredictionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.api_key = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Stock Price Prediction')
        self.setGeometry(100, 100, 900, 700)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QLabel {
                font-size: 14px;
            }
            QLineEdit {
                font-size: 14px;
                padding: 5px;
                border: 1px solid #ccc;
                border-radius: 3px;
            }
            QPushButton {
                font-size: 14px;
                padding: 5px 10px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QTextEdit {
                font-size: 14px;
                border: 1px solid #ccc;
                border-radius: 3px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #B1B1B1, stop:1 #c4c4c4);
                margin: 2px 0;
            }
            QSlider::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #b4b4b4, stop:1 #8f8f8f);
                border: 1px solid #5c5c5c;
                width: 18px;
                margin: -2px 0;
                border-radius: 3px;
            }
        """)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.api_key_window = ApiKeyWindow(self)
        self.main_window = AnimatedWidget()

        self.init_main_window()

        self.central_layout = QVBoxLayout(self.central_widget)
        self.central_layout.addWidget(self.api_key_window)
        self.central_layout.addWidget(self.main_window)

        self.main_window.hide()  # Initially hide the main window

    def init_main_window(self):
        layout = QVBoxLayout(self.main_window)
        layout.setSpacing(20)

        # Input section
        input_frame = QFrame()
        input_frame.setStyleSheet("background-color: white; border-radius: 5px; padding: 10px;")
        input_layout = QVBoxLayout(input_frame)

        # Ticker input
        ticker_layout = QHBoxLayout()
        self.ticker_input = QLineEdit(self.main_window)
        self.ticker_input.setPlaceholderText('Enter Stock Ticker')
        ticker_layout.addWidget(QLabel("Stock Ticker:"))
        ticker_layout.addWidget(self.ticker_input)
        ticker_info_button = QPushButton("(i)", self.main_window)
        ticker_info_button.setFixedSize(20, 20)
        ticker_info_button.clicked.connect(lambda: self.show_info("Stock Ticker", "Enter the stock symbol you want to predict (e.g., AAPL for Apple Inc.)."))
        ticker_layout.addWidget(ticker_info_button)
        input_layout.addLayout(ticker_layout)

        # Time frame slider
        time_frame_layout = QHBoxLayout()
        self.time_frame_slider = QSlider(Qt.Horizontal)
        self.time_frame_slider.setMinimum(3)
        self.time_frame_slider.setMaximum(12)
        self.time_frame_slider.setValue(1)
        self.time_frame_slider.setTickPosition(QSlider.TicksBelow)
        self.time_frame_slider.setTickInterval(1)
        self.time_frame_slider.valueChanged.connect(self.update_time_frame_label)
        self.time_frame_label = QLabel("Time Frame: 3 Months")
        time_frame_layout.addWidget(self.time_frame_label)
        time_frame_layout.addWidget(self.time_frame_slider)
        time_frame_info_button = QPushButton("(i)", self.main_window)
        time_frame_info_button.setFixedSize(20, 20)
        time_frame_info_button.clicked.connect(lambda: self.show_info("Time Frame", "Select the historical data range to use for prediction. This affects both the displayed graph and the data used to train the model."))
        time_frame_layout.addWidget(time_frame_info_button)
        input_layout.addLayout(time_frame_layout)

        # Epoch slider
        epoch_layout = QHBoxLayout()
        self.epoch_slider = QSlider(Qt.Horizontal)
        self.epoch_slider.setMinimum(1)
        self.epoch_slider.setMaximum(100)
        self.epoch_slider.setValue(25)
        self.epoch_slider.setTickPosition(QSlider.TicksBelow)
        self.epoch_slider.setTickInterval(10)
        self.epoch_slider.valueChanged.connect(self.update_epoch_label)
        self.epoch_label = QLabel("Epochs: 25")
        epoch_layout.addWidget(self.epoch_label)
        epoch_layout.addWidget(self.epoch_slider)
        epoch_info_button = QPushButton("(i)", self.main_window)
        epoch_info_button.setFixedSize(20, 20)
        epoch_info_button.clicked.connect(lambda: self.show_info("Epochs", "The number of times the model will iterate over the entire dataset during training. More epochs can lead to better accuracy but take longer to compute."))
        epoch_layout.addWidget(epoch_info_button)
        input_layout.addLayout(epoch_layout)

        layout.addWidget(input_frame)

        self.predict_button = QPushButton('Predict', self.main_window)
        self.predict_button.clicked.connect(self.fetch_data_and_plot)
        layout.addWidget(self.predict_button)

        self.results_text = QTextEdit(self.main_window)
        self.results_text.setReadOnly(True)
        layout.addWidget(self.results_text)

        self.epoch_progress = QProgressBar(self.main_window)
        self.epoch_progress.setRange(0, 100)
        self.epoch_progress.setValue(0)
        self.epoch_progress.setFormat("Epoch Progress: %p%")
        layout.addWidget(self.epoch_progress)
        
        self.loss_label = QLabel("Current Loss: N/A", self.main_window)
        layout.addWidget(self.loss_label)

        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.df = None
        self.sequences = None
        self.scaler = None
        self.prediction = None

    def update_time_frame_label(self, value):
        months = value
        self.time_frame_label.setText(f"Time Frame: {months} Month{'s' if months > 1 else ''}")

    def update_epoch_label(self, value):
        self.epoch_label.setText(f"Epochs: {value}")

    def show_info(self, title, message):
        QMessageBox.information(self, title, message)

    def set_api_key(self, key):
        self.api_key = key

    def show_main_window(self):
        self.api_key_window.hide()
        self.main_window.show()
        self.main_window.animate_in()

    def fetch_data_and_plot(self):
        ticker = self.ticker_input.text().upper()
        epochs = self.epoch_slider.value()
        time_frame_months = self.time_frame_slider.value()
        
        if not ticker:
            self.results_text.setText("Please enter a valid ticker symbol.")
            return

        try:
            self.epoch_progress.setValue(0)
            self.loss_label.setText("Current Loss: N/A")
            self.results_text.clear()

            self.status_bar.showMessage("Fetching stock data...")
            self.df = fetch_stock_data(ticker, self.api_key)
            
            if self.df is None or self.df.empty:
                raise ValueError(f"No data could be fetched for ticker '{ticker}'. Please check the ticker symbol or API key.")

            self.status_bar.showMessage("Preparing data...")
            
            # Determine the date cutoff based on the selected time frame
            date_cutoff = datetime.now() - timedelta(days=30 * time_frame_months)

            # Filter the dataframe based on the selected time frame
            self.df = self.df[self.df.index >= date_cutoff]
            
            self.status_bar.showMessage("Plotting current data...")
            self.update_plot()

            self.status_bar.showMessage("Starting prediction process...")
            self.prediction_thread = PredictionThread(ticker, self.df, epochs)
            self.prediction_thread.finished.connect(self.update_prediction)
            self.prediction_thread.progress.connect(self.update_detailed_progress)
            self.prediction_thread.error.connect(self.handle_prediction_error)
            self.prediction_thread.log.connect(self.log_message)
            self.prediction_thread.start()

            self.results_text.append(f"Calculating prediction using {epochs} epochs and {time_frame_months} month(s) of historical data...")

        except ValueError as ve:
            self.results_text.setText(f"An error occurred: {str(ve)}")
            self.status_bar.showMessage("Error occurred")

        except Exception as e:
            self.results_text.setText(f"An unexpected error occurred: {str(e)}")
            self.status_bar.showMessage("Error occurred")

    def update_plot(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        ax.plot(self.df.index, self.df['Close'], label='Actual Close Price', color='b', marker='o', markersize=4, linestyle='-')
        
        if self.prediction is not None:
            last_date = self.df.index[-1]
            next_date = last_date + pd.Timedelta(days=1)
            ax.plot([last_date, next_date], [self.df['Close'].iloc[-1], self.prediction], label='Predicted Price', color='orange', marker='o', markersize=4, linestyle='-')
        
        ax.set_title(f'Stock Price - {self.ticker_input.text().upper()}', fontsize=14)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price ($)', fontsize=12)
        
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.xaxis.set_minor_formatter(mdates.DateFormatter('%d'))
        
        ax.grid(which='both', linestyle='--', linewidth=0.5)
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        
        self.figure.tight_layout(pad=2.0)
        
        ax.legend(fontsize=12)
        
        self.canvas.draw()

    def update_detailed_progress(self, current_epoch, total_epochs, loss):
        progress_percentage = (current_epoch / total_epochs) * 100
        self.epoch_progress.setValue(int(progress_percentage))
        self.loss_label.setText(f"Current Loss: {loss:.6f}")
        self.status_bar.showMessage(f"Training: Epoch {current_epoch}/{total_epochs}")

    def update_prediction(self, next_day_price):
        self.prediction = next_day_price
        self.results_text.append(f"Predicted next day price: {next_day_price:.2f}")
        self.status_bar.showMessage("Prediction complete.")
        self.update_plot()

    def handle_prediction_error(self, error_message):
        self.results_text.append(f"An error occurred during prediction: {error_message}")
        self.status_bar.showMessage("Prediction error")

    def log_message(self, message):
        self.results_text.append(message)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = StockPredictionApp()
    window.show()
    sys.exit(app.exec_())