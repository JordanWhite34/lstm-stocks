# Stock Price Prediction

This project is a Python-based stock price prediction tool with a graphical user interface. It uses machine learning techniques to forecast the next day's closing price for a given stock ticker.

## Features

- Fetches historical stock data using the Alpha Vantage API
- Implements a Long Short-Term Memory (LSTM) neural network for price prediction
- Provides a user-friendly GUI for input and visualization
- Allows customization of prediction parameters (time frame and training epochs)
- Displays real-time training progress and loss metrics
- Visualizes historical and predicted stock prices

## Files in the Repository

1. `Stock_Prediction_GUI.py`: The main application file containing the graphical user interface.
2. `Stock_Predictor.py`: The backend script that handles data fetching, preprocessing, and the machine learning model.
3. `environment.yml`: Conda environment file specifying the required dependencies.

## Prerequisites

- Python 3.12
- Conda (for environment management)
- Alpha Vantage API key (free tier available at https://www.alphavantage.co/)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com//stock-price-prediction.git
   cd stock-price-prediction
   ```

2. Create and activate the Conda environment:
   ```
   conda env create -f environment.yml
   conda activate StockPrediction
   ```

## Usage

1. Run the GUI application:
   ```
   python Stock_Prediction_GUI.py
   ```

2. Enter your Alpha Vantage API key when prompted.

3. In the main window:
   - Enter a stock ticker symbol (e.g., AAPL for Apple Inc.)
   - Adjust the time frame slider to select the amount of historical data to use
   - Set the number of training epochs using the epoch slider
   - Click the "Predict" button to start the prediction process

4. The application will fetch the data, train the model, and display the results, including:
   - A graph of historical and predicted stock prices
   - The predicted next day's closing price
   - Real-time training progress and loss metrics

## Customization

You can modify the following files to customize the prediction model or GUI:

- `Stock_Predictor.py`: Adjust the LSTM model architecture, sequence length, or add new features to the prediction algorithm.
- `Stock_Prediction_GUI.py`: Customize the GUI layout, add new input parameters, or modify the visualization options.

## Troubleshooting

- If you encounter issues with data fetching, ensure your API key is correct and you have an active internet connection.
- For "No module named" errors, make sure you've activated the correct Conda environment.
- If the prediction seems inaccurate, try increasing the number of epochs or adjusting the time frame.

## Disclaimer

This tool is for educational and research purposes only. Stock price predictions are inherently uncertain and should not be used as the sole basis for financial decisions. Always consult with a qualified financial advisor before making investment choices.

## License

This project is licensed under the MIT License

## Acknowledgments

- [Alpha Vantage](https://www.alphavantage.co/) for providing the stock data API
- [TensorFlow](https://www.tensorflow.org/) and [Keras](https://keras.io/) for the machine learning framework
- [PyQt5](https://www.riverbankcomputing.com/software/pyqt/) for the GUI components
- [Matplotlib](https://matplotlib.org/) for data visualization

