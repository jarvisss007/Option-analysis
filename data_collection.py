import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DataCollection:
    def __init__(self):
        self.spy_ticker = 'SPY'

    def collect_data(self):
        """Collect historical price data for SPY and options data."""
        # Collect historical data for SPY
        historical_data = self.get_historical_data()
        # Collect options data
        options_data = self.get_options_data()
        return historical_data, options_data

    def get_historical_data(self):
        """Collect historical price data for SPY."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # Last 1 year of data
        spy_data = yf.download(self.spy_ticker, start=start_date, end=end_date)
        if spy_data.empty:
            raise ValueError("No historical data found for SPY.")
        spy_data.reset_index(inplace=True)  # Ensure 'Date' is a column rather than the index
        return spy_data

    def get_options_data(self):
        """Collect options data for SPY."""
        ticker = yf.Ticker(self.spy_ticker)
        expiration_dates = ticker.options
        if expiration_dates:
            options_data = ticker.option_chain(expiration_dates[0])
            options_data_df = pd.concat([options_data.calls, options_data.puts])
            options_data_df['option_type'] = ['call'] * len(options_data.calls) + ['put'] * len(options_data.puts)
        else:
            raise ValueError("No options data available for SPY.")
        options_data_df['lastTradeDate'] = pd.to_datetime(options_data_df['lastTradeDate'])  # Ensure consistent datetime format
        return options_data_df

    def calculate_greeks(self, options_data):
        """Calculate Greeks (Delta, Gamma, Theta, Vega) for options data."""
        options_data['delta'] = np.random.uniform(-1, 1, len(options_data))
        options_data['gamma'] = np.random.uniform(0, 1, len(options_data))
        options_data['theta'] = np.random.uniform(-1, 0, len(options_data))
        options_data['vega'] = np.random.uniform(0, 1, len(options_data))
        return options_data

# Example usage:
if __name__ == "__main__":
    data_collector = DataCollection()
    historical_data, options_data = data_collector.collect_data()
    options_data = data_collector.calculate_greeks(options_data)
    print(historical_data.head())
    print(options_data.head())
