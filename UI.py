import tkinter as tk
from tkinter import scrolledtext
from data_collection import DataCollection
from data_preparation import prepare_data
from model_training import ModelTraining
from my_backtesting import Backtesting
import pandas as pd
# UI.py

from data_preparation import prepare_data
import pandas as pd

# Example usage within the UI button handler function
def handle_prepare_data():
    historical_data = pd.read_csv('historical_data.csv')
    options_data = pd.read_csv('options_data.csv')
    try:
        prepared_data = prepare_data(historical_data, options_data)
        # You can now use `prepared_data` or display it in your UI
        print(prepared_data.head())
    except ValueError as e:
        print(f"An error occurred during data preparation: {e}")

class TradingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Algo Trading Bot")

        # Button to start the process
        self.start_button = tk.Button(root, text="Start Trading Process", command=self.run_trading_process)
        self.start_button.pack(pady=10)

        # Scrolled Text Widget to display logs
        self.log_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=20)
        self.log_area.pack(pady=10)

    def log_output(self, message):
        """Utility function to log messages to the GUI."""
        self.log_area.insert(tk.END, message + "\n")
        self.log_area.see(tk.END)
        self.root.update()

    def run_trading_process(self):
        """Run the entire trading process: data collection, preparation, training, and backtesting."""
        try:
            # Step 1: Data Collection
            self.log_output("Step 1: Data Collection")
            data_collector = DataCollection()
            historical_data, options_data = data_collector.collect_data()
            self.log_output(f"Historical Data Sample:\n{historical_data.head()}\n")
            self.log_output(f"Options Data Sample:\n{options_data.head()}\n")

            # Step 2: Data Preparation
            self.log_output("Step 2: Data Preparation")
            prepared_data = prepare_data(historical_data.reset_index(), options_data)
            prepared_data.to_csv('prepared_features.csv', index=False)
            self.log_output(f"Prepared Data Sample:\n{prepared_data.head()}\n")

            # Step 3: Model Training
            self.log_output("Step 3: Model Training")
            model_trainer = ModelTraining('prepared_features.csv')
            model_trainer.train_model()
            model_trainer.save_model('trained_model.pkl')
            self.log_output("Model trained and saved.\n")

            # Step 4: Backtesting
            self.log_output("Step 4: Backtesting the Strategy")
            backtester = Backtesting('prepared_features.csv', 'trained_model.pkl')
            strategy_results = backtester.run_backtest()
            self.log_output(f"Backtesting completed.\nCumulative Strategy Return:\n{strategy_results['cumulative_strategy_return'].head()}\n")
            self.log_output(f"Cumulative Market Return:\n{strategy_results['cumulative_market_return'].head()}\n")

        except Exception as e:
            self.log_output(f"An error occurred: {str(e)}")

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = TradingApp(root)
    root.mainloop()
