import pandas as pd


class Backtesting:
    def __init__(self, model, prepared_data_path):
        self.model = model
        self.prepared_data = pd.read_csv(prepared_data_path)

    def run_backtest(self):
        # Features for backtesting
        X = self.prepared_data.drop(columns=['target'])

        # Predictions using the trained model
        predictions = self.model.predict(X)
        self.prepared_data['predictions'] = predictions

        # Calculate returns based on predictions
        self.prepared_data['market_return'] = self.prepared_data['Close'].pct_change()
        self.prepared_data['strategy_return'] = self.prepared_data['market_return'] * self.prepared_data['predictions']
        self.prepared_data['cumulative_market_return'] = (1 + self.prepared_data['market_return']).cumprod()
        self.prepared_data['cumulative_strategy_return'] = (1 + self.prepared_data['strategy_return']).cumprod()

        print(self.prepared_data[['cumulative_market_return', 'cumulative_strategy_return']].tail())


# Example usage:
if __name__ == "__main__":
    model_training = ModelTraining('prepared_features.csv')
    trained_model = model_training.train_model()
    backtesting = Backtesting(trained_model, 'prepared_features.csv')
    backtesting.run_backtest()
