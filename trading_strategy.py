import pandas as pd


class TradingStrategy:
    def __init__(self, model, data_path):
        self.model = model
        self.data = pd.read_csv(data_path)

    def execute_strategy(self):
        # Extract the features and predict the actions
        X = self.data.drop(columns=['target'], errors='ignore')
        self.data['predictions'] = self.model.predict(X)

        # Define trading actions based on predictions
        self.data['action'] = self.data['predictions'].apply(lambda x: 'Buy' if x == 1 else 'Sell')

        print("Trading Strategy Actions:")
        print(self.data[['Date', 'Close', 'action']].tail())


# Example usage:
if __name__ == "__main__":
    model_training = ModelTraining('prepared_features.csv')
    trained_model = model_training.train_model()
    strategy = TradingStrategy(trained_model, 'prepared_features.csv')
    strategy.execute_strategy()
