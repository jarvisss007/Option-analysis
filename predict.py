import pandas as pd
import joblib


class Predict:
    def __init__(self, model_path):
        # Load the trained model from the provided path
        self.model = joblib.load(model_path)

    def predict(self, data):
        """
        Make predictions using the trained model.
        :param data: The features dataframe for which predictions need to be made.
        :return: List of predictions (e.g., [0, 1] where 0 = Down, 1 = Up)
        """
        if data.empty:
            raise ValueError("The provided data for prediction is empty.")

        # Ensure the data matches the training features
        return self.model.predict(data)

    def predict_proba(self, data):
        """
        Get the prediction probabilities.
        :param data: The features dataframe for which prediction probabilities need to be made.
        :return: Probability of each class (e.g., [[0.2, 0.8], [0.5, 0.5], ...])
        """
        if data.empty:
            raise ValueError("The provided data for prediction is empty.")

        return self.model.predict_proba(data)
