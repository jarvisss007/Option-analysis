import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

class ModelTraining:
    def __init__(self, feature_set_path):
        # Load prepared data
        self.feature_set_path = feature_set_path
        try:
            self.combined_data = pd.read_csv(feature_set_path)
        except FileNotFoundError:
            raise ValueError(f"The file {feature_set_path} does not exist. Please provide a valid path.")

        # Debugging step to check combined_data types (you can comment this out after debugging)
        print("Data types of combined_data:")
        print(self.combined_data.dtypes)

        # Drop irrelevant columns and prepare features and target
        self.features, self.target = self.prepare_features_and_target()

    def prepare_features_and_target(self):
        # Assuming 'target' is the label column we want to predict
        target_column = 'target'

        # Dropping non-numeric or irrelevant columns if they exist
        if 'Date' in self.combined_data.columns:
            self.combined_data.drop(columns=['Date'], inplace=True)

        features = self.combined_data.drop(columns=[target_column], errors='ignore')
        target = self.combined_data[target_column] if target_column in self.combined_data else None

        if target is None:
            raise ValueError("Target column not found in the dataset. Ensure the data preparation step adds the 'target' column.")

        return features, target

    def train_model(self):
        # Splitting the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.features, self.target, test_size=0.2, random_state=42)

        # Initializing and training the RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Making predictions on the test set
        predictions = model.predict(X_test)

        # Printing model accuracy and classification report
        accuracy = accuracy_score(y_test, predictions)
        print(f"Model Accuracy: {accuracy:.2f}")
        print("Classification Report:")
        print(classification_report(y_test, predictions))

        # Saving the trained model
        joblib.dump(model, 'trained_model.pkl')
        print("Model saved to 'trained_model.pkl'.")

# Example usage:
if __name__ == "__main__":
    feature_set_path = 'prepared_features.csv'
    try:
        model_training = ModelTraining(feature_set_path)
        model_training.train_model()
    except Exception as e:
        print(f"An error occurred during model training: {e}")
