import pandas as pd
from sklearn.preprocessing import StandardScaler

def prepare_data(historical_data, options_data):
    # Convert date columns to datetime for merging
    historical_data['Date'] = pd.to_datetime(historical_data['Date'], utc=True)
    options_data['lastTradeDate'] = pd.to_datetime(options_data['lastTradeDate'], utc=True)

    # Merge historical and options data (left join to retain as much historical data as possible)
    combined_data = pd.merge(historical_data, options_data, left_on='Date', right_on='lastTradeDate', how='inner')

    # Check if merged data is empty
    if combined_data.empty:
        raise ValueError("Merged dataset is empty. Ensure historical_data and options_data have overlapping dates.")

    # Fill NaN values and drop irrelevant columns
    combined_data = combined_data.fillna(method='ffill').fillna(method='bfill')
    combined_data.drop(columns=['contractSymbol', 'currency', 'lastTradeDate'], inplace=True, errors='ignore')

    # Check again if combined_data is empty after filling NaN values
    if combined_data.empty:
        raise ValueError("Dataset is empty after filling NaN values. Ensure there is enough valid data for training.")

    # Ensure that there are enough numeric columns for scaling
    numeric_columns = combined_data.select_dtypes(include=['float64', 'int64']).columns
    if numeric_columns.empty:
        raise ValueError("No numeric columns available for scaling. Please check your data.")

    # Feature Scaling
    scaler = StandardScaler()
    try:
        scaled_features = scaler.fit_transform(combined_data[numeric_columns])
        combined_data[numeric_columns] = scaled_features
    except ValueError as e:
        raise ValueError(f"Scaling failed: {e}")

    # Add a target column (this is a simple placeholder, you should define your own logic)
    # For example, we can set target as 1 if next day's price is greater than today's closing price, else 0
    combined_data['target'] = (combined_data['Close'].shift(-1) > combined_data['Close']).astype(int)

    # Drop rows where target is NaN (since we can't predict the next day for the last row)
    combined_data.dropna(subset=['target'], inplace=True)

    # Check if there's enough data for training
    if combined_data.empty:
        raise ValueError("After adding target, no data is left for training. Ensure proper target generation logic.")

    return combined_data

# Example usage:
if __name__ == "__main__":
    historical_data = pd.read_csv('historical_data.csv')
    options_data = pd.read_csv('options_data.csv')
    try:
        prepared_data = prepare_data(historical_data, options_data)
        prepared_data.to_csv('prepared_features.csv', index=False)
    except ValueError as e:
        print(f"An error occurred during data preparation: {e}")
