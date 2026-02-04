# ============================================
# File: 02_data_preprocessing.py
# Purpose: Clean and prepare flight data for ML
# ============================================

import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
DATA_PATH = os.path.join(project_root, "data", "flight_data.csv")

def preprocess_data():
    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Convert FlightDate to datetime
    df["FlightDate"] = pd.to_datetime(df["FlightDate"])

    # Extract date features
    df["Day"] = df["FlightDate"].dt.day
    df["Month"] = df["FlightDate"].dt.month
    df["Weekday"] = df["FlightDate"].dt.weekday

    # Convert departure time to hour
    df["DepHour"] = df["DepTime"].str.split(":").str[0].astype(int)

    # Encode categorical columns
    encoder = LabelEncoder()
    for col in ["Airline", "Origin", "Destination"]:
        df[col] = encoder.fit_transform(df[col])

    # Drop unnecessary columns
    df.drop(columns=["FlightDate", "DepTime", "ArrTime"], inplace=True)

    return df


if __name__ == "__main__":
    processed_df = preprocess_data()

    print("✅ Data preprocessing completed\n")
    print("🔹 Preprocessed data (first 5 rows):")
    print(processed_df.head())

    print("\n🔹 Preprocessed dataset info:")
    print(processed_df.info())
