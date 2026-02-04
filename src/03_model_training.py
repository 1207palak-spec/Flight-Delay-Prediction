# ============================================
# File: 03_model_training.py
# Purpose: Train ML model for flight delay prediction
# ============================================
print("🚀 Model training file started")

import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
DATA_PATH = os.path.join(project_root, "data", "flight_data.csv")
MODEL_PATH = os.path.join(project_root, "models", "delay_model.pkl")

def preprocess_data():
    df = pd.read_csv(DATA_PATH)

    # Date features
    df["FlightDate"] = pd.to_datetime(df["FlightDate"])
    df["Day"] = df["FlightDate"].dt.day
    df["Month"] = df["FlightDate"].dt.month
    df["Weekday"] = df["FlightDate"].dt.weekday

    # Time feature
    df["DepHour"] = df["DepTime"].str.split(":").str[0].astype(int)

    # Encode categorical data
    encoder = LabelEncoder()
    for col in ["Airline", "Origin", "Destination"]:
        df[col] = encoder.fit_transform(df[col])

    # Drop unused columns
    df.drop(columns=["FlightDate", "DepTime", "ArrTime"], inplace=True)

    return df


def train_model():
    df = preprocess_data()

    X = df.drop("Delay", axis=1)
    y = df["Delay"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train Random Forest model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Save model
    joblib.dump(model, MODEL_PATH)

    print("✅ Model trained successfully")
    print(f"🎯 Model Accuracy: {accuracy:.2f}")
    print(f"💾 Model saved at: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
