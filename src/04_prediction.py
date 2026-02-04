# ============================================
# File: 04_prediction.py
# Purpose: Predict delay risk score using trained ML model
# ============================================

import joblib
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
MODEL_PATH = os.path.join(project_root, "models", "delay_model.pkl")


def predict_delay_risk(input_features):
    # Load trained model
    model = joblib.load(MODEL_PATH)

    # Predict probability of delay (class = 1)
    delay_probability = model.predict_proba([input_features])[0][1]

    # Convert to percentage
    risk_score = round(delay_probability * 100, 2)

    # Risk categorization
    if risk_score < 30:
        risk_level = "Low Risk"
    elif risk_score < 60:
        risk_level = "Medium Risk"
    else:
        risk_level = "High Risk"

    return risk_score, risk_level


if __name__ == "__main__":
    print("🚀 Delay Risk Prediction Started\n")

    # SAMPLE INPUT
    # Order of features:
    # [Airline, Origin, Destination, Day, Month, Weekday, DepHour]

    sample_flight = [1, 2, 0, 15, 6, 2, 18]

    score, level = predict_delay_risk(sample_flight)

    print("✈️ Flight Delay Risk Result")
    print("----------------------------")
    print(f"⏱️ Delay Risk Score : {score}%")
    print(f"⚠️ Risk Level       : {level}")
