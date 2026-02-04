# ============================================
# File: 05_explainable_ai.py
# Purpose: Explain model predictions using feature importance
# ============================================

import pandas as pd
import joblib
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
MODEL_PATH = os.path.join(project_root, "models", "delay_model.pkl")

# Feature names (must match training order)
FEATURE_NAMES = [
    "Airline",
    "Origin",
    "Destination",
    "Day",
    "Month",
    "Weekday",
    "DepHour"
]


def explain_model():
    # Load trained model
    model = joblib.load(MODEL_PATH)

    # Get feature importance
    importances = model.feature_importances_

    # Create dataframe for better readability
    importance_df = pd.DataFrame({
        "Feature": FEATURE_NAMES,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    print("🧠 Explainable AI – Feature Importance")
    print("-------------------------------------")
    print(importance_df)

    # Top 3 contributing factors
    print("\n🔍 Top 3 Factors Affecting Flight Delay:")
    for i, row in importance_df.head(3).iterrows():
        print(f"- {row['Feature']} (Importance: {row['Importance']:.3f})")


if __name__ == "__main__":
    explain_model()
