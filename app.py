from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Get absolute path for model
app_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(app_dir, "models", "delay_model.pkl")
model = joblib.load(MODEL_PATH)

# Feature names (used for explainable AI)
FEATURE_NAMES = [
    "Airline",
    "Origin",
    "Destination",
    "Day",
    "Month",
    "Weekday",
    "DepHour"
]

@app.route("/", methods=["GET", "POST"])
def index():
    risk_score = None
    risk_level = None
    top_features = []

    if request.method == "POST":

        # ----------- ENCODING MAPS -----------
        airlines_map = {
            "AirIndia": 0,
            "IndiGo": 1,
            "Vistara": 2
        }

        airports_map = {
            "BLR": 0,
            "BOM": 1,
            "DEL": 2,
            "HYD": 3
        }

        weekdays_map = {
            "Monday": 0,
            "Tuesday": 1,
            "Wednesday": 2,
            "Thursday": 3,
            "Friday": 4,
            "Saturday": 5,
            "Sunday": 6
        }

        # ----------- GET FORM VALUES -----------
        airline_str = request.form["airline"]
        origin_str = request.form["origin"]
        destination_str = request.form["destination"]
        weekday_str = request.form["weekday"]

        day = int(request.form["day"])
        month = int(request.form["month"])
        dephour = int(request.form["dephour"])

        # ----------- CONVERT TO NUMERIC -----------
        airline = airlines_map[airline_str]
        origin = airports_map[origin_str]
        destination = airports_map[destination_str]
        weekday = weekdays_map[weekday_str]

        # ----------- MODEL INPUT -----------
        input_data = [[
            airline,
            origin,
            destination,
            day,
            month,
            weekday,
            dephour
        ]]

        # ----------- PREDICTION -----------
        risk_score = model.predict_proba(input_data)[0][1] * 100
        risk_score = round(risk_score, 2)

        # Risk level categorization
        if risk_score < 30:
            risk_level = "Low Risk ✅"
        elif risk_score < 60:
            risk_level = "Medium Risk ⚠️"
        else:
            risk_level = "High Risk 🔴"

        # Get feature importance for explainability
        importances = model.feature_importances_
        top_features = sorted(
            zip(FEATURE_NAMES, importances),
            key=lambda x: x[1],
            reverse=True
        )[:3]

    return render_template(
        "index.html",
        risk_score=risk_score,
        risk_level=risk_level,
        top_features=top_features
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
