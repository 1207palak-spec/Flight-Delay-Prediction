# ✈️ SkyPredict: Explainable Flight Delay Monitoring Platform

A premium, end-to-end **Machine Learning platform** designed for **aviation logistics and passenger intelligence**.
SkyPredict empowers users to **monitor flight reliability**, **predict delay risks**, and understand the **“why” behind every prediction** using **Explainable AI (XAI)**.

---

## 📑 Table of Contents

* Architecture
* Key Features
* Tech Stack
* Getting Started
* Analysis Workflow
* Project Structure

---

## 🏗️ Architecture

SkyPredict follows a modular, scalable architecture combining machine learning with a modern web interface.

### Frontend

* Responsive Web Dashboard
* Built with **HTML5 and CSS3**
* Optimized for **data density and clarity**

### Backend

* **Flask (Python)** micro-framework
* Handles:

  * Model inference
  * Feature mapping
  * Explainable AI (XAI) calculations

### ML Engine

* **Random Forest Classifier**
* Integrated with a **custom probability-to-risk mapping engine**

### Explainability Layer

* Custom-built **XAI Module**
* Identifies and ranks top contributing features such as:

  * Airline
  * Origin
  * Departure Time

---

## ⭐ Key Features

### 🔍 Advanced Delay Analysis

* **Probability-Based Risk Scoring**
  Goes beyond binary *Delay / No Delay* by generating a granular **Delay Risk Score (%)**.

* **Explainable AI (XAI) Insights**
  Automatically highlights top factors (e.g., route congestion, peak hours) causing the predicted delay.

* **Risk Categorization**
  Real-time classification into:

  * Low Risk
  * Medium Risk
  * High Risk

---

### 📊 Modern Dashboard

* **Intelligent Input Mapping**
  Human-readable dropdowns for Airlines and Airports, dynamically encoded for ML compatibility.

* **Professional Analytics UI**
  Clean, high-contrast interface designed for quick operational decision-making.

* **Real-Time Metrics**
  Immediate display of:

  * Delay Probability
  * Top Risk Drivers

---

### 🛠️ Model Management

* **Pre-trained Pipeline**
  Uses a serialized `delay_model.pkl` for instant inference without retraining.

* **Encoding Safety**
  Backend mappings prevent “Unknown Category” errors from user inputs.

---

## 🧰 Tech Stack

| Component        | Technology                   |
| ---------------- | ---------------------------- |
| Language         | Python 3.8+                  |
| ML Framework     | Scikit-learn (Random Forest) |
| Web Framework    | Flask                        |
| Data Processing  | Pandas, NumPy                |
| Model Deployment | Joblib                       |
| Frontend         | HTML5, CSS3 (Modern Theming) |

---

## 🚀 Getting Started

### Prerequisites

* Python 3.8+
* Pip (Python Package Manager)

---

### 1️⃣ Setup & Installation

```bash
# Clone the repository
git clone https://github.com/your-username/flight-delay-prediction.git
cd flight-delay-prediction

# Create and activate virtual environment
python -m venv .venv
# Windows: .\.venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

### 2️⃣ Run the Application

```bash
# Start the Flask server
python app.py
```

Open your browser and navigate to:
👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 🔬 Analysis Workflow

1. **Input Parameters**
   Select Airline, Origin Airport, and Destination from the dashboard.

2. **Temporal Data Handling**
   Choose scheduled departure date and time (system accounts for peak-hour trends).

3. **Risk Generation**
   Backend transforms categorical inputs into a numerical feature vector.

4. **XAI Processing**
   Random Forest computes delay probability while evaluating feature importance.

5. **Result Interpretation**

| Risk Level      | Meaning                               |
| --------------- | ------------------------------------- |
| 🟢 Blue / Green | Low Risk – Stable Schedule            |
| 🟠 Orange       | Medium Risk – Potential Congestion    |
| 🔴 Red          | High Risk – High Probability of Delay |

---

## 📂 Project Structure

```
SkyPredict/
│
├── app.py                  # Flask application
├── README.md               # Documentation
│
├── templates/
│   └── index.html          # Frontend dashboard
│
├── src/
│   ├── data_loading.py
│   ├── preprocessing.py
│   ├── model_training.py
│   ├── prediction.py
│   └── explainable_ai.py
│
├── data/
│   └── flight_data.csv
│
├── models/
│   └── delay_model.pkl
│
└── outputs/