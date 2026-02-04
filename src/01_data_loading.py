import pandas as pd
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
DATA_PATH = os.path.join(project_root, "data", "flight_data.csv")

def load_data():
    try:
        df = pd.read_csv(DATA_PATH)
        print("✅ Dataset loaded successfully\n")
        return df
    except FileNotFoundError:
        print(f"❌ Dataset not found at {DATA_PATH}")
        exit()

if __name__ == "__main__":
    df = load_data()
    print("First 5 rows:")
    print(df.head())
    print("\nDataset Info:")
    print(df.info())
