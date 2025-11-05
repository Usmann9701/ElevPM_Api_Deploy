from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
import os, gdown

# Auto-download models on Render if not present
os.makedirs("models", exist_ok=True)

files = {

    "models/elevator_random_forest_v6.pkl": "https://drive.google.com/uc?id=1j2L5OI0z5ymsJgHfJLHJv3aKVl5AWUbt",
    "models/scaler.pkl": "https://drive.google.com/uc?id=11MhLptMJtkBWUBOSuxA-JTQlSLvEm90u"
}



for path, url in files.items():
    if not os.path.exists(path):
        print(f"Downloading {path}...")
        gdown.download(url, path, quiet=False)

# ------------------------------------------------------------
# Initialize FastAPI
# ------------------------------------------------------------
app = FastAPI(title="Elevator Predictive Maintenance API")

# Load model and scaler
model = joblib.load("models/elevator_random_forest_v6.pkl")
scaler = joblib.load("models/scaler.pkl")

DATA_PATH = "predictive-maintenance-dataset.csv"


@app.get("/")
def root():
    return {"status": "Elevator PM API is running"}


# ------------------------------------------------------------
# PREDICTION + EVALUATION + ANOMALY DETECTION
# ------------------------------------------------------------
@app.get("/predict")
def predict_auto():
    """
    Runs the model on the full dataset and returns metrics, predictions, and anomalies.
    """
    try:
        df = pd.read_csv(DATA_PATH)
        df.columns = df.columns.str.strip()

        # Feature Engineering (same as model training)
        for lag in range(1, 11):
            df[f'vibration_lag_{lag}'] = df['vibration'].shift(lag)
        df['vibration_diff'] = df['vibration'].diff()
        df['vibration_roll_mean'] = df['vibration'].rolling(window=10).mean()
        df['vibration_roll_std'] = df['vibration'].rolling(window=10).std()
        df = df.dropna().reset_index(drop=True)

        features = [c for c in df.columns if c not in ['vibration', 'ID']]
        X = scaler.transform(df[features])
        y_true = df['vibration'].values
        y_pred = model.predict(X)

        # Evaluation Metrics
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # Anomaly Detection
        diff = np.abs(y_true - y_pred)
        threshold = np.mean(diff) + 3 * np.std(diff)
        anomalies = np.where(diff > threshold)[0]
        anomaly_count = len(anomalies)

        return {
            "status": "ok",
            "prediction_count": len(y_pred),
            "last_actual_vibration": float(y_true[-1]),
            "last_predicted_vibration": float(y_pred[-1]),
            "mae": round(float(mae), 5),
            "r2": round(float(r2), 5),
            "anomaly_threshold": round(float(threshold), 5),
            "anomaly_count": int(anomaly_count),
            "anomaly_indices": anomalies.tolist()[:10],  # preview of anomaly indices
            "message": "Predictions, evaluation, and anomaly detection completed successfully"
        }

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return {"error": str(e)}


# ------------------------------------------------------------
# FUTURE FORECAST (Next 30 steps with confidence band)
# ------------------------------------------------------------
@app.get("/forecast")
def forecast_next_30():
    """
    Generates a 30-step future forecast with confidence bands.
    """
    try:
        df = pd.read_csv(DATA_PATH)
        df.columns = df.columns.str.strip()

        recent_data = df.iloc[-200:].copy().dropna().reset_index(drop=True)

        for lag in range(1, 11):
            recent_data[f'vibration_lag_{lag}'] = recent_data['vibration'].shift(lag)
        recent_data['vibration_diff'] = recent_data['vibration'].diff()
        recent_data['vibration_roll_mean'] = recent_data['vibration'].rolling(window=10).mean()
        recent_data['vibration_roll_std'] = recent_data['vibration'].rolling(window=10).std()
        recent_data = recent_data.dropna().reset_index(drop=True)

        features = [c for c in recent_data.columns if c not in ['vibration', 'ID']]
        future_predictions, future_std = [], []
        future_steps = 30

        for step in range(future_steps):
            for lag in range(1, 11):
                recent_data[f'vibration_lag_{lag}'] = recent_data['vibration'].shift(lag)
            recent_data['vibration_diff'] = recent_data['vibration'].diff()
            recent_data['vibration_roll_mean'] = recent_data['vibration'].rolling(window=10).mean()
            recent_data['vibration_roll_std'] = recent_data['vibration'].rolling(window=10).std()
            valid = recent_data.dropna().reset_index(drop=True)

            if valid.empty:
                break

            X_future = scaler.transform(valid.iloc[[-1]][features])
            preds = np.array([tree.predict(X_future)[0] for tree in model.estimators_])
            next_pred = np.mean(preds)
            next_std = np.std(preds)

            future_predictions.append(next_pred)
            future_std.append(next_std)

            new_row = valid.iloc[-1].copy()
            new_row['vibration'] = next_pred
            recent_data = pd.concat([recent_data, pd.DataFrame([new_row])], ignore_index=True)

        lower = np.array(future_predictions) - np.array(future_std)
        upper = np.array(future_predictions) + np.array(future_std)

        return {
            "status": "ok",
            "forecast_steps": len(future_predictions),
            "predicted_mean": [round(float(x), 3) for x in future_predictions],
            "lower_bound": [round(float(x), 3) for x in lower],
            "upper_bound": [round(float(x), 3) for x in upper],
            "message": "30-step forecast generated successfully"
        }

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return {"error": str(e)}
