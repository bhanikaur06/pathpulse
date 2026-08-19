"""
Phase 7 — train the anomaly model on engineered features instead of raw
lat/lon alone. Run ml/preprocess.py first to produce processed_routes.csv.
"""

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = "ml/datasets/processed_routes.csv"
MODEL_PATH = "ml/models/anomaly_model.pkl"
SCALER_PATH = "ml/models/scaler.pkl"

FEATURES = ["latitude", "longitude", "hour", "dayofweek", "speed_kmh"]


def main():
    print("Loading processed dataset...")
    df = pd.read_csv(DATA_PATH)

    if len(df) < 50:
        raise SystemExit(
            f"Only {len(df)} rows in {DATA_PATH} — that's too few to train a "
            "meaningful model. Re-run ml/preprocess.py against the full Geolife dataset."
        )

    X = df[FEATURES]
    print(f"Training on {len(X)} points using features: {FEATURES}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test = train_test_split(X_scaled, test_size=0.1, random_state=42)

    model = IsolationForest(
        n_estimators=200,
        contamination=0.03,   # ~3% of points expected to be flagged as unusual
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )

    print("Training Isolation Forest model...")
    model.fit(X_train)

    train_flagged = (model.predict(X_train) == -1).mean()
    test_flagged = (model.predict(X_test) == -1).mean()
    print(f"Flagged as anomaly -> train: {train_flagged:.2%}, test: {test_flagged:.2%}")

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print("Model trained successfully!")
    print(f"Saved model -> {MODEL_PATH}")
    print(f"Saved scaler -> {SCALER_PATH}")


if __name__ == "__main__":
    main()
