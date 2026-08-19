"""
Model evaluation for the Path Pulse anomaly detector — generates the
numbers and plots you actually want for a final-year presentation.

IsolationForest is unsupervised (Geolife has no "this point is anomalous"
labels), so you can't get a textbook accuracy score straight out of the
box. This script handles that the standard way for unsupervised anomaly
detection: it takes real, normal GPS points from your processed dataset
and injects synthetic anomalies (implausible speeds, off-hours travel,
locations far outside your normal range) with KNOWN labels, then measures
how well the model tells them apart. That gives you a legitimate
precision/recall/F1 you can defend in a viva.

Run this AFTER ml/preprocess.py and ml/train_anomaly.py.

    python ml/evaluate_model.py

Outputs:
    ml/evaluation_outputs/metrics.txt
    ml/evaluation_outputs/score_distribution.png
    ml/evaluation_outputs/confusion_matrix.png
    ml/evaluation_outputs/geographic_scatter.png
    ml/evaluation_outputs/speed_distribution.png
"""

import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_PATH = "ml/datasets/processed_routes.csv"
MODEL_PATH = "ml/models/anomaly_model.pkl"
SCALER_PATH = "ml/models/scaler.pkl"
OUT_DIR = "ml/evaluation_outputs"

FEATURES = [
    "latitude", "longitude",
    "hour", "dayofweek",
    "speed_kmh",
    "is_night",
    "sudden_stop",
    "nearest_risk_km",
    "area_risk_score",
]
N_SYNTHETIC_ANOMALIES = 300
RANDOM_SEED = 42


def load_everything():
    df = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return df, model, scaler


def make_synthetic_anomalies(df, n, rng):
    base = df.sample(n=min(n, len(df)), random_state=RANDOM_SEED, replace=len(df) < n).copy()
    kind = rng.integers(0, 5, size=len(base))

    base.loc[kind == 0, "speed_kmh"] = rng.uniform(120, 300, size=(kind == 0).sum())

    base.loc[kind == 1, "hour"] = rng.integers(1, 4, size=(kind == 1).sum())
    base.loc[kind == 1, "speed_kmh"] = rng.uniform(20, 60, size=(kind == 1).sum())

    base.loc[kind == 2, "hour"] = rng.integers(1, 4, size=(kind == 2).sum())
    base.loc[kind == 2, "speed_kmh"] = rng.uniform(100, 250, size=(kind == 2).sum())

    base.loc[kind == 3, "is_night"] = 1
    base.loc[kind == 3, "nearest_risk_km"] = rng.uniform(0.0, 0.2, size=(kind == 3).sum())
    base.loc[kind == 3, "area_risk_score"] = rng.uniform(0.75, 1.0, size=(kind == 3).sum())
    base.loc[kind == 3, "hour"] = rng.integers(0, 4, size=(kind == 3).sum())

    base.loc[kind == 4, "sudden_stop"] = 1
    base.loc[kind == 4, "nearest_risk_km"] = rng.uniform(0.0, 0.3, size=(kind == 4).sum())
    base.loc[kind == 4, "area_risk_score"] = rng.uniform(0.7, 1.0, size=(kind == 4).sum())

    return base


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    df, model, scaler = load_everything()
    print(f"Loaded {len(df)} real points, model={type(model).__name__}")

    # ---- build a labeled test set: real normal points + synthetic anomalies ----
    normal_sample = df.sample(n=min(800, len(df)), random_state=RANDOM_SEED)[FEATURES].copy()
    normal_sample["label"] = 0  # assumed normal (real, unmodified GPS behaviour)

    synthetic = make_synthetic_anomalies(df, N_SYNTHETIC_ANOMALIES, rng)[FEATURES].copy()
    synthetic["label"] = 1  # known-injected anomaly

    test_set = pd.concat([normal_sample, synthetic], ignore_index=True).sample(
        frac=1, random_state=RANDOM_SEED
    )  # shuffle

    X = scaler.transform(test_set[FEATURES])
    y_true = test_set["label"].values

    raw_pred = model.predict(X)              # 1 = normal, -1 = anomaly
    y_pred = (raw_pred == -1).astype(int)    # 1 = flagged anomaly, 0 = flagged normal
    scores = model.decision_function(X)

    # ---- metrics ----
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(y_true)
    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0

    report = f"""Path Pulse — Anomaly Model Evaluation
=======================================
Dataset: {DATA_PATH}  ({len(df)} real GPS points after feature engineering)
Test set: {len(test_set)} points ({(y_true==0).sum()} real-normal + {(y_true==1).sum()} synthetic-anomaly)
Model: {type(model).__name__} (n_estimators={getattr(model, "n_estimators", "?")}, contamination={getattr(model, "contamination", "?")})
Features: {FEATURES}

Confusion matrix
                 Predicted Normal   Predicted Anomaly
Actual Normal    {tn:<18}{fp}
Actual Anomaly   {fn:<18}{tp}

Precision        {precision:.3f}   (of points flagged anomalous, how many really were)
Recall           {recall:.3f}   (of real injected anomalies, how many were caught)
F1 score         {f1:.3f}
Accuracy         {accuracy:.3f}
False positive rate on real normal data: {false_positive_rate:.3f}
"""
    print(report)

    with open(os.path.join(OUT_DIR, "metrics.txt"), "w") as f:
        f.write(report)

    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump({
            "n_real_points": len(df), "n_test": len(test_set),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
            "accuracy": accuracy, "false_positive_rate": false_positive_rate,
        }, f, indent=2)

    # ---- plot 1: score distribution by true label ----
    plt.figure(figsize=(7, 4.5))
    plt.hist(scores[y_true == 0], bins=40, alpha=0.6, label="Real normal points", color="#2dd4bf")
    plt.hist(scores[y_true == 1], bins=40, alpha=0.6, label="Synthetic anomalies", color="#ef4444")
    plt.axvline(0, color="gray", linestyle="--", linewidth=1, label="Decision boundary")
    plt.xlabel("Isolation Forest decision score (lower = more anomalous)")
    plt.ylabel("Count")
    plt.title("Anomaly score separation: normal vs injected anomalies")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "score_distribution.png"), dpi=150)
    plt.close()

    # ---- plot 2: confusion matrix heatmap ----
    cm = np.array([[tn, fp], [fn, tp]])
    plt.figure(figsize=(4.5, 4))
    plt.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                      color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    plt.xticks([0, 1], ["Predicted Normal", "Predicted Anomaly"])
    plt.yticks([0, 1], ["Actual Normal", "Actual Anomaly"])
    plt.title("Confusion matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    # ---- plot 3: geographic scatter, flagged vs not, on real data ----
    real_X = scaler.transform(df[FEATURES])
    real_pred = model.predict(real_X)
    plt.figure(figsize=(6.5, 6))
    normal_mask = real_pred == 1
    plt.scatter(df.loc[normal_mask, "longitude"], df.loc[normal_mask, "latitude"],
                s=4, alpha=0.4, color="#2dd4bf", label="Normal")
    plt.scatter(df.loc[~normal_mask, "longitude"], df.loc[~normal_mask, "latitude"],
                s=10, alpha=0.8, color="#ef4444", label="Flagged anomalous")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Where the model flags anomalies on real data")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "geographic_scatter.png"), dpi=150)
    plt.close()

    # ---- plot 4: speed distribution, normal vs flagged ----
    plt.figure(figsize=(7, 4.5))
    plt.hist(df.loc[normal_mask, "speed_kmh"], bins=40, alpha=0.6, label="Normal", color="#2dd4bf")
    plt.hist(df.loc[~normal_mask, "speed_kmh"], bins=40, alpha=0.6, label="Flagged anomalous", color="#ef4444")
    plt.xlabel("Speed (km/h)")
    plt.ylabel("Count")
    plt.title("Speed distribution: normal vs flagged points")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "speed_distribution.png"), dpi=150)
    plt.close()

    print(f"Saved metrics + 4 plots to {OUT_DIR}/ — drop these straight into your slides.")


if __name__ == "__main__":
    main()
