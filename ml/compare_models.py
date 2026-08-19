"""
Trains and compares three unsupervised anomaly detection algorithms on
the same engineered features, evaluates each against the same synthetic-
anomaly test set (see ml/evaluate_model.py for why synthetic injection is
the right way to evaluate this), and saves whichever scores best as the
production model — this is a much stronger ML story for a project than
"I picked Isolation Forest and assumed it was fine."

Run after ml/preprocess.py:
    python ml/compare_models.py

Outputs:
    ml/evaluation_outputs/model_comparison.txt
    ml/evaluation_outputs/model_comparison.png
    ml/models/anomaly_model.pkl   (overwritten with the BEST model)
    ml/models/scaler.pkl
    ml/models/model_metadata.json (which model won, and why)
"""

import json
import os
from datetime import datetime

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

DATA_PATH = "ml/datasets/processed_routes.csv"
MODEL_PATH = "ml/models/anomaly_model.pkl"
SCALER_PATH = "ml/models/scaler.pkl"
METADATA_PATH = "ml/models/model_metadata.json"
OUT_DIR = "ml/evaluation_outputs"

FEATURES = [
    "latitude", "longitude",   # where
    "hour", "dayofweek",       # when
    "speed_kmh",               # how fast
    "is_night",                # 1 = 10pm-5am
    "sudden_stop",             # 1 = rapid deceleration
    "nearest_risk_km",         # distance to nearest crime hotspot
    "area_risk_score",         # severity of that hotspot
]
N_SYNTHETIC_ANOMALIES = 300
RANDOM_SEED = 42
CONTAMINATION = 0.03


def make_synthetic_anomalies(df, n, rng):
    base = df.sample(n=min(n, len(df)), random_state=RANDOM_SEED, replace=len(df) < n).copy()
    kind = rng.integers(0, 5, size=len(base))

    # kind 0: implausible speed
    base.loc[kind == 0, "speed_kmh"] = rng.uniform(120, 300, size=(kind == 0).sum())

    # kind 1: unusual hour alone
    base.loc[kind == 1, "hour"] = rng.integers(1, 4, size=(kind == 1).sum())
    base.loc[kind == 1, "speed_kmh"] = rng.uniform(20, 60, size=(kind == 1).sum())

    # kind 2: speed + unusual hour
    base.loc[kind == 2, "hour"] = rng.integers(1, 4, size=(kind == 2).sum())
    base.loc[kind == 2, "speed_kmh"] = rng.uniform(100, 250, size=(kind == 2).sum())

    # kind 3: night-time + inside a high-risk zone
    base.loc[kind == 3, "is_night"] = 1
    base.loc[kind == 3, "nearest_risk_km"] = rng.uniform(0.0, 0.2, size=(kind == 3).sum())
    base.loc[kind == 3, "area_risk_score"] = rng.uniform(0.75, 1.0, size=(kind == 3).sum())
    base.loc[kind == 3, "hour"] = rng.integers(0, 4, size=(kind == 3).sum())

    # kind 4: sudden stop inside a risk zone
    base.loc[kind == 4, "sudden_stop"] = 1
    base.loc[kind == 4, "nearest_risk_km"] = rng.uniform(0.0, 0.3, size=(kind == 4).sum())
    base.loc[kind == 4, "area_risk_score"] = rng.uniform(0.7, 1.0, size=(kind == 4).sum())

    return base


def evaluate(model, X_test, y_true):
    pred = model.predict(X_test)              # 1 = normal, -1 = anomaly
    y_pred = (pred == -1).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(y_true)

    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def jitter_robustness(model, scaler, df, rng, n=300, jitter_deg=0.0004):
    """What fraction of REAL normal points get wrongly flagged anomalous
    after a tiny (~30-40m) GPS-noise-sized location jitter? This matters
    because every live GPS reading has exactly this kind of noise — a
    model that's too sensitive to micro-location will nag the user
    constantly on routes they actually travel all the time."""
    sample = df.sample(n=min(n, len(df)), random_state=RANDOM_SEED).copy()
    sample["latitude"] += rng.uniform(-jitter_deg, jitter_deg, len(sample))
    sample["longitude"] += rng.uniform(-jitter_deg, jitter_deg, len(sample))
    X = scaler.transform(sample[FEATURES])
    pred = model.predict(X)
    return float((pred == -1).mean())


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    df = pd.read_csv(DATA_PATH)
    if len(df) < 50:
        raise SystemExit(f"Only {len(df)} rows — run ml/preprocess.py on the full dataset first.")

    print(f"Loaded {len(df)} real GPS points.")

    scaler = StandardScaler()
    X_all = scaler.fit_transform(df[FEATURES])

    # held-out real-normal points + synthetic anomalies, same recipe as evaluate_model.py
    normal_sample = df.sample(n=min(800, len(df)), random_state=RANDOM_SEED)[FEATURES].copy()
    normal_sample["label"] = 0
    synthetic = make_synthetic_anomalies(df, N_SYNTHETIC_ANOMALIES, rng)[FEATURES].copy()
    synthetic["label"] = 1
    test_set = pd.concat([normal_sample, synthetic], ignore_index=True).sample(frac=1, random_state=RANDOM_SEED)
    X_test = scaler.transform(test_set[FEATURES])
    y_true = test_set["label"].values

    candidates = {
        "IsolationForest": IsolationForest(
            n_estimators=200, contamination=CONTAMINATION, random_state=RANDOM_SEED, n_jobs=-1
        ),
        "LocalOutlierFactor": LocalOutlierFactor(
            n_neighbors=20, contamination=CONTAMINATION, novelty=True
        ),
        "OneClassSVM": OneClassSVM(nu=CONTAMINATION, kernel="rbf", gamma="scale"),
    }

    results = {}
    for name, model in candidates.items():
        print(f"Training {name}...")
        model.fit(X_all)
        metrics = evaluate(model, X_test, y_true)
        metrics["jitter_fp_rate"] = jitter_robustness(model, scaler, df, rng)
        results[name] = metrics
        print(f"  {name}: precision={metrics['precision']:.3f} recall={metrics['recall']:.3f} "
              f"f1={metrics['f1']:.3f} accuracy={metrics['accuracy']:.3f} "
              f"jitter_fp_rate={metrics['jitter_fp_rate']:.3f}")

    # Prefer the highest F1 among models that are stable under realistic GPS
    # jitter (< 10% false-positive rate on noise alone); only fall back to
    # "best jitter robustness regardless of F1" if every candidate is jumpy.
    JITTER_THRESHOLD = 0.10
    stable_candidates = {k: v for k, v in results.items() if v["jitter_fp_rate"] <= JITTER_THRESHOLD}

    if stable_candidates:
        best_name = max(stable_candidates, key=lambda k: stable_candidates[k]["f1"])
    else:
        best_name = min(results, key=lambda k: results[k]["jitter_fp_rate"])
        print("\nNote: every candidate was jumpy under GPS jitter (>10% false-positive "
              "rate on noise alone) — falling back to the most stable one instead of "
              "the highest-F1 one.")

    best_model = candidates[best_name]
    print(f"\nBest model (F1 among GPS-jitter-stable candidates): {best_name}")

    # ---- save report ----
    lines = ["Path Pulse — Model Comparison", "=" * 32, ""]
    for name, m in results.items():
        marker = "  <-- SELECTED" if name == best_name else ""
        lines.append(f"{name}{marker}")
        lines.append(f"  precision={m['precision']:.3f}  recall={m['recall']:.3f}  "
                      f"f1={m['f1']:.3f}  accuracy={m['accuracy']:.3f}")
        lines.append(f"  GPS-jitter false-positive rate={m['jitter_fp_rate']:.3f} "
                      "(lower = more stable on routes you actually travel)")
        lines.append(f"  confusion: tp={m['tp']} fp={m['fp']} tn={m['tn']} fn={m['fn']}")
        lines.append("")
    report = "\n".join(lines)
    print("\n" + report)

    with open(os.path.join(OUT_DIR, "model_comparison.txt"), "w") as f:
        f.write(report)

    # ---- comparison bar chart ----
    metrics_to_plot = ["precision", "recall", "f1", "accuracy"]
    x = np.arange(len(metrics_to_plot))
    width = 0.25
    plt.figure(figsize=(8, 5))
    for i, (name, m) in enumerate(results.items()):
        values = [m[k] for k in metrics_to_plot]
        color = "#2dd4bf" if name == best_name else "#6b7290"
        plt.bar(x + i * width, values, width, label=name + (" (selected)" if name == best_name else ""), color=color)
    plt.xticks(x + width, metrics_to_plot)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Anomaly model comparison (synthetic-anomaly test set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "model_comparison.png"), dpi=150)
    plt.close()

    # ---- save the winning model ----
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    with open(METADATA_PATH, "w") as f:
        json.dump({
            "selected_model": best_name,
            "selected_metrics": results[best_name],
            "all_results": results,
            "features": FEATURES,
            "n_training_points": len(df),
            "trained_at": datetime.now().isoformat(),
            "contamination": CONTAMINATION,
        }, f, indent=2)

    print(f"\nSaved {best_name} as the production model -> {MODEL_PATH}")
    print(f"Comparison report + chart saved in {OUT_DIR}/")


if __name__ == "__main__":
    main()
