"""
Feature engineering for the anomaly model — Phase 7 + Phase 8 combined.

Geolife .plt files (after the 6-line header):
  0: latitude   1: longitude   2: unused   3: altitude
  4: numeric date (unused)     5: date string   6: time string

Features produced per GPS fix:
  latitude, longitude          — where
  hour, dayofweek              — when
  speed_kmh                    — how fast
  is_night                     — 1 if 22:00-05:00 (higher-risk time band)
  sudden_stop                  — 1 if speed dropped from >15 km/h to <3 km/h
                                  within 60s (possible forced stop)
  nearest_risk_km              — haversine distance to nearest Delhi crime
                                  hotspot (0 = inside a hotspot)
  area_risk_score              — risk_score of the nearest hotspot
                                  (0.0 if no hotspot data loaded)

Run ml/generate_delhi_risk_zones.py first so risk zone data exists, then:
    python ml/preprocess.py
"""

import os
from math import radians, sin, cos, sqrt, asin

import numpy as np
import pandas as pd

DATASET_PATH = "ml/datasets/Data"
OUTPUT_PATH  = "ml/datasets/processed_routes.csv"
RISK_CSV     = "ml/datasets/delhi_risk_zones.csv"

MAX_GAP_SECONDS = 3600
MAX_SPEED_KMH   = 200
NIGHT_START, NIGHT_END = 22, 5     # 10pm – 5am
SUDDEN_STOP_FROM_KMH   = 15        # was moving at least this fast
SUDDEN_STOP_TO_KMH     = 3         # then dropped to below this
SUDDEN_STOP_MAX_SECS   = 60        # within this window


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return 2 * 6371.0 * asin(sqrt(max(0.0, a)))


def load_risk_zones(csv_path):
    """Returns (lat_arr, lon_arr, risk_arr) numpy arrays, or empty arrays."""
    if not os.path.exists(csv_path):
        print(f"[preprocess] No risk zone CSV at {csv_path} — "
              "run ml/generate_delhi_risk_zones.py first for Phase 8 features.")
        return np.array([]), np.array([]), np.array([])
    rz = pd.read_csv(csv_path)
    return rz["latitude"].values, rz["longitude"].values, rz["severity"].values


def nearest_risk(lat, lon, rz_lats, rz_lons, rz_risks):
    if len(rz_lats) == 0:
        return 999.0, 0.0
    # vectorised approximate distance (fast enough for ~1200 hotspot points)
    dlat = np.radians(rz_lats - lat)
    dlon = np.radians(rz_lons - lon)
    rlat = radians(lat)
    a = np.sin(dlat/2)**2 + np.cos(rlat)*np.cos(np.radians(rz_lats))*np.sin(dlon/2)**2
    dists = 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    idx = np.argmin(dists)
    return float(dists[idx]), float(rz_risks[idx])


def load_trajectory(file_path):
    df = pd.read_csv(file_path, skiprows=6, header=None)
    df = df[[0, 1, 5, 6]]
    df.columns = ["latitude", "longitude", "date", "time"]
    df["timestamp"] = pd.to_datetime(df["date"] + " " + df["time"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


def engineer_features(df, user, rz_lats, rz_lons, rz_risks):
    rows = []
    lats = df["latitude"].values
    lons = df["longitude"].values
    ts   = df["timestamp"]

    prev_speed = 0.0

    for i in range(1, len(df)):
        dt = (ts.iloc[i] - ts.iloc[i-1]).total_seconds()
        if dt <= 0 or dt > MAX_GAP_SECONDS:
            prev_speed = 0.0
            continue

        dist_km   = haversine_km(lats[i-1], lons[i-1], lats[i], lons[i])
        speed_kmh = dist_km / (dt / 3600.0)

        if speed_kmh > MAX_SPEED_KMH:
            prev_speed = 0.0
            continue

        hour = ts.iloc[i].hour
        is_night = 1 if (hour >= NIGHT_START or hour < NIGHT_END) else 0

        sudden_stop = 0
        if (prev_speed >= SUDDEN_STOP_FROM_KMH
                and speed_kmh < SUDDEN_STOP_TO_KMH
                and dt <= SUDDEN_STOP_MAX_SECS):
            sudden_stop = 1

        nr_km, nr_risk = nearest_risk(lats[i], lons[i], rz_lats, rz_lons, rz_risks)

        rows.append({
            "user":             user,
            "latitude":         lats[i],
            "longitude":        lons[i],
            "hour":             hour,
            "dayofweek":        ts.iloc[i].dayofweek,
            "speed_kmh":        speed_kmh,
            "is_night":         is_night,
            "sudden_stop":      sudden_stop,
            "nearest_risk_km":  round(nr_km, 4),
            "area_risk_score":  round(nr_risk, 4),
        })

        prev_speed = speed_kmh

    return rows


def main():
    if not os.path.exists(DATASET_PATH):
        raise SystemExit(
            f"Dataset not found at '{DATASET_PATH}'.\n"
            "Copy the Geolife 'Data' folder (user sub-folders 000, 001 … each with a\n"
            "'Trajectory' folder of .plt files) to ml/datasets/Data and re-run."
        )

    rz_lats, rz_lons, rz_risks = load_risk_zones(RISK_CSV)
    has_risk = len(rz_lats) > 0
    print(f"Risk zones loaded: {len(rz_lats)} points" if has_risk else "No risk zones — Phase 8 features will be zero.")

    all_rows   = []
    users_done = 0

    for user in sorted(os.listdir(DATASET_PATH)):
        traj_path = os.path.join(DATASET_PATH, user, "Trajectory")
        if not os.path.exists(traj_path):
            continue

        user_rows = []
        for fname in sorted(os.listdir(traj_path)):
            if not fname.endswith(".plt"):
                continue
            try:
                df = load_trajectory(os.path.join(traj_path, fname))
            except Exception:
                continue
            if len(df) < 2:
                continue
            user_rows.extend(engineer_features(df, user, rz_lats, rz_lons, rz_risks))

        all_rows.extend(user_rows)
        users_done += 1
        if users_done % 20 == 0:
            print(f"  Processed {users_done} users, {len(all_rows):,} points so far…")

    final_df = pd.DataFrame(all_rows)
    print(f"\nTotal GPS points: {len(final_df):,}")
    print(final_df[["speed_kmh", "nearest_risk_km", "area_risk_score", "is_night", "sudden_stop"]].describe())

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nDataset saved → {OUTPUT_PATH}")
    print("Run ml/compare_models.py next.")


if __name__ == "__main__":
    main()
