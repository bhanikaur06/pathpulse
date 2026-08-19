"""
Generates a realistic synthetic Delhi crime hotspot CSV and imports it
into routes.db as risk zones. Based on publicly known high-crime areas in
Delhi (Central Delhi, South-West Delhi, Outer Delhi corridors) derived
from Delhi Police annual crime statistics and press reports — not
individual incident data, but area-level hotspots with realistic density.

Run this once to seed risk zones:
    python ml/generate_delhi_risk_zones.py

If you later download a real crime dataset from Kaggle with lat/lon
columns, run ml/import_risk_zones.py instead (or as well — they stack).
"""

import os
import sqlite3
import numpy as np
import pandas as pd

DB_PATH = "routes.db"
CSV_OUT = "ml/datasets/delhi_risk_zones.csv"
RANDOM_SEED = 42

# Each entry: (area name, centre_lat, centre_lon, radius_km, n_points, risk_score)
# Risk scores: 0.8-1.0 = high crime (serious offences), 0.5-0.7 = moderate
DELHI_HOTSPOTS = [
    # Central Delhi — highest density crimes (theft, snatching, assault)
    ("Connaught Place",     28.6315, 77.2167, 1.2, 80, 0.85),
    ("Paharganj",           28.6448, 77.2149, 0.8, 60, 0.90),
    ("Karol Bagh",          28.6520, 77.1910, 1.0, 70, 0.82),
    ("Old Delhi / Chandni Chowk", 28.6506, 77.2309, 1.0, 65, 0.88),
    ("Sadar Bazar",         28.6587, 77.2101, 0.7, 45, 0.80),

    # South-West Delhi — known snatching/robbery corridor
    ("Dwarka Sector 10",    28.5921, 77.0460, 0.9, 50, 0.72),
    ("Uttam Nagar",         28.6214, 77.0592, 1.1, 65, 0.75),
    ("Janakpuri",           28.6290, 77.0837, 0.8, 45, 0.68),
    ("Vikaspuri",           28.6398, 77.0707, 0.9, 55, 0.71),
    ("Nawada",              28.6220, 77.0680, 0.7, 40, 0.73),

    # Outer Delhi / border corridors — late-night risk areas
    ("Bawana Industrial",   28.7890, 77.0380, 1.2, 55, 0.78),
    ("Narela",              28.8507, 77.0934, 1.0, 45, 0.76),
    ("Alipur",              28.7998, 77.1327, 0.9, 40, 0.70),
    ("Rohini Sector 3",     28.7292, 77.1141, 1.0, 55, 0.65),

    # East Delhi
    ("Shahdara",            28.6726, 77.2895, 1.1, 60, 0.80),
    ("Trilokpuri",          28.6239, 77.3132, 0.9, 50, 0.75),
    ("Mayur Vihar Ph1",     28.6094, 77.2962, 0.8, 40, 0.62),
    ("Geeta Colony",        28.6617, 77.2740, 0.7, 35, 0.68),

    # North Delhi
    ("Seelampur",           28.6823, 77.2730, 0.8, 45, 0.82),
    ("Mustafabad",          28.7094, 77.2980, 0.9, 50, 0.79),
    ("Jafrabad",            28.6905, 77.2820, 0.7, 40, 0.77),

    # South Delhi (lower risk but still notable)
    ("Malviya Nagar",       28.5355, 77.2100, 0.9, 30, 0.55),
    ("Mehrauli",            28.5245, 77.1853, 1.0, 40, 0.60),
    ("Badarpur border",     28.5020, 77.2951, 1.0, 45, 0.72),
]


def scatter_points(centre_lat, centre_lon, radius_km, n, risk_score, rng):
    """Scatter n points inside a circle, weighted toward the centre (crime
    density is highest at the core of each hotspot, not uniform)."""
    rows = []
    for _ in range(n):
        # Beta distribution — more points near centre
        r = rng.beta(2, 5) * (radius_km / 111.0)
        theta = rng.uniform(0, 2 * np.pi)
        lat = centre_lat + r * np.cos(theta)
        lon = centre_lon + r * np.sin(theta) / np.cos(np.radians(centre_lat))
        # Jitter the risk score slightly so nearby points aren't identical
        jittered_risk = float(np.clip(risk_score + rng.normal(0, 0.05), 0.3, 1.0))
        rows.append({"latitude": round(lat, 6), "longitude": round(lon, 6),
                     "severity": jittered_risk, "area": centre_lat})
    return rows


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    all_rows = []

    for (name, clat, clon, radius, n, risk) in DELHI_HOTSPOTS:
        points = scatter_points(clat, clon, radius, n, risk, rng)
        for p in points:
            p["area_name"] = name
        all_rows.extend(points)

    df = pd.DataFrame(all_rows)
    print(f"Generated {len(df)} risk zone points across {len(DELHI_HOTSPOTS)} hotspot areas")
    print(df[["severity"]].describe())

    os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
    df.to_csv(CSV_OUT, index=False)
    print(f"Saved to {CSV_OUT}")

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS risk_zones(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            latitude REAL, longitude REAL, risk_score REAL, source TEXT
        )
    """)

    # Clear any existing synthetic zones before re-inserting
    conn.execute("DELETE FROM risk_zones WHERE source LIKE '%delhi_risk_zones%'")

    rows = [(r["latitude"], r["longitude"], r["severity"], CSV_OUT) for _, r in df.iterrows()]
    conn.executemany(
        "INSERT INTO risk_zones(latitude, longitude, risk_score, source) VALUES (?,?,?,?)", rows
    )
    conn.commit()
    conn.close()
    print(f"Imported {len(rows)} risk zones into routes.db")


if __name__ == "__main__":
    main()
