"""
Phase 8 — load a crime/incident dataset into the risk_zones table so the
dashboard map can show risk overlays and update_location() can flag
high-risk areas.

Usage:
    python ml/import_risk_zones.py path/to/crime_data.csv

Expected columns (case-insensitive, common aliases auto-detected):
    latitude / lat / LAT / LATITUDE
    longitude / lon / lng / LONG / LONGITUDE
    severity (optional, 0.0-1.0, defaults to 0.5)

Any open city/police crime-incident export with point-level coordinates
will work. Many "crime in India" datasets are state/district statistics
WITHOUT per-incident coordinates — those won't work here; you need a
dataset with one row per incident/location and lat/lon columns.

This only populates risk_zones. True turn-by-turn "safer route"
suggestions would additionally need a routing engine (e.g. OSRM or
OpenRouteService) — see README Phase 8 notes.
"""

import sqlite3
import sys

import pandas as pd

DB_PATH = "routes.db"

LAT_ALIASES = ["latitude", "lat", "y"]
LON_ALIASES = ["longitude", "lon", "lng", "long", "x"]
SEVERITY_ALIASES = ["severity", "risk", "risk_score", "weight", "score"]


def find_column(df, aliases):
    lower_map = {c.lower(): c for c in df.columns}
    for alias in aliases:
        if alias in lower_map:
            return lower_map[alias]
    return None


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    csv_path = sys.argv[1]
    df = pd.read_csv(csv_path)

    lat_col = find_column(df, LAT_ALIASES)
    lon_col = find_column(df, LON_ALIASES)
    sev_col = find_column(df, SEVERITY_ALIASES)

    if not lat_col or not lon_col:
        print(f"Could not find latitude/longitude columns. Found columns: {list(df.columns)}")
        print("This dataset likely doesn't have point-level coordinates "
              "(common for state/district crime statistics). Rename the "
              "relevant columns to 'latitude' and 'longitude' and re-run, "
              "or pick a different dataset with per-incident coordinates.")
        sys.exit(1)

    df = df.dropna(subset=[lat_col, lon_col])
    severities = df[sev_col] if sev_col else 0.5

    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS risk_zones(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            latitude REAL,
            longitude REAL,
            risk_score REAL,
            source TEXT
        )
    """)

    rows = []
    for i, r in df.iterrows():
        try:
            lat, lon = float(r[lat_col]), float(r[lon_col])
        except (ValueError, TypeError):
            continue
        sev = float(severities.iloc[i]) if sev_col else 0.5
        sev = max(0.0, min(1.0, sev))  # clamp to 0-1
        rows.append((lat, lon, sev, csv_path))

    conn.executemany(
        "INSERT INTO risk_zones(latitude, longitude, risk_score, source) VALUES (?,?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()

    print(f"Found columns: lat='{lat_col}', lon='{lon_col}', severity='{sev_col or 'none (defaulted to 0.5)'}'")
    print(f"Imported {len(rows)} risk zones from {csv_path} into {DB_PATH}")


if __name__ == "__main__":
    main()



if __name__ == "__main__":
    main()
