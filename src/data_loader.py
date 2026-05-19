"""
Load and clean the raw CTA ridership + station CSVs.

Public function: `load_clean_data()` returns a tidy DataFrame ready for
feature engineering.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RAW_RIDERSHIP_CSV, RAW_STATIONS_CSV


def load_stations() -> pd.DataFrame:
    """Load the station info file and parse out useful columns."""
    df = pd.read_csv(RAW_STATIONS_CSV)

    # Pick the first row per MAP_ID (each station has multiple stop entries
    # for different directions; we just need one row per station for lookups)
    df = df.drop_duplicates(subset=["MAP_ID"], keep="first").copy()

    # Parse the "(lat, lon)" Location string into two numeric columns
    coords = df["Location"].str.strip("()").str.split(",", expand=True)
    df["lat"] = pd.to_numeric(coords[0], errors="coerce")
    df["lon"] = pd.to_numeric(coords[1], errors="coerce")

    # Compute a single "line" label from the boolean columns
    line_cols = ["RED", "BLUE", "G", "BRN", "P", "Pexp", "Y", "Pnk", "O"]
    present_cols = [c for c in line_cols if c in df.columns]
    df["line"] = df[present_cols].apply(
        lambda row: next((c for c in present_cols if row[c]), "UNKNOWN"), axis=1
    )

    return df[["MAP_ID", "STATION_NAME", "line", "lat", "lon", "ADA"]].rename(
        columns={"MAP_ID": "station_id", "STATION_NAME": "station_name"}
    )


def load_clean_data() -> pd.DataFrame:
    """Load ridership + stations, merge, return tidy time-indexed DataFrame.

    Returns columns:
        station_id, station_name, line, lat, lon, ada,
        date (datetime64), rides (int), daytype (str)
    Sorted by (station_id, date) so time-ordered ops are safe downstream.
    """
    rides = pd.read_csv(RAW_RIDERSHIP_CSV)
    rides["date"] = pd.to_datetime(rides["date"], format="mixed")
    rides = rides[["station_id", "stationname", "date", "daytype", "rides"]]

    stations = load_stations()

    # Left-join: keep all ridership rows; stations may not have every ID
    df = rides.merge(stations, on="station_id", how="left", suffixes=("", "_st"))

    # Use the stations file's name when available, otherwise the ridership file's
    df["station_name"] = df["station_name"].fillna(df["stationname"])
    df = df.drop(columns=["stationname"])

    # Sort so groupby-shift / rolling operations are time-correct
    df = df.sort_values(["station_id", "date"]).reset_index(drop=True)

    return df


if __name__ == "__main__":
    df = load_clean_data()
    print(f"Loaded: {len(df):,} rows, {df['station_id'].nunique()} stations")
    print(f"Date range: {df['date'].min().date()} -> {df['date'].max().date()}")
    print()
    print("Sample:")
    print(df.head(10).to_string(index=False))
    print()
    print(f"Missing values:")
    print(df.isna().sum())
