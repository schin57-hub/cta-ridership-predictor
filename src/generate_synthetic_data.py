"""
Generate a SYNTHETIC dataset that matches the schema of the real Chicago
'L' ridership data.

WHY THIS EXISTS:
    The training/evaluation pipeline needs to be tested before plugging in
    the real data. This script generates fake data with the same columns,
    same dtypes, and realistic-looking patterns (seasonality + day-of-week
    + per-station base rates).

    The real CSV from `download_data.py` has the same column structure, so
    once the user downloads it the rest of the pipeline runs unchanged.

SCHEMA (matches the Chicago Data Portal):
    ridership_raw.csv:
        station_id : int     # matches MAP_ID in stations file
        stationname: str
        date       : str     # original is "MM/DD/YYYY"
        daytype    : str     # W=Weekday, A=Saturday, U=Sunday/Holiday
        rides      : int

    stations_raw.csv:
        STOP_ID                  : int
        DIRECTION_ID             : str
        STOP_NAME                : str
        STATION_NAME             : str
        STATION_DESCRIPTIVE_NAME : str
        MAP_ID                   : int    # joins to ridership.station_id
        ADA                      : bool
        RED, BLUE, G, BRN, P, Pexp, Y, Pnk, O : bool  # line membership
        Location                 : str    # "(lat, long)"
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR, RANDOM_SEED, RAW_RIDERSHIP_CSV, RAW_STATIONS_CSV


# A small set of real-ish CTA station names + lines.
# These are real station names but the ridership numbers below are fabricated.
STATIONS = [
    # (station_id, name, line, lat, long, ridership_scale)
    (40380, "UIC-Halsted",            "BLUE",  41.8757, -87.6493, 6500),
    (40350, "Roosevelt",              "RED",   41.8676, -87.6276, 8200),
    (40670, "Belmont",                "RED",   41.9398, -87.6533, 7800),
    (40730, "Fullerton",              "RED",   41.9252, -87.6526, 9500),
    (41450, "Clark/Lake",             "BLUE",  41.8859, -87.6307, 12000),
    (40360, "Jackson",                "BLUE",  41.8783, -87.6298, 8800),
    (41680, "Logan Square",           "BLUE",  41.9293, -87.7088, 7200),
    (41070, "California",             "BLUE",  41.9216, -87.6964, 4500),
    (40990, "Western (O'Hare Branch)", "BLUE", 41.9163, -87.6877, 5100),
    (40080, "Cumberland",             "BLUE",  41.9846, -87.8385, 3200),
    (41320, "Davis",                  "P",     42.0479, -87.6831, 3800),
    (40540, "Howard",                 "RED",   42.0190, -87.6727, 5400),
    (40460, "Garfield",               "GREEN", 41.7948, -87.6189, 1900),
    (41170, "Midway",                 "ORG",   41.7867, -87.7378, 11500),
    (40890, "Sox-35th",               "RED",   41.8311, -87.6303, 3300),
]


def make_stations_df() -> pd.DataFrame:
    """Build a stations DataFrame matching the real Chicago portal schema."""
    rows = []
    line_cols = ["RED", "BLUE", "G", "BRN", "P", "Pexp", "Y", "Pnk", "O"]
    line_to_col = {
        "RED": "RED", "BLUE": "BLUE", "GREEN": "G", "BROWN": "BRN",
        "P": "P", "PEXP": "Pexp", "YELLOW": "Y", "PINK": "Pnk", "ORG": "O",
    }
    for station_id, name, line, lat, lon, _scale in STATIONS:
        row = {
            "STOP_ID": station_id * 10,  # stop_id != map_id in the real data
            "DIRECTION_ID": "N",
            "STOP_NAME": f"{name} (Northbound)",
            "STATION_NAME": name,
            "STATION_DESCRIPTIVE_NAME": f"{name} ({line} Line)",
            "MAP_ID": station_id,
            "ADA": True,
            "Location": f"({lat}, {lon})",
        }
        for col in line_cols:
            row[col] = (col == line_to_col.get(line))
        rows.append(row)
    return pd.DataFrame(rows)


def make_ridership_df(start: str = "2018-01-01", end: str = "2024-12-31") -> pd.DataFrame:
    """
    Generate daily ridership with realistic patterns:
      - per-station baseline
      - day-of-week effect (weekdays > weekends, with station-specific intensity)
      - yearly seasonality (summer up, winter down)
      - COVID dip in 2020
      - small holiday effects
      - gaussian noise
    """
    rng = np.random.default_rng(RANDOM_SEED)
    dates = pd.date_range(start, end, freq="D")

    out_rows = []
    for station_id, name, _line, _lat, _lon, scale in STATIONS:
        # Day-of-week effect: 0=Mon ... 6=Sun
        dow_effect = np.array([1.00, 1.02, 1.03, 1.02, 0.98, 0.55, 0.45])

        # Yearly seasonality: sin wave, peak in summer
        day_of_year = dates.dayofyear.to_numpy()
        yearly = 1.0 + 0.10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)

        # COVID dip in 2020 (sharp drop, slow recovery)
        years = dates.year.to_numpy()
        months = dates.month.to_numpy()
        covid = np.ones(len(dates))
        covid[(years == 2020) & (months >= 3)] = 0.35
        covid[(years == 2021)] = 0.55
        covid[(years == 2022)] = 0.75

        # Per-day random noise
        noise = rng.normal(1.0, 0.06, size=len(dates))

        # Combine
        rides = (
            scale
            * dow_effect[dates.dayofweek.to_numpy()]
            * yearly
            * covid
            * noise
        ).astype(int)
        rides = np.maximum(rides, 0)  # no negative ridership

        # Daytype encoding
        daytype = np.where(
            dates.dayofweek.to_numpy() < 5, "W",
            np.where(dates.dayofweek.to_numpy() == 5, "A", "U"),
        )

        for d, r, dt in zip(dates, rides, daytype):
            out_rows.append({
                "station_id": station_id,
                "stationname": name,
                "date": d.strftime("%m/%d/%Y"),  # match Chicago portal format
                "daytype": dt,
                "rides": int(r),
            })

    return pd.DataFrame(out_rows)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating synthetic CTA ridership data...")
    rides_df = make_ridership_df()
    rides_df.to_csv(RAW_RIDERSHIP_CSV, index=False)
    print(f"  ridership : {len(rides_df):>8,} rows -> {RAW_RIDERSHIP_CSV}")

    stations_df = make_stations_df()
    stations_df.to_csv(RAW_STATIONS_CSV, index=False)
    print(f"  stations  : {len(stations_df):>8,} rows -> {RAW_STATIONS_CSV}")

    print()
    print("NOTE: This is SYNTHETIC data for testing the pipeline.")
    print("      Run `python src/download_data.py` for real Chicago Data Portal data.")


if __name__ == "__main__":
    main()
