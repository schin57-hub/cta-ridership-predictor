"""
Feature engineering for the CTA ridership model.

LEAKAGE NOTES (this is the part most ML projects get wrong):

  1. Lag features are computed PER STATION via groupby + shift.
     If we shifted the whole dataframe globally, station A's lag would
     leak from station B's history.

  2. Rolling features use `.shift(1)` BEFORE `.rolling()` so the rolling
     window for date D only sees data from D-1 backward. Without this
     shift, today's value would be included in today's rolling mean —
     classic look-ahead leakage.

  3. Holiday flags only use the date itself (no future info), safe.

  4. The first ~year of each station's history will have NaN lag/rolling
     values; we drop those rows AFTER feature creation, not before.
"""
import sys
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import FEATURE_DATA, LAG_DAYS, ROLLING_WINDOWS
from data_loader import load_clean_data


US_HOLIDAYS = holidays.country_holidays("US", subdiv="IL")


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar features derived from the date column. No leakage risk."""
    d = df["date"].dt
    df["dayofweek"] = d.dayofweek
    df["month"] = d.month
    df["quarter"] = d.quarter
    df["day_of_year"] = d.dayofyear
    df["week_of_year"] = d.isocalendar().week.astype(int)
    df["is_weekend"] = (d.dayofweek >= 5).astype(int)
    df["is_monday"] = (d.dayofweek == 0).astype(int)
    df["is_friday"] = (d.dayofweek == 4).astype(int)
    return df


def add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """Federal holidays + day-before/day-after flags (travel days)."""
    dates = df["date"].dt.date
    df["is_holiday"] = dates.map(lambda d: d in US_HOLIDAYS).astype(int)
    # Day before / day after a holiday: often anomalous ridership
    df["is_day_before_holiday"] = dates.map(
        lambda d: (d + pd.Timedelta(days=1)) in US_HOLIDAYS
    ).astype(int)
    df["is_day_after_holiday"] = dates.map(
        lambda d: (d - pd.Timedelta(days=1)) in US_HOLIDAYS
    ).astype(int)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-station lag features. Sorted by date upstream so shift() is correct."""
    for lag in LAG_DAYS:
        df[f"rides_lag_{lag}"] = df.groupby("station_id")["rides"].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-station rolling mean / std features.

    CRITICAL: shift(1) BEFORE rolling so today's value isn't used in
    today's rolling mean. This is the most common time-series leakage bug.
    """
    grp = df.groupby("station_id")["rides"]
    for window in ROLLING_WINDOWS:
        shifted = grp.shift(1)
        df[f"rides_roll_mean_{window}"] = shifted.groupby(df["station_id"]).transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )
        df[f"rides_roll_std_{window}"] = shifted.groupby(df["station_id"]).transform(
            lambda s: s.rolling(window, min_periods=1).std()
        )
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode the train line; keep station_id as a numeric category."""
    df = pd.get_dummies(df, columns=["line"], prefix="line", drop_first=False)
    # daytype is highly correlated with dayofweek but the real data has it
    # as an explicit field, so we keep it as a feature too
    df = pd.get_dummies(df, columns=["daytype"], prefix="dt", drop_first=False)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature pipeline."""
    df = df.copy()
    df = add_calendar_features(df)
    df = add_holiday_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = encode_categoricals(df)
    return df


def main() -> None:
    print("Loading raw data...")
    df = load_clean_data()
    print(f"  {len(df):,} rows")

    print("Building features...")
    df = build_features(df)
    print(f"  {len(df):,} rows, {df.shape[1]} columns")

    # Drop rows with NaN lag features (start of each station's history)
    before = len(df)
    df = df.dropna()
    print(f"  Dropped {before - len(df):,} rows with NaN lags (warm-up period)")

    # Save processed data
    FEATURE_DATA.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FEATURE_DATA, index=False)
    print(f"Saved -> {FEATURE_DATA}")
    print()
    print("Sample columns:")
    print(list(df.columns))


if __name__ == "__main__":
    main()
