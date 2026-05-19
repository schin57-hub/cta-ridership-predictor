"""
Streamlit dashboard for the CTA L Ridership Predictor.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import MODELS_DIR

st.set_page_config(page_title="CTA L Ridership Predictor", page_icon="🚇", layout="wide")

MAPE_MIN_RIDES = 10


@st.cache_data
def load_predictions() -> pd.DataFrame:
    df = pd.read_parquet(MODELS_DIR / "test_predictions.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df["error_xgb"] = df["pred_xgb"] - df["rides"]
    df["error_linear"] = df["pred_linear"] - df["rides"]
    df["abs_pct_error_xgb"] = np.where(
        df["rides"] >= MAPE_MIN_RIDES,
        (df["error_xgb"].abs() / df["rides"]) * 100,
        np.nan,
    )
    df["abs_pct_error_linear"] = np.where(
        df["rides"] >= MAPE_MIN_RIDES,
        (df["error_linear"].abs() / df["rides"]) * 100,
        np.nan,
    )
    return df


@st.cache_data
def load_feature_importance() -> pd.DataFrame:
    return pd.read_csv(MODELS_DIR / "feature_importance.csv")


st.title("🚇 CTA L Ridership Predictor")
st.markdown(
    "Forecasts daily ridership at every CTA 'L' station using historical entries, "
    "calendar features, and holiday signals. Trained on Chicago Data Portal data; "
    "XGBoost beats naive baselines by ~2-3×."
)

predictions = load_predictions()
fi = load_feature_importance()

col1, col2, col3, col4 = st.columns(4)
mae = predictions["error_xgb"].abs().mean()
mape = predictions["abs_pct_error_xgb"].mean(skipna=True)
n_stations = predictions["station_id"].nunique()
date_range = f"{predictions['date'].min().date()} → {predictions['date'].max().date()}"

col1.metric("Mean Absolute Error", f"{mae:,.0f} riders/day")
col2.metric("MAPE (XGBoost)", f"{mape:.1f}%")
col3.metric("Stations", n_stations)
col4.metric("Test window", date_range)

st.caption(
    f"MAPE excludes days with <{MAPE_MIN_RIDES} riders to avoid divide-by-zero noise "
    "on closed/no-service days. MAE uses all days."
)

st.markdown("---")
st.subheader("Predictions vs actual ridership")

stations = sorted(predictions["station_name"].dropna().unique().tolist())
default_idx = stations.index("UIC-Halsted") if "UIC-Halsted" in stations else 0
station = st.selectbox("Pick a station", stations, index=default_idx)

station_df = predictions[predictions["station_name"] == station].sort_values("date")

chart_df = station_df.set_index("date")[["rides", "pred_xgb", "pred_linear"]].rename(
    columns={"rides": "Actual", "pred_xgb": "XGBoost prediction", "pred_linear": "Ridge prediction"}
)
st.line_chart(chart_df, height=400)

col_a, col_b = st.columns(2)
with col_a:
    st.markdown(f"**Per-station metrics for {station}:**")
    n_days = len(station_df)
    n_zero = (station_df["rides"] < MAPE_MIN_RIDES).sum()
    st.write({
        "MAE (XGBoost)":  f"{station_df['error_xgb'].abs().mean():.0f} riders/day",
        "MAPE (XGBoost)": f"{station_df['abs_pct_error_xgb'].mean(skipna=True):.2f}%",
        "MAE (Ridge)":    f"{station_df['error_linear'].abs().mean():.0f} riders/day",
        "Days predicted": n_days,
        "Days excluded from MAPE": f"{n_zero} (low/zero ridership)",
    })

with col_b:
    st.markdown("**Error distribution (predicted − actual):**")
    errors = station_df["error_xgb"].to_numpy()
    if len(errors) > 0:
        lo, hi = np.percentile(errors, [1, 99])
        if hi - lo < 1:
            lo, hi = errors.min() - 1, errors.max() + 1
        bins = np.linspace(lo, hi, 31)
        counts, edges = np.histogram(errors, bins=bins)
        centers = ((edges[:-1] + edges[1:]) / 2).round().astype(int)
        hist_df = pd.DataFrame({"error (riders)": centers, "count": counts}).set_index("error (riders)")
        st.bar_chart(hist_df, height=320)
        st.caption(
            f"Mean error: {errors.mean():+.0f}  •  "
            f"Std: {errors.std():.0f}  •  "
            "A symmetric bell around 0 means the model is unbiased."
        )

st.markdown("---")
st.subheader("Why the model predicts what it predicts")
st.markdown(
    "Top 15 features by XGBoost gain. Lag features dominate, as expected for time series — "
    "**ridership 7 days ago is the single strongest signal** because weekly patterns are highly stable."
)
top_fi = fi.head(15).set_index("feature")
st.bar_chart(top_fi, height=400)

st.markdown("---")
st.subheader("Per-station model performance")
st.caption("Sorted by average daily ridership. MAPE excludes low-ridership days.")

per_station = (
    predictions.groupby("station_name")
    .agg(
        avg_rides=("rides", "mean"),
        mae_xgb=("error_xgb", lambda s: s.abs().mean()),
        mape_xgb=("abs_pct_error_xgb", lambda s: s.mean(skipna=True)),
        mae_linear=("error_linear", lambda s: s.abs().mean()),
        n_days=("rides", "size"),
    )
    .round(2)
    .sort_values("avg_rides", ascending=False)
)
st.dataframe(per_station, use_container_width=True)

st.markdown("---")
st.caption(
    "Data source: Chicago Data Portal (dataset id `5neh-572f`). "
    "Pipeline: time-aware feature engineering → Ridge baseline → XGBoost → SHAP analysis. "
    "Validation: walk-forward time-series cross-validation."
)
