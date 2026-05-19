"""
Train and evaluate models on the CTA ridership data.

KEY METHODOLOGY POINTS (for interview defense):

  1. TIME-SERIES SPLIT, NOT RANDOM SPLIT.
     Random k-fold on time series leaks future info into training.
     We hold out the LAST N months as the test set (chronologically).
     For CV during model selection, we use sklearn's TimeSeriesSplit
     which expands a training window forward in time.

  2. NO STATION LEAKAGE.
     Train/test split is purely by date. Each station appears in
     BOTH train and test, but only at different dates. That's the
     correct setup for "predict ridership at known stations into
     the future" — the actual use case.

  3. METRICS.
     - MAE   : easy to explain ("off by ~X riders on average")
     - RMSE  : penalizes big misses
     - MAPE  : interpretable across stations of different sizes
     - R^2   : variance explained

  4. BASELINES.
     We compare against:
       - "Use yesterday's value" (lag-1 persistence)
       - "Use the average of the last 7 days" (lag-7 rolling mean)
       - Linear regression on all features
       - XGBoost on all features
"""
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    FEATURE_DATA, LINEAR_MODEL_PATH, MODELS_DIR, RANDOM_SEED,
    TEST_MONTHS, XGB_MODEL_PATH,
)


# Features we DO NOT use as model inputs
NON_FEATURE_COLS = ["date", "rides", "station_name", "ADA"]


def load_features() -> pd.DataFrame:
    df = pd.read_parquet(FEATURE_DATA)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["station_id", "date"]).reset_index(drop=True)


def time_split(df: pd.DataFrame, test_months: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the last `test_months` months as the test set."""
    cutoff = df["date"].max() - pd.DateOffset(months=test_months)
    train = df[df["date"] <= cutoff].copy()
    test = df[df["date"] > cutoff].copy()
    return train, test


def get_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=NON_FEATURE_COLS)
    y = df["rides"]
    return X, y


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error. Returns 0..100 (a percentage)."""
    # Guard against zero ridership days
    mask = y_true > 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE_%": mape(np.asarray(y_true), np.asarray(y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def baseline_lag1(test_df: pd.DataFrame) -> np.ndarray:
    """Naive baseline: predict today's ridership = yesterday's ridership."""
    return test_df["rides_lag_1"].to_numpy()


def baseline_roll7(test_df: pd.DataFrame) -> np.ndarray:
    """Slightly smarter baseline: predict = 7-day rolling mean."""
    return test_df["rides_roll_mean_7"].to_numpy()


def train_linear(X_train: pd.DataFrame, y_train: pd.Series) -> Ridge:
    """Ridge regression (linear with L2 regularization)."""
    model = Ridge(alpha=1.0, random_state=RANDOM_SEED)
    model.fit(X_train, y_train)
    return model


def train_xgb(X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBRegressor:
    """XGBoost with conservative hyperparameters tuned by light CV."""
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=RANDOM_SEED,
        tree_method="hist",
        n_jobs=-1,
    )
    model.fit(X_train, y_train, verbose=False)
    return model


def cross_validate_xgb(df: pd.DataFrame, n_splits: int = 5) -> list[dict]:
    """TimeSeriesSplit cross-validation to confirm performance is stable.

    Each fold trains on an expanding window and tests on the next chunk.
    This is the correct CV for time-series — it never trains on the future.
    """
    # Sort by date for the CV split to make sense
    df_sorted = df.sort_values("date").reset_index(drop=True)
    X, y = get_xy(df_sorted)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []
    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X), start=1):
        model = train_xgb(X.iloc[tr_idx], y.iloc[tr_idx])
        preds = model.predict(X.iloc[te_idx])
        m = evaluate(y.iloc[te_idx].to_numpy(), preds)
        m["fold"] = fold
        m["n_train"] = len(tr_idx)
        m["n_test"] = len(te_idx)
        fold_metrics.append(m)
    return fold_metrics


def main() -> None:
    print("=" * 70)
    print("CTA Ridership Predictor — Training")
    print("=" * 70)

    print("\n[1/5] Loading features...")
    df = load_features()
    print(f"  {len(df):,} rows, {df.shape[1]} columns")
    print(f"  Date range: {df['date'].min().date()} -> {df['date'].max().date()}")

    print(f"\n[2/5] Time-based train/test split (last {TEST_MONTHS} months = test)...")
    train_df, test_df = time_split(df, TEST_MONTHS)
    print(f"  Train: {len(train_df):,} rows, "
          f"{train_df['date'].min().date()} -> {train_df['date'].max().date()}")
    print(f"  Test : {len(test_df):,} rows, "
          f"{test_df['date'].min().date()} -> {test_df['date'].max().date()}")

    X_train, y_train = get_xy(train_df)
    X_test, y_test = get_xy(test_df)

    print("\n[3/5] Evaluating baselines + models on held-out test set...")
    results = {}

    # Baseline 1: yesterday's ridership
    pred_b1 = baseline_lag1(test_df)
    results["Baseline: Yesterday's value"] = evaluate(y_test.to_numpy(), pred_b1)

    # Baseline 2: 7-day rolling mean
    pred_b2 = baseline_roll7(test_df)
    results["Baseline: 7-day rolling mean"] = evaluate(y_test.to_numpy(), pred_b2)

    # Linear model
    print("    training Ridge...")
    linear = train_linear(X_train, y_train)
    pred_lin = linear.predict(X_test)
    results["Linear (Ridge)"] = evaluate(y_test.to_numpy(), pred_lin)

    # XGBoost
    print("    training XGBoost...")
    xgb_model = train_xgb(X_train, y_train)
    pred_xgb = xgb_model.predict(X_test)
    results["XGBoost"] = evaluate(y_test.to_numpy(), pred_xgb)

    print("\n[4/5] Test-set results:")
    res_df = pd.DataFrame(results).T
    res_df = res_df[["MAE", "RMSE", "MAPE_%", "R2"]]
    print(res_df.round(2).to_string())

    print("\n[5/5] Time-series cross-validation for XGBoost (5 folds)...")
    cv_results = cross_validate_xgb(train_df, n_splits=5)
    cv_df = pd.DataFrame(cv_results)[["fold", "n_train", "n_test", "MAE", "RMSE", "MAPE_%", "R2"]]
    print(cv_df.round(2).to_string(index=False))
    print(f"\n  Mean CV MAPE: {cv_df['MAPE_%'].mean():.2f}%  "
          f"(std: {cv_df['MAPE_%'].std():.2f}%)")

    # Save models
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LINEAR_MODEL_PATH, "wb") as f:
        pickle.dump(linear, f)
    with open(XGB_MODEL_PATH, "wb") as f:
        pickle.dump(xgb_model, f)

    # Save test predictions for the dashboard
    test_df_out = test_df[["station_id", "station_name", "date", "rides"]].copy()
    test_df_out["pred_linear"] = pred_lin
    test_df_out["pred_xgb"] = pred_xgb
    test_df_out.to_parquet(MODELS_DIR / "test_predictions.parquet", index=False)

    # Save feature importance
    fi = pd.DataFrame({
        "feature": X_train.columns,
        "importance": xgb_model.feature_importances_,
    }).sort_values("importance", ascending=False)
    fi.to_csv(MODELS_DIR / "feature_importance.csv", index=False)

    print(f"\nSaved models -> {MODELS_DIR}")
    print("\nTop 10 most important features:")
    print(fi.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
