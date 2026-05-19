"""
Configuration for the CTA L Ridership Predictor.

All file paths, dataset URLs, and project-wide constants live here.
"""
from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Raw input files (downloaded from Chicago Data Portal)
RAW_RIDERSHIP_CSV = DATA_DIR / "ridership_raw.csv"
RAW_STATIONS_CSV = DATA_DIR / "stations_raw.csv"

# Processed files
PROCESSED_DATA = DATA_DIR / "ridership_processed.parquet"
FEATURE_DATA = DATA_DIR / "ridership_features.parquet"

# Trained models
LINEAR_MODEL_PATH = MODELS_DIR / "linear.pkl"
XGB_MODEL_PATH = MODELS_DIR / "xgboost.pkl"
LSTM_MODEL_PATH = MODELS_DIR / "lstm.pt"

# --- Dataset URLs (Chicago Data Portal — public, no API key required) ---
# Ridership: daily entries per station, 2001 to present
RIDERSHIP_URL = (
    "https://data.cityofchicago.org/api/views/5neh-572f/rows.csv?accessType=DOWNLOAD"
)
# Station list: station_id <-> name, line membership, lat/long
STATIONS_URL = (
    "https://data.cityofchicago.org/api/views/8pix-ypme/rows.csv?accessType=DOWNLOAD"
)

# --- Modeling ---
RANDOM_SEED = 42
# Time-series cross-validation: predict the LAST N months using everything before
TEST_MONTHS = 6
CV_N_SPLITS = 5

# --- Features ---
# Lag features (in days) to compute for each station
LAG_DAYS = [1, 7, 14, 28, 365]
# Rolling window sizes (in days) for moving averages
ROLLING_WINDOWS = [7, 28]
