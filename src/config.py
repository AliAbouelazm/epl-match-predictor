"""Configuration constants for the EPL match predictor project."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

RANDOM_SEED = 42

RAW_DATA_FILE = RAW_DATA_DIR / "epl_matches_raw.csv"
SCRAPED_BS4_FILE = RAW_DATA_DIR / "scraped_matches_bs4.csv"
SCRAPED_SELENIUM_FILE = RAW_DATA_DIR / "scraped_matches_selenium.csv"
CLEANED_DATA_FILE = INTERIM_DATA_DIR / "epl_matches_cleaned.csv"

X_FEATURES_FILE = PROCESSED_DATA_DIR / "X_features.parquet"
Y_TARGET_FILE = PROCESSED_DATA_DIR / "y_target.parquet"

MODEL_FILE = MODELS_DIR / "xgboost_epl_match_outcome.pkl"
FEATURE_IMPORTANCE_CSV = REPORTS_DIR / "feature_importances.csv"
FEATURE_IMPORTANCE_PNG = REPORTS_DIR / "feature_importances.png"

ROLLING_WINDOW = 5

