"""Download or load public EPL historical datasets."""

import logging
import pandas as pd
from pathlib import Path

from src.config import RAW_DATA_DIR, RAW_DATA_FILE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_public_datasets() -> pd.DataFrame:
    """
    Load public EPL dataset from local CSV or attempt download.
    
    Returns:
        DataFrame with columns: date, home_team, away_team, home_goals, away_goals, result
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    if RAW_DATA_FILE.exists():
        logger.info(f"Loading existing dataset from {RAW_DATA_FILE}")
        df = pd.read_csv(RAW_DATA_FILE)
        return df
    
    logger.warning(
        f"Dataset not found at {RAW_DATA_FILE}. "
        "Please download EPL historical data from Kaggle or other sources "
        "and place it in data/raw/epl_matches_raw.csv"
    )
    
    sample_data = {
        "date": [],
        "home_team": [],
        "away_team": [],
        "home_goals": [],
        "away_goals": [],
        "result": []
    }
    
    return pd.DataFrame(sample_data)


if __name__ == "__main__":
    df = download_public_datasets()
    logger.info(f"Loaded {len(df)} rows")

