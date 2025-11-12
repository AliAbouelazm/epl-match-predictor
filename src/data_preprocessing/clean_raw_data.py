"""Clean and standardize raw EPL match data from multiple sources."""

import logging
import pandas as pd
from pathlib import Path

from src.config import (
    RAW_DATA_FILE,
    SCRAPED_BS4_FILE,
    SCRAPED_SELENIUM_FILE,
    CLEANED_DATA_FILE,
    INTERIM_DATA_DIR
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def standardize_column_names(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Standardize column names across different data sources."""
    column_mapping = {
        "date": "match_date",
        "Date": "match_date",
        "datetime": "match_date",
        "home": "home_team",
        "HomeTeam": "home_team",
        "Home": "home_team",
        "away": "away_team",
        "AwayTeam": "away_team",
        "Away": "away_team",
        "FTHG": "home_goals",
        "HG": "home_goals",
        "home_score": "home_goals",
        "FTAG": "away_goals",
        "AG": "away_goals",
        "away_score": "away_goals",
        "FTR": "result",
        "Result": "result",
        "Res": "result"
    }
    
    df = df.rename(columns=column_mapping)
    return df


def clean_raw_data() -> pd.DataFrame:
    """
    Load and clean raw data from all sources.
    
    Returns:
        Cleaned DataFrame with standardized columns
    """
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    all_dataframes = []
    
    if RAW_DATA_FILE.exists():
        logger.info(f"Loading public dataset from {RAW_DATA_FILE}")
        df_public = pd.read_csv(RAW_DATA_FILE)
        df_public = standardize_column_names(df_public, "public")
        all_dataframes.append(df_public)
    
    if SCRAPED_BS4_FILE.exists():
        logger.info(f"Loading BeautifulSoup scraped data from {SCRAPED_BS4_FILE}")
        df_bs4 = pd.read_csv(SCRAPED_BS4_FILE)
        df_bs4 = standardize_column_names(df_bs4, "bs4")
        all_dataframes.append(df_bs4)
    
    if SCRAPED_SELENIUM_FILE.exists():
        logger.info(f"Loading Selenium scraped data from {SCRAPED_SELENIUM_FILE}")
        df_selenium = pd.read_csv(SCRAPED_SELENIUM_FILE)
        df_selenium = standardize_column_names(df_selenium, "selenium")
        all_dataframes.append(df_selenium)
    
    if not all_dataframes:
        logger.warning("No raw data files found. Creating empty DataFrame.")
        return pd.DataFrame(columns=["match_date", "home_team", "away_team", "home_goals", "away_goals", "result"])
    
    df_combined = pd.concat(all_dataframes, ignore_index=True)
    
    df_combined["match_date"] = pd.to_datetime(df_combined["match_date"], errors="coerce")
    
    df_combined["home_goals"] = pd.to_numeric(df_combined["home_goals"], errors="coerce")
    df_combined["away_goals"] = pd.to_numeric(df_combined["away_goals"], errors="coerce")
    
    if "result" not in df_combined.columns or df_combined["result"].isna().all():
        df_combined["result"] = df_combined.apply(
            lambda row: "H" if row["home_goals"] > row["away_goals"]
            else "A" if row["away_goals"] > row["home_goals"]
            else "D" if pd.notna(row["home_goals"]) and pd.notna(row["away_goals"]) and row["home_goals"] == row["away_goals"]
            else None,
            axis=1
        )
    
    initial_rows = len(df_combined)
    
    df_combined = df_combined.dropna(subset=["match_date", "home_team", "away_team"])
    
    df_combined = df_combined.drop_duplicates(subset=["match_date", "home_team", "away_team"], keep="first")
    
    df_combined = df_combined.sort_values("match_date").reset_index(drop=True)
    
    final_rows = len(df_combined)
    logger.info(f"Cleaned data: {initial_rows} -> {final_rows} rows")
    
    df_combined.to_csv(CLEANED_DATA_FILE, index=False)
    logger.info(f"Saved cleaned data to {CLEANED_DATA_FILE}")
    
    return df_combined


if __name__ == "__main__":
    df = clean_raw_data()
    logger.info(f"Cleaned dataset contains {len(df)} matches")

