"""Feature engineering for EPL match outcome prediction."""

import logging
import pandas as pd
import numpy as np
from typing import Tuple

from src.config import ROLLING_WINDOW, X_FEATURES_FILE, Y_TARGET_FILE, PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_rolling_stats(df: pd.DataFrame, team: str, is_home: bool, date: pd.Timestamp) -> dict:
    """Calculate rolling statistics for a team up to a given date."""
    team_matches = df[
        ((df["home_team"] == team) | (df["away_team"] == team)) &
        (df["match_date"] < date)
    ].sort_values("match_date").tail(ROLLING_WINDOW)
    
    if len(team_matches) == 0:
        return {
            "goals_scored": 0.0,
            "goals_conceded": 0.0,
            "points": 0.0,
            "home_goals_scored": 0.0,
            "home_goals_conceded": 0.0,
            "home_points": 0.0,
            "away_goals_scored": 0.0,
            "away_goals_conceded": 0.0,
            "away_points": 0.0
        }
    
    goals_scored = []
    goals_conceded = []
    points = []
    
    home_goals_scored = []
    home_goals_conceded = []
    home_points = []
    
    away_goals_scored = []
    away_goals_conceded = []
    away_points = []
    
    for _, match in team_matches.iterrows():
        if match["home_team"] == team:
            scored = match["home_goals"] if pd.notna(match["home_goals"]) else 0
            conceded = match["away_goals"] if pd.notna(match["away_goals"]) else 0
            home_goals_scored.append(scored)
            home_goals_conceded.append(conceded)
            
            if match["result"] == "H":
                home_points.append(3)
                points.append(3)
            elif match["result"] == "D":
                home_points.append(1)
                points.append(1)
            else:
                home_points.append(0)
                points.append(0)
        else:
            scored = match["away_goals"] if pd.notna(match["away_goals"]) else 0
            conceded = match["home_goals"] if pd.notna(match["home_goals"]) else 0
            away_goals_scored.append(scored)
            away_goals_conceded.append(conceded)
            
            if match["result"] == "A":
                away_points.append(3)
                points.append(3)
            elif match["result"] == "D":
                away_points.append(1)
                points.append(1)
            else:
                away_points.append(0)
                points.append(0)
        
        goals_scored.append(scored)
        goals_conceded.append(conceded)
    
    return {
        "goals_scored": np.mean(goals_scored) if goals_scored else 0.0,
        "goals_conceded": np.mean(goals_conceded) if goals_conceded else 0.0,
        "points": np.mean(points) if points else 0.0,
        "home_goals_scored": np.mean(home_goals_scored) if home_goals_scored else 0.0,
        "home_goals_conceded": np.mean(home_goals_conceded) if home_goals_conceded else 0.0,
        "home_points": np.mean(home_points) if home_points else 0.0,
        "away_goals_scored": np.mean(away_goals_scored) if away_goals_scored else 0.0,
        "away_goals_conceded": np.mean(away_goals_conceded) if away_goals_conceded else 0.0,
        "away_points": np.mean(away_points) if away_points else 0.0
    }


def build_feature_matrix(clean_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build feature matrix and target vector from cleaned data.
    
    Args:
        clean_data: DataFrame with columns match_date, home_team, away_team, home_goals, away_goals, result
    
    Returns:
        Tuple of (feature_matrix, target_vector)
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    features_list = []
    targets = []
    
    df_sorted = clean_data.sort_values("match_date").reset_index(drop=True)
    
    for idx, row in df_sorted.iterrows():
        if pd.isna(row["result"]) or row["result"] not in ["H", "D", "A"]:
            continue
        
        match_date = row["match_date"]
        home_team = row["home_team"]
        away_team = row["away_team"]
        
        home_stats = calculate_rolling_stats(df_sorted, home_team, True, match_date)
        away_stats = calculate_rolling_stats(df_sorted, away_team, False, match_date)
        
        feature_dict = {
            "home_goals_scored_avg": home_stats["goals_scored"],
            "home_goals_conceded_avg": home_stats["goals_conceded"],
            "home_points_avg": home_stats["points"],
            "home_home_goals_scored_avg": home_stats["home_goals_scored"],
            "home_home_goals_conceded_avg": home_stats["home_goals_conceded"],
            "home_home_points_avg": home_stats["home_points"],
            "away_goals_scored_avg": away_stats["goals_scored"],
            "away_goals_conceded_avg": away_stats["goals_conceded"],
            "away_points_avg": away_stats["points"],
            "away_away_goals_scored_avg": away_stats["away_goals_scored"],
            "away_away_goals_conceded_avg": away_stats["away_goals_conceded"],
            "away_away_points_avg": away_stats["away_points"],
            "goals_scored_diff": home_stats["goals_scored"] - away_stats["goals_scored"],
            "goals_conceded_diff": home_stats["goals_conceded"] - away_stats["goals_conceded"],
            "points_diff": home_stats["points"] - away_stats["points"]
        }
        
        features_list.append(feature_dict)
        
        if row["result"] == "H":
            target = 2
        elif row["result"] == "D":
            target = 1
        else:
            target = 0
        
        targets.append(target)
    
    X = pd.DataFrame(features_list)
    y = pd.Series(targets, name="result")
    
    X.to_parquet(X_FEATURES_FILE, index=False)
    y.to_frame().to_parquet(Y_TARGET_FILE, index=False)
    
    logger.info(f"Built feature matrix: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Saved features to {X_FEATURES_FILE}")
    logger.info(f"Saved targets to {Y_TARGET_FILE}")
    
    return X, y


if __name__ == "__main__":
    from src.data_preprocessing.clean_raw_data import clean_raw_data
    
    df_clean = clean_raw_data()
    X, y = build_feature_matrix(df_clean)
    logger.info(f"Feature engineering complete: {len(X)} samples")

