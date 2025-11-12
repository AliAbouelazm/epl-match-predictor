"""Utilities for making predictions with the trained model."""

import logging
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict

from src.config import MODEL_FILE, CLEANED_DATA_FILE
from src.data_preprocessing.feature_engineering import calculate_rolling_stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_trained_model():
    """Load the trained XGBoost model."""
    return joblib.load(MODEL_FILE)


def prepare_single_match_features(
    home_team: str,
    away_team: str,
    match_date: datetime,
    historical_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Prepare features for a single match prediction.
    
    Args:
        home_team: Name of home team
        away_team: Name of away team
        match_date: Date of the match
        historical_data: Historical match data for calculating rolling stats
    
    Returns:
        DataFrame with single row of features
    """
    match_date = pd.to_datetime(match_date)
    
    home_stats = calculate_rolling_stats(historical_data, home_team, True, match_date)
    away_stats = calculate_rolling_stats(historical_data, away_team, False, match_date)
    
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
    
    return pd.DataFrame([feature_dict])


def predict_match(home_team: str, away_team: str, match_date: datetime) -> Dict:
    """
    Predict match outcome for a given match.
    
    Args:
        home_team: Name of home team
        away_team: Name of away team
        match_date: Date of the match
    
    Returns:
        Dictionary with prediction results
    """
    model = load_trained_model()
    
    historical_data = pd.read_csv(CLEANED_DATA_FILE)
    historical_data["match_date"] = pd.to_datetime(historical_data["match_date"])
    
    X = prepare_single_match_features(home_team, away_team, match_date, historical_data)
    
    probabilities = model.predict_proba(X)[0]
    predicted_class = model.predict(X)[0]
    
    class_names = ["Away Win", "Draw", "Home Win"]
    
    result = {
        "home_team": home_team,
        "away_team": away_team,
        "match_date": match_date.strftime("%Y-%m-%d"),
        "predicted_outcome": class_names[predicted_class],
        "probabilities": {
            "Away Win": float(probabilities[0]),
            "Draw": float(probabilities[1]),
            "Home Win": float(probabilities[2])
        },
        "probabilities_percent": {
            "Away Win": f"{probabilities[0] * 100:.1f}%",
            "Draw": f"{probabilities[1] * 100:.1f}%",
            "Home Win": f"{probabilities[2] * 100:.1f}%"
        }
    }
    
    return result

