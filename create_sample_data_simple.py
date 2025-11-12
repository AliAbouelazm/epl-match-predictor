"""Create sample EPL data for testing the visualizer (without XGBoost dependency)."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

from src.config import (
    CLEANED_DATA_FILE,
    INTERIM_DATA_DIR,
    FEATURE_IMPORTANCE_CSV,
    REPORTS_DIR
)

teams = [
    "Arsenal", "Chelsea", "Liverpool", "Manchester City", "Manchester United",
    "Tottenham", "Newcastle", "Brighton", "Aston Villa", "West Ham",
    "Crystal Palace", "Fulham", "Brentford", "Wolves", "Everton",
    "Nottingham Forest", "Bournemouth", "Burnley", "Sheffield United", "Luton"
]

np.random.seed(42)
start_date = datetime(2020, 8, 1)
matches = []

for i in range(500):
    match_date = start_date + timedelta(days=i*3)
    home_team = np.random.choice(teams)
    away_team = np.random.choice([t for t in teams if t != home_team])
    
    home_goals = np.random.poisson(1.5)
    away_goals = np.random.poisson(1.2)
    
    if home_goals > away_goals:
        result = "H"
    elif away_goals > home_goals:
        result = "A"
    else:
        result = "D"
    
    matches.append({
        "match_date": match_date,
        "home_team": home_team,
        "away_team": away_team,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "result": result
    })

df = pd.DataFrame(matches)
INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
df.to_csv(CLEANED_DATA_FILE, index=False)
print(f"✅ Created sample data: {len(df)} matches")

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
sample_features = [
    "home_goals_scored_avg", "home_goals_conceded_avg", "home_points_avg",
    "home_home_goals_scored_avg", "home_home_goals_conceded_avg", "home_home_points_avg",
    "away_goals_scored_avg", "away_goals_conceded_avg", "away_points_avg",
    "away_away_goals_scored_avg", "away_away_goals_conceded_avg", "away_away_points_avg",
    "goals_scored_diff", "goals_conceded_diff", "points_diff"
]

feature_importance_df = pd.DataFrame({
    "feature": sample_features,
    "importance": np.random.uniform(0.01, 0.15, len(sample_features))
}).sort_values("importance", ascending=False)

feature_importance_df.to_csv(FEATURE_IMPORTANCE_CSV, index=False)
print(f"✅ Created feature importance file")

print("\n⚠️  Note: Model file not created (XGBoost requires OpenMP).")
print("   The Streamlit app will show an error when trying to predict, but you can still see the feature importance visualization.")
print("\n   To fully test predictions, install OpenMP: brew install libomp")

