"""Create sample EPL data for testing the visualizer."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import joblib
import xgboost as xgb

from src.config import (
    CLEANED_DATA_FILE,
    INTERIM_DATA_DIR,
    X_FEATURES_FILE,
    Y_TARGET_FILE,
    PROCESSED_DATA_DIR,
    MODEL_FILE,
    MODELS_DIR,
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
print(f"Created sample data: {len(df)} matches")

from src.data_preprocessing.feature_engineering import build_feature_matrix

X, y = build_feature_matrix(df)
print(f"Created features: {X.shape}")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
model = xgb.XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    n_estimators=50,
    learning_rate=0.1,
    max_depth=4,
    random_state=42
)

split_idx = int(len(X) * 0.8)
X_train = X.iloc[:split_idx]
y_train = y.iloc[:split_idx]

model.fit(X_train, y_train)
joblib.dump(model, MODEL_FILE)
print(f"Trained and saved model")

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
feature_importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

feature_importance_df.to_csv(FEATURE_IMPORTANCE_CSV, index=False)
print(f"Created feature importance file")

print("\n✅ Sample data created! You can now run the Streamlit app.")

