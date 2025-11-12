"""Tests for feature engineering functions."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.data_preprocessing.feature_engineering import build_feature_matrix


def test_build_feature_matrix():
    """Test that build_feature_matrix returns expected structure."""
    dates = [datetime(2020, 1, 1) + timedelta(days=i*7) for i in range(20)]
    
    clean_data = pd.DataFrame({
        "match_date": dates,
        "home_team": ["Team A", "Team B"] * 10,
        "away_team": ["Team B", "Team A"] * 10,
        "home_goals": [2, 1] * 10,
        "away_goals": [1, 2] * 10,
        "result": ["H", "A"] * 10
    })
    
    X, y = build_feature_matrix(clean_data)
    
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == len(y)
    assert X.shape[1] > 0
    assert len(y.unique()) <= 3


def test_feature_matrix_dimensions():
    """Test feature matrix has expected number of features."""
    dates = [datetime(2020, 1, 1) + timedelta(days=i*7) for i in range(15)]
    
    clean_data = pd.DataFrame({
        "match_date": dates,
        "home_team": ["Team A", "Team B"] * 7 + ["Team A"],
        "away_team": ["Team B", "Team A"] * 7 + ["Team B"],
        "home_goals": [2, 1] * 7 + [1],
        "away_goals": [1, 2] * 7 + [1],
        "result": ["H", "A"] * 7 + ["D"]
    })
    
    X, y = build_feature_matrix(clean_data)
    
    assert X.shape[1] == 15

