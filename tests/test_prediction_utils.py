"""Tests for prediction utilities."""

import pytest
import pandas as pd
from datetime import datetime

from src.models.prediction_utils import prepare_single_match_features


def test_prepare_single_match_features():
    """Test feature preparation for single match."""
    historical_data = pd.DataFrame({
        "match_date": pd.date_range("2020-01-01", periods=20, freq="7D"),
        "home_team": ["Team A", "Team B"] * 10,
        "away_team": ["Team B", "Team A"] * 10,
        "home_goals": [2, 1] * 10,
        "away_goals": [1, 2] * 10,
        "result": ["H", "A"] * 10
    })
    
    features = prepare_single_match_features(
        "Team A",
        "Team B",
        datetime(2021, 1, 1),
        historical_data
    )
    
    assert isinstance(features, pd.DataFrame)
    assert len(features) == 1
    assert features.shape[1] == 15


def test_prediction_probabilities_sum():
    """Test that prediction probabilities sum to approximately 1."""
    try:
        from src.models.prediction_utils import predict_match
        
        result = predict_match("Arsenal", "Chelsea", datetime(2024, 1, 1))
        
        probs = result["probabilities"]
        total = probs["Home Win"] + probs["Draw"] + probs["Away Win"]
        
        assert abs(total - 1.0) < 0.01
    except (FileNotFoundError, ValueError):
        pytest.skip("Model or data not available for testing")

