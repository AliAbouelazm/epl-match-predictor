"""Train XGBoost model for EPL match outcome prediction."""

import logging
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import xgboost as xgb

from src.config import (
    X_FEATURES_FILE,
    Y_TARGET_FILE,
    MODEL_FILE,
    MODELS_DIR,
    RANDOM_SEED
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_xgboost_model() -> xgb.XGBClassifier:
    """
    Train XGBoost classifier on EPL match data.
    
    Returns:
        Trained XGBoost model
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading feature matrix and target vector")
    X = pd.read_parquet(X_FEATURES_FILE)
    y = pd.read_parquet(Y_TARGET_FILE)
    
    logger.info(f"Dataset shape: {X.shape[0]} samples, {X.shape[1]} features")
    
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    
    logger.info(f"Train set: {len(X_train)} samples, Validation set: {len(X_val)} samples")
    
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        eval_metric="mlogloss"
    )
    
    logger.info("Training XGBoost model...")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    balanced_acc = balanced_accuracy_score(y_val, y_pred)
    
    logger.info(f"Validation Accuracy: {accuracy:.4f}")
    logger.info(f"Validation Balanced Accuracy: {balanced_acc:.4f}")
    
    joblib.dump(model, MODEL_FILE)
    logger.info(f"Model saved to {MODEL_FILE}")
    
    return model


if __name__ == "__main__":
    model = train_xgboost_model()
    logger.info("Training complete")

