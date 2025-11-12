"""Evaluate trained XGBoost model performance."""

import logging
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report
)
from pathlib import Path

from src.config import (
    X_FEATURES_FILE,
    Y_TARGET_FILE,
    MODEL_FILE,
    REPORTS_DIR
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_model():
    """Evaluate model and generate performance reports."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading model and data")
    model = joblib.load(MODEL_FILE)
    X = pd.read_parquet(X_FEATURES_FILE)
    y = pd.read_parquet(Y_TARGET_FILE)
    
    split_idx = int(len(X) * 0.8)
    X_val = X.iloc[split_idx:]
    y_val = y.iloc[split_idx:]
    
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)
    
    accuracy = accuracy_score(y_val, y_pred)
    balanced_acc = balanced_accuracy_score(y_val, y_pred)
    
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Balanced Accuracy: {balanced_acc:.4f}")
    
    cm = confusion_matrix(y_val, y_pred)
    
    class_names = ["Away Win", "Draw", "Home Win"]
    report = classification_report(y_val, y_pred, target_names=class_names)
    logger.info("\nClassification Report:\n" + report)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=300)
    logger.info(f"Confusion matrix saved to {REPORTS_DIR / 'confusion_matrix.png'}")
    plt.close()


if __name__ == "__main__":
    evaluate_model()
    logger.info("Evaluation complete")

