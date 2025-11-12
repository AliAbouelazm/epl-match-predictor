"""Extract and visualize feature importances from trained XGBoost model."""

import logging
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    MODEL_FILE,
    X_FEATURES_FILE,
    FEATURE_IMPORTANCE_CSV,
    FEATURE_IMPORTANCE_PNG,
    REPORTS_DIR
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_feature_importances() -> pd.DataFrame:
    """
    Extract feature importances from trained model.
    
    Returns:
        DataFrame with features and importance scores
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading trained model")
    model = joblib.load(MODEL_FILE)
    
    X = pd.read_parquet(X_FEATURES_FILE)
    
    importances = model.feature_importances_
    
    feature_importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance": importances
    }).sort_values("importance", ascending=False)
    
    feature_importance_df.to_csv(FEATURE_IMPORTANCE_CSV, index=False)
    logger.info(f"Feature importances saved to {FEATURE_IMPORTANCE_CSV}")
    
    return feature_importance_df


def plot_feature_importances(top_n: int = 15):
    """Plot top N feature importances."""
    df = extract_feature_importances()
    
    top_features = df.head(top_n)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=top_features, y="feature", x="importance", palette="viridis")
    plt.title(f"Top {top_n} Feature Importances")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(FEATURE_IMPORTANCE_PNG, dpi=300, bbox_inches="tight")
    logger.info(f"Feature importance plot saved to {FEATURE_IMPORTANCE_PNG}")
    plt.close()


if __name__ == "__main__":
    plot_feature_importances()
    logger.info("Feature importance analysis complete")

