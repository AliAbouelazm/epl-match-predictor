"""Create the XGBoost model - requires OpenMP to be installed."""

import sys
from pathlib import Path

try:
    import xgboost as xgb
    import joblib
    import pandas as pd
    import numpy as np
    from src.config import X_FEATURES_FILE, Y_TARGET_FILE, MODEL_FILE, MODELS_DIR
    from src.data_preprocessing.feature_engineering import build_feature_matrix
    from src.data_preprocessing.clean_raw_data import clean_raw_data
    
    print("Creating model...")
    
    if not X_FEATURES_FILE.exists() or not Y_TARGET_FILE.exists():
        print("Feature files not found. Building features from cleaned data...")
        from src.config import CLEANED_DATA_FILE
        if CLEANED_DATA_FILE.exists():
            print(f"Loading cleaned data from {CLEANED_DATA_FILE}")
            df_clean = pd.read_csv(CLEANED_DATA_FILE)
            df_clean["match_date"] = pd.to_datetime(df_clean["match_date"])
        else:
            df_clean = clean_raw_data()
        X, y = build_feature_matrix(df_clean)
    else:
        X = pd.read_parquet(X_FEATURES_FILE)
        y = pd.read_parquet(Y_TARGET_FILE)
    
    print(f"Dataset shape: {X.shape[0]} samples, {X.shape[1]} features")
    
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Train set: {len(X_train)} samples, Validation set: {len(X_val)} samples")
    
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="mlogloss"
    )
    
    print("Training model...")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    print(f"✅ Model saved to {MODEL_FILE}")
    
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    balanced_acc = balanced_accuracy_score(y_val, y_pred)
    
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation Balanced Accuracy: {balanced_acc:.4f}")
    
except ImportError as e:
    if "xgboost" in str(e):
        print("❌ ERROR: XGBoost cannot be imported.")
        print("\nThis is likely because OpenMP is not installed.")
        print("\nTo fix this on macOS:")
        print("  1. Install Homebrew if you don't have it: https://brew.sh")
        print("  2. Run: brew install libomp")
        print("  3. Then run this script again")
        print("\nAlternatively, you can use conda:")
        print("  conda install -c conda-forge libomp")
        sys.exit(1)
    else:
        raise
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

