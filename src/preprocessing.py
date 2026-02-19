"""
Vancouver Housing Market - Data Preprocessing
==============================================
Feature engineering, encoding, and train/test splitting.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os


def load_data(path: str = "data/vancouver_housing.csv") -> pd.DataFrame:
    """Load the housing dataset."""
    df = pd.read_csv(path)
    print(f"📂 Loaded {len(df)} records from {path}")
    return df


def preprocess(df: pd.DataFrame, fit: bool = True, artifacts_dir: str = "models"):
    """
    Preprocess the housing data for model training.
    
    Args:
        df: Raw dataframe
        fit: If True, fit encoders/scalers and save them. If False, load existing.
        artifacts_dir: Directory to save/load preprocessing artifacts.
    
    Returns:
        X: Feature matrix (DataFrame)
        y: Target vector (Series)
        feature_names: List of feature names after encoding
    """
    os.makedirs(artifacts_dir, exist_ok=True)
    
    df = df.copy()
    
    # ---------------------------------------------------------------
    # Drop columns not useful for modeling
    # ---------------------------------------------------------------
    drop_cols = ["latitude", "longitude"]  # Keep for visualization but not modeling
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    # ---------------------------------------------------------------
    # Separate target
    # ---------------------------------------------------------------
    y = df["price"]
    X = df.drop(columns=["price"])
    
    # ---------------------------------------------------------------
    # Encode categoricals
    # ---------------------------------------------------------------
    # One-hot encode neighborhood and property_type
    categorical_cols = ["neighborhood", "property_type"]
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=False)
    
    # Convert boolean dummies to int
    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype(int)
    
    feature_names = list(X.columns)
    
    # ---------------------------------------------------------------
    # Scale numeric features
    # ---------------------------------------------------------------
    numeric_cols = ["bedrooms", "bathrooms", "sqft", "lot_size", "year_built",
                    "age", "distance_to_downtown_km", "walk_score"]
    numeric_cols = [c for c in numeric_cols if c in X.columns]
    
    if fit:
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
        joblib.dump(scaler, os.path.join(artifacts_dir, "scaler.pkl"))
        joblib.dump(feature_names, os.path.join(artifacts_dir, "feature_names.pkl"))
        print(f"💾 Saved scaler and feature names to {artifacts_dir}/")
    else:
        scaler = joblib.load(os.path.join(artifacts_dir, "scaler.pkl"))
        X[numeric_cols] = scaler.transform(X[numeric_cols])
    
    return X, y, feature_names


def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """Split into train and test sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"📊 Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    return X_train, X_test, y_train, y_test


def main():
    """Run preprocessing pipeline and save artifacts."""
    df = load_data()
    X, y, feature_names = preprocess(df, fit=True)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    print(f"\n✅ Preprocessing complete!")
    print(f"   Features: {len(feature_names)}")
    print(f"   Numeric features scaled with StandardScaler")
    print(f"   Categorical features one-hot encoded")
    print(f"\n📋 Feature list:")
    for i, name in enumerate(feature_names):
        print(f"   {i+1:2d}. {name}")


if __name__ == "__main__":
    main()
