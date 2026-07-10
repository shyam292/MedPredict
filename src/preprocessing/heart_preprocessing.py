"""
Heart Disease Dataset Preprocessing Module
============================================
Handles loading, cleaning, and preprocessing of the UCI Heart Disease Dataset.
- Handles missing / '?' values
- Encodes categorical features
- Applies StandardScaler normalization
- Provides train/test split functionality
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import urllib.request

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
DATASET_URL = (
    "https://raw.githubusercontent.com/dsrscientist/dataset1/master/heart.csv"
)
FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
TARGET_COLUMN = "target"


def load_dataset(data_dir: str | None = None) -> pd.DataFrame:
    """
    Load the UCI Heart Disease dataset.
    Auto-downloads if CSV not found locally.
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "datasets", "heart")
    data_dir = os.path.abspath(data_dir)
    os.makedirs(data_dir, exist_ok=True)

    csv_path = os.path.join(data_dir, "heart.csv")

    if not os.path.exists(csv_path):
        print(f"[INFO] Downloading heart disease dataset to {csv_path} ...")
        urllib.request.urlretrieve(DATASET_URL, csv_path)

    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded heart disease dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the heart disease dataset:
    - Replace '?' with NaN and forward-fill
    - Convert columns to numeric types
    - Drop rows with remaining NaN values
    """
    df_clean = df.copy()

    # Replace '?' with NaN
    df_clean = df_clean.replace("?", np.nan)

    # Convert all columns to numeric
    for col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    # Fill NaN with column median
    for col in df_clean.columns:
        if df_clean[col].isnull().sum() > 0:
            median_val = df_clean[col].median()
            df_clean[col].fillna(median_val, inplace=True)

    print(f"[INFO] Cleaned dataset: {df_clean.shape[0]} rows remaining")
    return df_clean


def preprocess(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    save_scaler: bool = True,
    scaler_path: str | None = None,
) -> tuple:
    """
    Full preprocessing pipeline:
    1. Separate features and target
    2. Train/test split
    3. StandardScaler normalization
    Returns: (X_train, X_test, y_train, y_test, scaler)
    """
    # Determine feature columns dynamically (all except target)
    feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
    X = df[feature_cols].values
    y = df[TARGET_COLUMN].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if save_scaler:
        if scaler_path is None:
            scaler_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "saved_models", "heart_scaler.pkl"
            )
        os.makedirs(os.path.dirname(os.path.abspath(scaler_path)), exist_ok=True)
        joblib.dump(scaler, scaler_path)
        print(f"[INFO] Scaler saved to {scaler_path}")

    print(f"[INFO] Train set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")
    return X_train, X_test, y_train, y_test, scaler


def get_preprocessed_data(data_dir: str | None = None) -> tuple:
    """Convenience function: load → clean → preprocess in one call."""
    df = load_dataset(data_dir)
    df = clean_data(df)
    return preprocess(df)
