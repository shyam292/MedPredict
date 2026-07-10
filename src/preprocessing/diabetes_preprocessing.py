"""
Diabetes Dataset Preprocessing Module
======================================
Handles loading, cleaning, and preprocessing of the Pima Indians Diabetes Dataset.
- Replaces biologically impossible zero values with median
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
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
)
FEATURE_COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
TARGET_COLUMN = "Outcome"
ZERO_REPLACE_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def load_dataset(data_dir: str | None = None) -> pd.DataFrame:
    """
    Load the Pima Indians Diabetes dataset.
    If the CSV is not found locally, it is auto-downloaded.
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "datasets", "diabetes")
    data_dir = os.path.abspath(data_dir)
    os.makedirs(data_dir, exist_ok=True)

    csv_path = os.path.join(data_dir, "diabetes.csv")

    if not os.path.exists(csv_path):
        print(f"[INFO] Downloading diabetes dataset to {csv_path} ...")
        urllib.request.urlretrieve(DATASET_URL, csv_path)
        # Add header row since the raw file has none
        df = pd.read_csv(csv_path, header=None, names=FEATURE_COLUMNS + [TARGET_COLUMN])
        df.to_csv(csv_path, index=False)
    else:
        df = pd.read_csv(csv_path)
        # Handle case where file has no header
        if df.columns[0] != FEATURE_COLUMNS[0]:
            df = pd.read_csv(csv_path, header=None, names=FEATURE_COLUMNS + [TARGET_COLUMN])

    print(f"[INFO] Loaded diabetes dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace biologically impossible zero values with column median.
    Columns affected: Glucose, BloodPressure, SkinThickness, Insulin, BMI.
    """
    df_clean = df.copy()
    for col in ZERO_REPLACE_COLS:
        median_val = df_clean[col].replace(0, np.nan).median()
        df_clean[col] = df_clean[col].replace(0, median_val)
    print("[INFO] Replaced zero values with median for:", ZERO_REPLACE_COLS)
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
    X = df[FEATURE_COLUMNS].values
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
                os.path.dirname(__file__), "..", "..", "saved_models", "diabetes_scaler.pkl"
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
