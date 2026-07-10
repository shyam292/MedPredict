"""
Heart Disease ML Model Definitions
=====================================
Provides factory functions for building classifiers:
- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def build_logistic_regression(C: float = 1.0, max_iter: int = 1000) -> LogisticRegression:
    """Build a Logistic Regression classifier."""
    return LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver="lbfgs",
        random_state=42,
    )


def build_random_forest(n_estimators: int = 200, max_depth: int | None = None) -> RandomForestClassifier:
    """Build a Random Forest classifier."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )


def build_xgboost(
    n_estimators: int = 200,
    max_depth: int = 6,
    learning_rate: float = 0.1,
) -> XGBClassifier:
    """Build an XGBoost classifier."""
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        use_label_encoder=False,
    )


def get_all_models() -> dict:
    """Return a dictionary of all heart disease model builders."""
    return {
        "Logistic Regression": build_logistic_regression(),
        "Random Forest": build_random_forest(),
        "XGBoost": build_xgboost(),
    }
