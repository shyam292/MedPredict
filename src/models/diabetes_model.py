"""
Diabetes ML Model Definitions
===============================
Provides factory functions for building classifiers:
- Support Vector Machine (SVM)
- Random Forest Classifier
- XGBoost Classifier
"""

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def build_svm(kernel: str = "rbf", C: float = 1.0, gamma: str = "scale", probability: bool = True) -> SVC:
    """Build a Support Vector Machine classifier with probability estimates."""
    return SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        probability=probability,
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
    """Return a dictionary of all diabetes model builders."""
    return {
        "SVM": build_svm(),
        "Random Forest": build_random_forest(),
        "XGBoost": build_xgboost(),
    }
