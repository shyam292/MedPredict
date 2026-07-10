"""
Explainable AI Module
=======================
Provides SHAP-based model explanations and feature importance visualizations.
- SHAP summary plots
- SHAP force plots for individual predictions
- Feature importance bar charts
"""

import numpy as np
import matplotlib.pyplot as plt
import shap


def get_shap_explainer(model, X_train: np.ndarray, model_type: str = "tree"):
    """
    Create a SHAP explainer for the given model.

    Args:
        model: Trained model
        X_train: Training data for background distribution
        model_type: One of 'tree', 'linear', 'kernel'

    Returns:
        shap.Explainer object
    """
    if model_type == "tree":
        explainer = shap.TreeExplainer(model)
    elif model_type == "linear":
        explainer = shap.LinearExplainer(model, X_train)
    else:
        # Kernel explainer works for any model but is slower
        # Use a subsample of training data for speed
        background = shap.sample(X_train, min(100, len(X_train)))
        explainer = shap.KernelExplainer(model.predict_proba, background)
    return explainer


def compute_shap_values(explainer, X: np.ndarray):
    """Compute SHAP values for the given data."""
    shap_values = explainer.shap_values(X)
    return shap_values


def plot_shap_summary(
    shap_values,
    X: np.ndarray,
    feature_names: list,
    title: str = "SHAP Feature Importance",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Generate a SHAP summary (bee-swarm) plot.
    For binary classification, uses class 1 SHAP values.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Handle multi-output SHAP values (binary classification returns list of 2)
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    shap.summary_plot(
        sv, X,
        feature_names=feature_names,
        show=False,
        plot_size=(10, 6),
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] SHAP summary plot saved to {save_path}")

    return plt.gcf()


def plot_feature_importance(
    model,
    feature_names: list,
    title: str = "Feature Importance",
    top_n: int = 10,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot a horizontal bar chart of feature importances.
    Works with tree-based models that have `feature_importances_` attribute.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(indices)))

    ax.barh(
        range(len(indices)),
        importances[indices][::-1],
        color=colors,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices[::-1]], fontsize=11)
    ax.set_xlabel("Importance Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Feature importance plot saved to {save_path}")

    return fig


def explain_single_prediction(
    explainer,
    instance: np.ndarray,
    feature_names: list,
    class_index: int = 1,
) -> dict:
    """
    Explain a single prediction using SHAP values.
    Returns a dict of feature → SHAP value contributions.
    """
    shap_values = explainer.shap_values(instance)

    # Handle binary classification (list of arrays)
    sv = shap_values[class_index] if isinstance(shap_values, list) else shap_values

    if sv.ndim > 1:
        sv = sv[0]

    contributions = {}
    for i, name in enumerate(feature_names):
        contributions[name] = float(sv[i])

    # Sort by absolute contribution
    contributions = dict(
        sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
    )

    return contributions
