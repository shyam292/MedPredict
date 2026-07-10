"""
Model Evaluation Utilities
============================
Provides shared functions for evaluating ML models:
- Accuracy, Precision, Recall, F1 Score, ROC-AUC
- Confusion matrix plotting
- ROC curve plotting
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, auc,
)


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
) -> dict:
    """
    Evaluate a trained model and return all metrics as a dictionary.

    Returns:
        dict with keys: accuracy, precision, recall, f1, roc_auc, confusion_matrix
    """
    y_pred = model.predict(X_test)

    # Get probability predictions for ROC-AUC
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_test, y_prob)
    except (AttributeError, IndexError):
        y_prob = None
        roc = None

    metrics = {
        "model_name": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="binary", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="binary", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="binary", zero_division=0),
        "roc_auc": roc,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "y_pred": y_pred,
        "y_prob": y_prob,
    }

    # Print results
    print(f"\n{'='*50}")
    print(f"  {model_name} — Evaluation Results")
    print(f"{'='*50}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1 Score  : {metrics['f1']:.4f}")
    if roc is not None:
        print(f"  ROC AUC   : {metrics['roc_auc']:.4f}")
    print(f"{'='*50}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return metrics


def plot_confusion_matrix(
    cm: np.ndarray,
    title: str = "Confusion Matrix",
    labels: list = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot a styled confusion matrix heatmap."""
    if labels is None:
        labels = ["Negative", "Positive"]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, linecolor="gray", ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Confusion matrix saved to {save_path}")

    return fig


def plot_roc_curve(
    y_test: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    save_path: str | None = None,
) -> plt.Figure:
    """Plot the ROC curve for a binary classifier."""
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc_val = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="#2196F3", lw=2, label=f"{model_name} (AUC = {roc_auc_val:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random Classifier")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curve — {model_name}", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] ROC curve saved to {save_path}")

    return fig


def compare_models(results: list[dict]) -> None:
    """Print a comparison table of multiple model results."""
    print(f"\n{'='*70}")
    print(f"  MODEL COMPARISON")
    print(f"{'='*70}")
    print(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10}")
    print(f"{'-'*70}")
    for r in results:
        roc_str = f"{r['roc_auc']:.4f}" if r['roc_auc'] is not None else "N/A"
        print(
            f"{r['model_name']:<25} {r['accuracy']:>10.4f} {r['precision']:>10.4f} "
            f"{r['recall']:>10.4f} {r['f1']:>10.4f} {roc_str:>10}"
        )
    print(f"{'='*70}")

    # Find best model
    best = max(results, key=lambda x: x["f1"])
    print(f"\n  ★ Best Model: {best['model_name']} (F1 = {best['f1']:.4f})")
