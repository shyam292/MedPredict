"""
Heart Disease Model Training Script
======================================
Trains Logistic Regression, Random Forest, and XGBoost models on the
UCI Heart Disease dataset.
- Evaluates all models with full metrics
- Saves the best-performing model to saved_models/
- Generates confusion matrix and ROC curve plots

Usage:
    python -m src.training.train_heart_model
    OR
    python src/training/train_heart_model.py
"""

import os
import sys
import joblib

# Ensure project root is in path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing.heart_preprocessing import get_preprocessed_data, FEATURE_COLUMNS
from src.models.heart_model import get_all_models
from src.utils.evaluation import evaluate_model, plot_confusion_matrix, plot_roc_curve, compare_models
from src.utils.explainability import get_shap_explainer, plot_shap_summary, plot_feature_importance


def main():
    print("\n" + "=" * 60)
    print("  MedPredict — Heart Disease Model Training")
    print("=" * 60)

    # ── Step 1: Load and preprocess data ──
    print("\n[STEP 1] Loading and preprocessing dataset...")
    X_train, X_test, y_train, y_test, scaler = get_preprocessed_data()

    # ── Step 2: Train all models ──
    print("\n[STEP 2] Training models...")
    models = get_all_models()
    results = []

    for name, model in models.items():
        print(f"\n  Training {name}...")
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test, model_name=name)
        results.append(metrics)

    # ── Step 3: Compare models ──
    print("\n[STEP 3] Comparing models...")
    compare_models(results)

    # ── Step 4: Save the best model ──
    best_result = max(results, key=lambda x: x["f1"])
    best_model_name = best_result["model_name"]
    best_model = models[best_model_name]

    save_dir = os.path.join(PROJECT_ROOT, "saved_models")
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, "heart_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"\n[STEP 4] Best model ({best_model_name}) saved to {model_path}")

    # ── Step 5: Generate plots ──
    print("\n[STEP 5] Generating evaluation plots...")

    # Confusion Matrix
    plot_confusion_matrix(
        best_result["confusion_matrix"],
        title=f"Heart Disease — {best_model_name} Confusion Matrix",
        save_path=os.path.join(save_dir, "heart_confusion_matrix.png"),
    )

    # ROC Curve
    if best_result["y_prob"] is not None:
        plot_roc_curve(
            y_test, best_result["y_prob"],
            model_name=best_model_name,
            save_path=os.path.join(save_dir, "heart_roc_curve.png"),
        )

    # ── Step 6: Feature importance & SHAP ──
    print("\n[STEP 6] Generating SHAP explanations...")
    try:
        if hasattr(best_model, "feature_importances_"):
            plot_feature_importance(
                best_model, FEATURE_COLUMNS,
                title=f"Heart Disease — {best_model_name} Feature Importance",
                save_path=os.path.join(save_dir, "heart_feature_importance.png"),
            )

        # SHAP explainer
        if best_model_name in ["Random Forest", "XGBoost"]:
            model_type = "tree"
        elif best_model_name == "Logistic Regression":
            model_type = "linear"
        else:
            model_type = "kernel"

        explainer = get_shap_explainer(best_model, X_train, model_type=model_type)
        plot_shap_summary(
            explainer.shap_values(X_test[:100]),
            X_test[:100],
            feature_names=FEATURE_COLUMNS,
            title=f"Heart Disease — SHAP Summary ({best_model_name})",
            save_path=os.path.join(save_dir, "heart_shap_summary.png"),
        )
    except Exception as e:
        print(f"[WARNING] SHAP analysis skipped: {e}")

    print("\n" + "=" * 60)
    print("  ✓ Heart Disease model training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
