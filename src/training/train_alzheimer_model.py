"""
Alzheimer's Disease Model Training Script
============================================
Trains CNN and VGG16 Transfer Learning models on brain MRI images.
- Evaluates models on validation set
- Saves the best-performing model to saved_models/
- Generates training history plots

Usage:
    python -m src.training.train_alzheimer_model
    OR
    python src/training/train_alzheimer_model.py

Requirements:
    Download the Alzheimer MRI dataset from Kaggle and place it in:
    datasets/alzheimers/
    Expected structure:
        datasets/alzheimers/
            NonDemented/
            VeryMildDemented/
            MildDemented/
            ModerateDemented/
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Ensure project root is in path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing.alzheimer_preprocessing import get_data_generators, IMG_HEIGHT, IMG_WIDTH
from src.models.alzheimer_model import build_custom_cnn, build_vgg16_transfer


def plot_training_history(history, model_name: str, save_dir: str) -> None:
    """Plot and save training/validation accuracy and loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history.history["accuracy"], label="Train Accuracy", color="#2196F3", linewidth=2)
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy", color="#FF5722", linewidth=2)
    axes[0].set_title(f"{model_name} — Accuracy", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history["loss"], label="Train Loss", color="#2196F3", linewidth=2)
    axes[1].plot(history.history["val_loss"], label="Val Loss", color="#FF5722", linewidth=2)
    axes[1].set_title(f"{model_name} — Loss", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"alzheimer_{model_name.lower().replace(' ', '_')}_history.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[INFO] Training history plot saved to {save_path}")
    plt.close(fig)


def main():
    print("\n" + "=" * 60)
    print("  MedPredict — Alzheimer's Disease Model Training")
    print("=" * 60)

    data_dir = os.path.join(PROJECT_ROOT, "datasets", "alzheimers")

    # Check if dataset exists
    if not os.path.exists(data_dir) or len(os.listdir(data_dir)) <= 1:
        print("\n[ERROR] Alzheimer MRI dataset not found!")
        print(f"  Expected location: {data_dir}")
        print("  Please download from Kaggle:")
        print("  https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images")
        print("\n  Expected structure:")
        print("    datasets/alzheimers/")
        print("      NonDemented/")
        print("      VeryMildDemented/")
        print("      MildDemented/")
        print("      ModerateDemented/")
        return

    save_dir = os.path.join(PROJECT_ROOT, "saved_models")
    os.makedirs(save_dir, exist_ok=True)

    # ── Step 1: Load data generators ──
    print("\n[STEP 1] Loading and preprocessing MRI images...")
    train_gen, val_gen, class_names = get_data_generators(data_dir)
    print(f"  Classes: {class_names}")

    # ── Step 2: Train Custom CNN ──
    print("\n[STEP 2] Training Custom CNN...")
    cnn_model = build_custom_cnn()
    cnn_model.summary()

    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1),
        ModelCheckpoint(
            os.path.join(save_dir, "alzheimer_cnn_best.h5"),
            monitor="val_accuracy", save_best_only=True, verbose=1,
        ),
    ]

    cnn_history = cnn_model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=50,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate CNN
    cnn_loss, cnn_acc = cnn_model.evaluate(val_gen, verbose=0)
    print(f"\n  Custom CNN — Val Accuracy: {cnn_acc:.4f}, Val Loss: {cnn_loss:.4f}")
    plot_training_history(cnn_history, "Custom CNN", save_dir)

    # ── Step 3: Train VGG16 Transfer Learning ──
    print("\n[STEP 3] Training VGG16 Transfer Learning model...")
    vgg_model = build_vgg16_transfer()
    vgg_model.summary()

    vgg_callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1),
        ModelCheckpoint(
            os.path.join(save_dir, "alzheimer_vgg16_best.h5"),
            monitor="val_accuracy", save_best_only=True, verbose=1,
        ),
    ]

    vgg_history = vgg_model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=30,
        callbacks=vgg_callbacks,
        verbose=1,
    )

    # Evaluate VGG16
    vgg_loss, vgg_acc = vgg_model.evaluate(val_gen, verbose=0)
    print(f"\n  VGG16 Transfer — Val Accuracy: {vgg_acc:.4f}, Val Loss: {vgg_loss:.4f}")
    plot_training_history(vgg_history, "VGG16 Transfer", save_dir)

    # ── Step 4: Select and save best model ──
    print("\n[STEP 4] Selecting best model...")
    print(f"  Custom CNN   : {cnn_acc:.4f}")
    print(f"  VGG16 Transfer: {vgg_acc:.4f}")

    if vgg_acc >= cnn_acc:
        best_name = "VGG16 Transfer Learning"
        best_model = vgg_model
    else:
        best_name = "Custom CNN"
        best_model = cnn_model

    best_path = os.path.join(save_dir, "alzheimer_model.h5")
    best_model.save(best_path)
    print(f"\n  ★ Best model: {best_name}")
    print(f"  Saved to: {best_path}")

    print("\n" + "=" * 60)
    print("  ✓ Alzheimer's model training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
