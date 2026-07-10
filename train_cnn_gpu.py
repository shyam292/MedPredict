"""
MedPredict — Alzheimer's CNN Training Script (GPU Optimized)
==============================================================
Trains a Custom CNN on brain MRI images for Alzheimer's classification.
- Automatically detects and uses GPU if available
- Optimized for NVIDIA RTX 2050 (4GB VRAM)
- Saves the trained model to saved_models/alzheimer_model.h5

Usage:
    python train_cnn_gpu.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# ══════════════════════════════════════════════════════════════
# GPU CONFIGURATION
# ══════════════════════════════════════════════════════════════

def setup_gpu():
    """Configure TensorFlow to use GPU with memory growth."""
    import tensorflow as tf

    print("\n" + "=" * 60)
    print("  GPU / Device Configuration")
    print("=" * 60)
    print(f"  TensorFlow Version: {tf.__version__}")
    print(f"  Built with CUDA  : {tf.test.is_built_with_cuda()}")

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"  GPUs Detected     : {len(gpus)}")
        for gpu in gpus:
            print(f"    → {gpu.name}")
        # Enable memory growth to avoid OOM
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"  Memory Growth     : Enabled for {gpu.name}")
            except RuntimeError as e:
                print(f"  [WARNING] {e}")
    else:
        print("  GPUs Detected     : None (using CPU)")
        print("  [INFO] Training will proceed on CPU.")
        print("  [TIP]  For GPU support on Windows, use TF 2.10 with CUDA 11.x")
        print("         or install WSL2 with TensorFlow-GPU.")

    # Also check for DirectML devices
    try:
        dml_devices = tf.config.list_physical_devices('DML')
        if dml_devices:
            print(f"  DirectML Devices  : {len(dml_devices)}")
    except Exception:
        pass

    print("=" * 60)
    return gpus


def main():
    print("\n" + "=" * 60)
    print("  MedPredict -- Alzheimer's CNN Training")
    print("  GPU-Optimized Training Script")
    print("=" * 60)

    # Setup GPU
    gpus = setup_gpu()

    import tensorflow as tf
    from tensorflow.keras.callbacks import (
        EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    )
    from src.preprocessing.alzheimer_preprocessing import get_data_generators
    from src.models.alzheimer_model import build_custom_cnn

    data_dir = os.path.join(PROJECT_ROOT, "datasets", "alzheimers")
    save_dir = os.path.join(PROJECT_ROOT, "saved_models")
    os.makedirs(save_dir, exist_ok=True)

    # Check dataset
    if not os.path.exists(data_dir):
        print("\n[ERROR] Dataset not found!")
        print(f"  Expected: {data_dir}")
        return

    subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    if len(subdirs) < 4:
        print("\n[ERROR] Incomplete dataset!")
        print(f"  Found: {subdirs}")
        print("  Expected: NonDemented, VeryMildDemented, MildDemented, ModerateDemented")
        return

    # ── Step 1: Load data ──
    print("\n[STEP 1/4] Loading MRI dataset...")
    # Adjust batch size based on device
    batch_size = 16 if not gpus else 32
    train_gen, val_gen, class_names = get_data_generators(data_dir, batch_size=batch_size)
    print(f"  Batch size: {batch_size}")
    print(f"  Classes: {class_names}")
    print(f"  Training batches: {len(train_gen)}")
    print(f"  Validation batches: {len(val_gen)}")

    # ── Step 2: Build CNN ──
    print("\n[STEP 2/4] Building Custom CNN model...")
    model = build_custom_cnn()
    total_params = model.count_params()
    print(f"  Total parameters: {total_params:,}")
    print(f"  Model size (est): {total_params * 4 / (1024**2):.1f} MB")

    # Print model summary
    model.summary()

    # ── Step 3: Train ──
    print("\n[STEP 3/4] Training CNN...")

    # Determine epochs — if existing best model exists, do fewer epochs
    existing_best = os.path.join(save_dir, "alzheimer_cnn_best.h5")
    if os.path.exists(existing_best):
        print(f"  [INFO] Found existing checkpoint: alzheimer_cnn_best.h5")
        print(f"  [INFO] Loading weights from checkpoint...")
        try:
            model.load_weights(existing_best)
            epochs = 15  # Fine-tune with fewer epochs
            print(f"  [INFO] Loaded! Fine-tuning for {epochs} epochs...")
        except Exception as e:
            print(f"  [WARNING] Could not load weights: {e}")
            print(f"  [INFO] Training from scratch...")
            epochs = 30
    else:
        epochs = 30

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            os.path.join(save_dir, "alzheimer_cnn_best.h5"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
    ]

    # Enable mixed precision for faster training (if GPU available)
    if gpus:
        try:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            print("  [INFO] Mixed precision (FP16) enabled for GPU acceleration")
        except Exception:
            pass

    print(f"\n  >> Starting training for {epochs} epochs...\n")

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Step 4: Evaluate & Save ──
    print("\n[STEP 4/4] Evaluating and saving model...")

    val_loss, val_acc = model.evaluate(val_gen, verbose=0)
    print(f"\n  +======================================+")
    print(f"  |  Final Results                       |")
    print(f"  +======================================+")
    print(f"  |  Validation Accuracy : {val_acc:.4f}         |")
    print(f"  |  Validation Loss     : {val_loss:.4f}         |")
    print(f"  +======================================+")

    # Save as the main model
    final_path = os.path.join(save_dir, "alzheimer_model.h5")
    model.save(final_path)
    print(f"\n  [OK] Model saved to: {final_path}")

    # Plot training history
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Accuracy
        axes[0].plot(history.history["accuracy"], label="Train Accuracy", color="#6366f1", linewidth=2)
        axes[0].plot(history.history["val_accuracy"], label="Val Accuracy", color="#ef4444", linewidth=2)
        axes[0].set_title("CNN — Accuracy", fontsize=13, fontweight="bold")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Accuracy")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Loss
        axes[1].plot(history.history["loss"], label="Train Loss", color="#6366f1", linewidth=2)
        axes[1].plot(history.history["val_loss"], label="Val Loss", color="#ef4444", linewidth=2)
        axes[1].set_title("CNN — Loss", fontsize=13, fontweight="bold")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(save_dir, "alzheimer_cnn_training_history.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"  [PLOT] Training history plot saved to: {plot_path}")
        plt.close(fig)
    except Exception as e:
        print(f"  [WARNING] Could not save training plot: {e}")

    print("\n" + "=" * 60)
    print("  [DONE] Alzheimer's CNN training COMPLETE!")
    print(f"  Model: {final_path}")
    print(f"  Accuracy: {val_acc*100:.1f}%")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
