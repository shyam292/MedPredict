"""
Quick script to convert existing CNN checkpoint to final model.
Loads alzheimer_cnn_best.h5 and saves it as alzheimer_model.h5
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Set encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'

print("\n" + "=" * 60)
print("  MedPredict - Convert CNN Checkpoint to Final Model")
print("=" * 60)

import tensorflow as tf
print(f"  TensorFlow: {tf.__version__}")

save_dir = os.path.join(PROJECT_ROOT, "saved_models")
checkpoint_path = os.path.join(save_dir, "alzheimer_cnn_best.h5")
final_path = os.path.join(save_dir, "alzheimer_model.h5")

if not os.path.exists(checkpoint_path):
    print(f"\n  [ERROR] Checkpoint not found: {checkpoint_path}")
    sys.exit(1)

print(f"\n  Loading checkpoint: alzheimer_cnn_best.h5")
print(f"  Size: {os.path.getsize(checkpoint_path) / (1024*1024):.1f} MB")

model = tf.keras.models.load_model(checkpoint_path)
print(f"  Model loaded successfully!")
print(f"  Total params: {model.count_params():,}")

# Evaluate on validation data
print(f"\n  Evaluating on validation data...")
from src.preprocessing.alzheimer_preprocessing import get_data_generators

data_dir = os.path.join(PROJECT_ROOT, "datasets", "alzheimers")
_, val_gen, class_names = get_data_generators(data_dir, batch_size=32)

val_loss, val_acc = model.evaluate(val_gen, verbose=1)
print(f"\n  Validation Accuracy: {val_acc:.4f} ({val_acc*100:.1f}%)")
print(f"  Validation Loss:    {val_loss:.4f}")

# Save as final model
model.save(final_path)
print(f"\n  [SAVED] alzheimer_model.h5")
print(f"  Size: {os.path.getsize(final_path) / (1024*1024):.1f} MB")

print("\n" + "=" * 60)
print("  DONE! alzheimer_model.h5 is ready for the Streamlit app")
print("=" * 60 + "\n")
