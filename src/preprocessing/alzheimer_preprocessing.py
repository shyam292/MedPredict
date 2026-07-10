"""
Alzheimer's MRI Dataset Preprocessing Module
==============================================
Handles loading and preprocessing of brain MRI images for Alzheimer's classification.
- Expects a folder-per-class directory layout
- Resizes images to 176×208
- Normalizes pixel values to [0, 1]
- Applies data augmentation via ImageDataGenerator
- Provides train/val/test generators
"""

import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
IMG_HEIGHT = 208
IMG_WIDTH = 176
BATCH_SIZE = 32
CLASS_NAMES = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]
NUM_CLASSES = len(CLASS_NAMES)


def get_data_generators(
    data_dir: str | None = None,
    img_size: tuple = (IMG_HEIGHT, IMG_WIDTH),
    batch_size: int = BATCH_SIZE,
    validation_split: float = 0.2,
) -> tuple:
    """
    Create train and validation data generators from the dataset directory.

    Expected directory structure:
        data_dir/
            NonDemented/
            VeryMildDemented/
            MildDemented/
            ModerateDemented/

    Returns: (train_generator, val_generator, class_names)
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "datasets", "alzheimers")
    data_dir = os.path.abspath(data_dir)

    # Training augmentation generator
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=validation_split,
    )

    # Validation generator (only rescaling)
    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=validation_split,
    )

    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=42,
    )

    val_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=42,
    )

    class_names = list(train_generator.class_indices.keys())
    print(f"[INFO] Classes found: {class_names}")
    print(f"[INFO] Training samples: {train_generator.samples}")
    print(f"[INFO] Validation samples: {val_generator.samples}")

    return train_generator, val_generator, class_names


def preprocess_single_image(image_array: np.ndarray, img_size: tuple = (IMG_HEIGHT, IMG_WIDTH)) -> np.ndarray:
    """
    Preprocess a single image for prediction:
    - Resize to target size
    - Normalize to [0, 1]
    - Add batch dimension
    """
    from tensorflow.keras.preprocessing.image import img_to_array
    from PIL import Image

    if isinstance(image_array, np.ndarray) and image_array.ndim == 3:
        img = Image.fromarray(image_array)
    else:
        img = image_array

    img = img.resize((img_size[1], img_size[0]))  # PIL uses (width, height)
    img_arr = img_to_array(img) / 255.0

    # Ensure 3 channels (RGB)
    if img_arr.shape[-1] == 1:
        img_arr = np.repeat(img_arr, 3, axis=-1)
    elif img_arr.shape[-1] == 4:
        img_arr = img_arr[:, :, :3]

    return np.expand_dims(img_arr, axis=0)
