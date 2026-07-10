"""
Alzheimer's Disease CNN Model Definitions
============================================
Provides factory functions for deep learning models:
- Custom CNN architecture
- VGG16 Transfer Learning model
"""

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Dropout, Flatten,
    BatchNormalization, GlobalAveragePooling2D, Input,
)
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
IMG_HEIGHT = 208
IMG_WIDTH = 176
NUM_CLASSES = 4  # NonDemented, VeryMildDemented, MildDemented, ModerateDemented


def build_custom_cnn(
    input_shape: tuple = (IMG_HEIGHT, IMG_WIDTH, 3),
    num_classes: int = NUM_CLASSES,
    learning_rate: float = 0.0001,
) -> Sequential:
    """
    Build a custom CNN architecture for brain MRI classification.

    Architecture:
    - 4 Conv blocks with BatchNorm and MaxPooling
    - Flatten → Dense(512) → Dropout → Dense(256) → Dropout → Output
    """
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Block 2
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Block 3
        Conv2D(128, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        # Block 4
        Conv2D(256, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        # Classifier head
        Flatten(),
        Dense(512, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def build_vgg16_transfer(
    input_shape: tuple = (IMG_HEIGHT, IMG_WIDTH, 3),
    num_classes: int = NUM_CLASSES,
    learning_rate: float = 0.0001,
    fine_tune_layers: int = 4,
) -> Model:
    """
    Build a VGG16 transfer learning model.
    Freezes all VGG16 layers except the last `fine_tune_layers`.
    Adds a custom classifier head with GlobalAveragePooling.
    """
    base_model = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )

    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Unfreeze last n layers for fine-tuning
    if fine_tune_layers > 0:
        for layer in base_model.layers[-fine_tune_layers:]:
            layer.trainable = True

    # Build classifier head
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
