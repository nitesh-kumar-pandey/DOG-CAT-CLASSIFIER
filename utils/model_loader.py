"""utils/model_loader.py — Load saved Keras model safely."""

import os
import tensorflow as tf


MODEL_PATH = "./models/cnn_model.h5"
BEST_PATH  = "./models/best_model.h5"


def load_model():
    """
    Try to load best_model.h5, fall back to cnn_model.h5.
    Returns None if neither exists (user must run train.py first).
    """
    for path in [BEST_PATH, MODEL_PATH]:
        if os.path.exists(path):
            try:
                model = tf.keras.models.load_model(path)
                print(f"✅ Loaded model from: {path}")
                return model
            except Exception as e:
                print(f"⚠️  Failed to load {path}: {e}")
    return None
