"""utils/predictor.py — Inference and Grad-CAM explainability."""

import numpy as np
import cv2
from PIL import Image
import tensorflow as tf


IMG_SIZE = 224


def preprocess(img: Image.Image) -> np.ndarray:
    """Convert PIL image to model-ready tensor."""
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)          # (1, 224, 224, 3)


def predict_image(model, img: Image.Image):
    """
    Returns:
        label      : str  ("Dog" or "Cat")
        confidence : float (0-1, always >= 0.5)
        raw_prob   : float (sigmoid output, 0=dog, 1=cat)
    """
    tensor = preprocess(img)
    raw_prob = float(model.predict(tensor, verbose=0)[0][0])
    if raw_prob >= 0.5:
        label = "Cat"
        confidence = raw_prob
    else:
        label = "Dog"
        confidence = 1.0 - raw_prob
    return label, confidence, raw_prob


def get_grad_cam(model, img: Image.Image) -> Image.Image | None:
    """
    Generate a Grad-CAM heatmap blended with the original image.
    Works for models that have a Conv2D layer near the end of the base.
    Returns a PIL image or None if no suitable layer is found.
    """
    # Find last Conv2D layer
    conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            conv_layer = layer
            break

    if conv_layer is None:
        # Try looking inside a sub-model (e.g. EfficientNetB0 nested)
        for layer in reversed(model.layers):
            if hasattr(layer, "layers"):
                for sub in reversed(layer.layers):
                    if isinstance(sub, tf.keras.layers.Conv2D):
                        conv_layer = sub
                        break
            if conv_layer:
                break

    if conv_layer is None:
        return None

    try:
        # Build grad model up to that conv layer + final output
        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[conv_layer.output, model.output]
        )

        tensor = preprocess(img)
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(tensor, training=False)
            loss = predictions[:, 0]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]

        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap).numpy()
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() != 0:
            heatmap /= heatmap.max()

        # Resize and colorize
        orig = np.array(img.resize((IMG_SIZE, IMG_SIZE)))
        heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        superimposed = (0.6 * orig + 0.4 * heatmap_colored).astype(np.uint8)
        return Image.fromarray(superimposed)

    except Exception as e:
        print(f"Grad-CAM failed: {e}")
        return None
