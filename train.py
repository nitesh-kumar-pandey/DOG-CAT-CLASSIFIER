import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd

# ─── Config ─────────────────────────────────────────────
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS_P1 = 10
EPOCHS_P2 = 10
SEED = 42

DATA_DIR = Path("Data/images")   # Data/images/cats and Data/images/dogs
MODEL_DIR = Path("models")
PRED_DIR = Path("prediction")

MODEL_DIR.mkdir(exist_ok=True)
PRED_DIR.mkdir(exist_ok=True)

# Optional: reduce TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def load_data(data_dir):
    print("📂 Loading images...")
    X, Y = [], []

    class_map = {
        "dogs": 0,
        "cats": 1
    }

    for class_name, label in class_map.items():
        folder = Path(data_dir) / class_name

        if not folder.exists():
            print(f"⚠️ Folder not found: {folder}")
            continue

        files = (
            list(folder.glob("*.jpg")) +
            list(folder.glob("*.jpeg")) +
            list(folder.glob("*.png"))
        )

        print(f"→ {class_name}: {len(files)} images")

        for path in files:
            img = cv2.imread(str(path))

            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            X.append(img)
            Y.append(label)

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.int32)

    print(f"✅ Total: {len(X)} images | Labels: {np.bincount(Y) if len(Y) else []}")

    return X, Y


def get_augmentation_layer():
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ],
        name="augmentation"
    )


def build_model(freeze_base=True):
    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )

    base_model.trainable = not freeze_base

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    x = get_augmentation_layer()(inputs)
    x = tf.keras.applications.efficientnet.preprocess_input(x)

    x = base_model(x, training=False)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs, name="DogCat_EfficientNetB0")

    return model


def compile_model(model, lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc")
        ],
    )


def get_callbacks(checkpoint_path):
    return [
        EarlyStopping(
            monitor="val_auc",
            patience=5,
            restore_best_weights=True,
            mode="max",
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_auc",
            save_best_only=True,
            save_weights_only=False,
            mode="max",
            verbose=1
        )
    ]


def train():
    X, Y = load_data(DATA_DIR)

    if len(X) == 0:
        print("❌ No images found. Check folder path.")
        sys.exit(1)

    if len(np.unique(Y)) < 2:
        print("❌ Need both cat and dog images.")
        sys.exit(1)

    x_train, x_test, y_train, y_test = train_test_split(
        X,
        Y,
        test_size=0.15,
        random_state=SEED,
        stratify=Y
    )

    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=0.15,
        random_state=SEED,
        stratify=y_train
    )

    print(f"\n📊 Split — Train: {len(x_train)} | Val: {len(x_val)} | Test: {len(x_test)}")

    class_weights_arr = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )

    class_weights = {
        int(cls): float(weight)
        for cls, weight in zip(np.unique(y_train), class_weights_arr)
    }

    print(f"⚖️ Class weights: {class_weights}")

    checkpoint_path = MODEL_DIR / "best_model.keras"

    print("\n🔒 Phase 1: Training classification head...")
    model = build_model(freeze_base=True)
    compile_model(model, lr=1e-3)

    model.summary()

    h1 = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS_P1,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=get_callbacks(checkpoint_path),
        verbose=1
    )

    print("\n🔓 Phase 2: Fine-tuning top layers...")

    base_model = model.get_layer("efficientnetb0")
    base_model.trainable = True

    for layer in base_model.layers[:-30]:
        layer.trainable = False

    compile_model(model, lr=1e-5)

    h2 = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS_P2,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=get_callbacks(checkpoint_path),
        verbose=1
    )

    print("\n📈 Evaluating on test set...")

    best_model = tf.keras.models.load_model(checkpoint_path)

    loss, acc, auc = best_model.evaluate(x_test, y_test, verbose=0)

    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test AUC: {auc:.4f}")

    pred_probs = best_model.predict(x_test).ravel()
    preds = (pred_probs > 0.5).astype(int)

    print("\n📋 Classification Report:")
    print(classification_report(y_test, preds, target_names=["Dog", "Cat"]))

    cm = confusion_matrix(y_test, preds)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Dog", "Cat"],
        yticklabels=["Dog", "Cat"]
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(PRED_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    compare_df = pd.DataFrame(
        {
            "Actual": y_test,
            "Predicted": preds,
            "Probability": pred_probs
        }
    )

    compare_df.to_csv(PRED_DIR / "predicted.csv", index=False)

    final_path = MODEL_DIR / "cnn_model.keras"
    best_model.save(final_path)

    print(f"\n✅ Best model saved → {checkpoint_path}")
    print(f"✅ Final model saved → {final_path}")
    print(f"✅ Predictions saved → {PRED_DIR / 'predicted.csv'}")
    print(f"✅ Confusion matrix saved → {PRED_DIR / 'confusion_matrix.png'}")

    plot_history(h1, h2)


def plot_history(h1, h2):
    metrics = ["loss", "accuracy", "auc"]

    for metric in metrics:
        plt.figure(figsize=(7, 5))

        p1 = h1.history.get(metric, [])
        p2 = h2.history.get(metric, [])

        vp1 = h1.history.get(f"val_{metric}", [])
        vp2 = h2.history.get(f"val_{metric}", [])

        train_values = p1 + p2
        val_values = vp1 + vp2

        plt.plot(train_values, label=f"Train {metric}")
        plt.plot(val_values, label=f"Val {metric}")
        plt.axvline(len(p1) - 1, linestyle="--", label="Fine-tune start")

        plt.title(metric.upper())
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.legend()
        plt.tight_layout()

        save_path = PRED_DIR / f"{metric}_history.png"
        plt.savefig(save_path, dpi=150)
        plt.close()

    print(f"✅ Training graphs saved in → {PRED_DIR}")


if __name__ == "__main__":
    train()