from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

ML_ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = ML_ROOT / "data" / "splits"
MODEL_SAVE_DIR = ML_ROOT / "models"
MODEL_SAVE_PATH = MODEL_SAVE_DIR / "plant_model.keras"
CLASS_NAMES_PATH = MODEL_SAVE_DIR / "class_names.txt"
TRAINING_SUMMARY_PATH = MODEL_SAVE_DIR / "training_summary.json"
TRAINING_HISTORY_PATH = MODEL_SAVE_DIR / "training_history.json"


def get_class_names(train_dir: Path) -> list[str]:
    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    return sorted([d.name for d in train_dir.iterdir() if d.is_dir()])


def compute_class_weights(train_flow) -> dict[int, float]:
    counts = np.bincount(train_flow.classes)
    total = counts.sum()
    num_classes = len(counts)
    class_weights = {
        idx: float(total / (num_classes * count))
        for idx, count in enumerate(counts)
        if count > 0
    }
    return class_weights


def build_model(
    img_size: tuple[int, int], num_classes: int, base_dropout: float
) -> tuple[Model, Model]:
    base = MobileNetV2(
        input_shape=(*img_size, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    base.trainable = False

    x = base.output
    x = layers.Dropout(base_dropout)(x)
    x = layers.Dense(192, activation="relu")(x)
    x = layers.Dropout(0.35)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=base.input, outputs=out)
    return model, base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train medicinal leaf classifier with MobileNetV2 transfer learning."
    )
    parser.add_argument("--data-dir", type=Path, default=SPLITS_DIR)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-width", type=int, default=224)
    parser.add_argument("--img-height", type=int, default=224)
    parser.add_argument("--initial-epochs", type=int, default=10)
    parser.add_argument("--fine-tune-epochs", type=int, default=8)
    parser.add_argument("--initial-lr", type=float, default=1e-4)
    parser.add_argument("--fine-tune-lr", type=float, default=1e-5)
    parser.add_argument("--fine-tune-last-n", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    img_size = (args.img_width, args.img_height)

    train_dir = args.data_dir / "train"
    val_dir = args.data_dir / "val"
    test_dir = args.data_dir / "test"
    if not train_dir.exists() or not val_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            f"Expected train/val/test under {args.data_dir}. Run preprocessing first."
        )

    class_names = get_class_names(train_dir)
    num_classes = len(class_names)
    if num_classes < 2:
        raise ValueError("Need at least 2 classes for classification training.")

    print(f"Classes: {class_names}")

    train_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=(0.85, 1.15),
        fill_mode="nearest",
    )
    eval_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_flow = train_gen.flow_from_directory(
        str(train_dir),
        target_size=img_size,
        batch_size=args.batch_size,
        class_mode="categorical",
        shuffle=True,
        seed=args.seed,
    )
    val_flow = eval_gen.flow_from_directory(
        str(val_dir),
        target_size=img_size,
        batch_size=args.batch_size,
        class_mode="categorical",
        shuffle=False,
    )
    test_flow = eval_gen.flow_from_directory(
        str(test_dir),
        target_size=img_size,
        batch_size=args.batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    class_names_by_index = [
        name for name, _ in sorted(train_flow.class_indices.items(), key=lambda item: item[1])
    ]
    class_weights = compute_class_weights(train_flow)
    print(f"Class weights: {class_weights}")

    model, base_model = build_model(img_size=img_size, num_classes=num_classes, base_dropout=0.25)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.initial_lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = MODEL_SAVE_DIR / "best_model.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.4, patience=2, min_lr=1e-7, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint_path), monitor="val_accuracy", save_best_only=True, verbose=1
        ),
    ]

    print("Stage 1: training classifier head...")
    history_initial = model.fit(
        train_flow,
        validation_data=val_flow,
        epochs=args.initial_epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    print("Stage 2: fine-tuning tail of MobileNetV2...")
    base_model.trainable = True
    for layer in base_model.layers[:-args.fine_tune_last_n]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.fine_tune_lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    history_finetune = model.fit(
        train_flow,
        validation_data=val_flow,
        initial_epoch=len(history_initial.history["loss"]),
        epochs=len(history_initial.history["loss"]) + args.fine_tune_epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    test_loss, test_acc = model.evaluate(test_flow, verbose=1)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    model.save(MODEL_SAVE_PATH)
    with CLASS_NAMES_PATH.open("w", encoding="utf-8") as f:
        f.write("\n".join(class_names_by_index))

    training_summary = {
        "classes": class_names_by_index,
        "class_weights": class_weights,
        "initial_epochs_ran": len(history_initial.history.get("loss", [])),
        "fine_tune_epochs_ran": len(history_finetune.history.get("loss", [])),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "best_checkpoint": str(checkpoint_path),
    }
    with TRAINING_SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(training_summary, f, indent=2)

    merged_history = {
        "accuracy": history_initial.history.get("accuracy", []) + history_finetune.history.get("accuracy", []),
        "val_accuracy": history_initial.history.get("val_accuracy", []) + history_finetune.history.get("val_accuracy", []),
        "loss": history_initial.history.get("loss", []) + history_finetune.history.get("loss", []),
        "val_loss": history_initial.history.get("val_loss", []) + history_finetune.history.get("val_loss", []),
    }
    with TRAINING_HISTORY_PATH.open("w", encoding="utf-8") as f:
        json.dump(merged_history, f, indent=2)

    print(f"Model saved to: {MODEL_SAVE_PATH}")
    print(f"Class names saved to: {CLASS_NAMES_PATH}")
    print(f"Training summary saved to: {TRAINING_SUMMARY_PATH}")
    print(f"Training history saved to: {TRAINING_HISTORY_PATH}")


if __name__ == "__main__":
    main()
