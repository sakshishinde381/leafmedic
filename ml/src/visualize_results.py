import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
from keras.applications.mobilenet_v2 import preprocess_input
from keras.src.legacy.preprocessing.image import ImageDataGenerator

ML_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLITS_DIR = ML_ROOT / "data" / "splits"
DEFAULT_MODELS_DIR = ML_ROOT / "models"
DEFAULT_OUTPUT_DIR = ML_ROOT / "reports" / "figures"
DEFAULT_CLASS_NAMES_PATH = DEFAULT_MODELS_DIR / "class_names.txt"


def load_keras_model(model_path: Path):
    return keras.saving.load_model(model_path, compile=False)


def get_class_names(train_dir: Path) -> list[str]:
    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    return sorted([d.name for d in train_dir.iterdir() if d.is_dir()])


def load_class_names(train_dir: Path, class_names_path: Path) -> list[str]:
    if train_dir.exists():
        class_names = get_class_names(train_dir)
        if class_names:
            return class_names

    if class_names_path.exists():
        class_names = [
            line.strip()
            for line in class_names_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if class_names:
            return class_names

    raise FileNotFoundError(
        f"No class names found in {train_dir} or {class_names_path}."
    )


def dataset_distribution(splits_dir: Path, class_names: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {name: 0 for name in class_names}
    for split in ["train", "val", "test"]:
        split_dir = splits_dir / split
        if not split_dir.exists():
            continue
        for class_name in class_names:
            class_dir = split_dir / class_name
            if class_dir.exists():
                counts[class_name] += len([p for p in class_dir.iterdir() if p.is_file()])
    return counts


def save_dataset_distribution_chart(counts: dict[str, int], out_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    classes = list(counts.keys())
    values = list(counts.values())
    ax = sns.barplot(x=classes, y=values, palette="viridis")
    for i, v in enumerate(values):
        ax.text(i, v + max(values) * 0.01, str(v), ha="center", va="bottom", fontsize=10)
    plt.title("Dataset Class Distribution", fontsize=14, weight="bold")
    plt.xlabel("Plant Class", fontsize=12)
    plt.ylabel("Number of Images", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_training_accuracy_plot(history: dict, out_path: Path) -> None:
    epochs = range(1, len(history.get("accuracy", [])) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history.get("accuracy", []), label="Training Accuracy", linewidth=2)
    plt.plot(epochs, history.get("val_accuracy", []), label="Validation Accuracy", linewidth=2)
    plt.title("Training vs Validation Accuracy", fontsize=14, weight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.legend()
    plt.grid(linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_training_loss_plot(history: dict, out_path: Path) -> None:
    epochs = range(1, len(history.get("loss", [])) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history.get("loss", []), label="Training Loss", linewidth=2)
    plt.plot(epochs, history.get("val_loss", []), label="Validation Loss", linewidth=2)
    plt.title("Training vs Validation Loss", fontsize=14, weight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend()
    plt.grid(linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_confusion_matrix_plot(
    model,
    test_dir: Path,
    class_names: list[str],
    img_size: tuple[int, int],
    batch_size: int,
    out_path: Path,
) -> None:
    test_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_flow = test_gen.flow_from_directory(
        str(test_dir),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    preds = model.predict(test_flow, verbose=0)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_flow.classes
    n = len(class_names)
    cm = np.zeros((n, n), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)

    plt.figure(figsize=(8, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
    )
    plt.title("Confusion Matrix on Test Dataset", fontsize=14, weight="bold")
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_sample_prediction_plot(
    model,
    sample_image_path: Path,
    class_names: list[str],
    img_size: tuple[int, int],
    out_path: Path,
) -> None:
    image = Image.open(sample_image_path).convert("RGB")
    display_image = image.copy()
    resized = image.resize(img_size, Image.Resampling.LANCZOS)
    arr = np.asarray(resized, dtype=np.float32)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    probs = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(probs))
    pred_label = class_names[idx]
    pred_conf = float(probs[idx])

    plt.figure(figsize=(7, 7))
    plt.imshow(display_image)
    plt.axis("off")
    plt.title(
        f"Sample Prediction\nPredicted: {pred_label} (Confidence: {pred_conf:.2%})",
        fontsize=13,
        weight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def find_default_sample(test_dir: Path) -> Path:
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]:
        files = list(test_dir.rglob(ext))
        if files:
            return files[0]
    raise FileNotFoundError(f"No sample image found in {test_dir}")


def resolve_history_path(history_path: Path, models_dir: Path) -> Path | None:
    if history_path.exists():
        return history_path

    candidates = [
        models_dir / "training_history.json",
        models_dir / "training_summary.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate report-ready visualization figures.")
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--history-path", type=Path, default=DEFAULT_MODELS_DIR / "training_history.json")
    parser.add_argument("--class-names-path", type=Path, default=DEFAULT_CLASS_NAMES_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-image", type=Path, default=None)
    parser.add_argument("--img-width", type=int, default=224)
    parser.add_argument("--img-height", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dir = args.splits_dir / "train"
    test_dir = args.splits_dir / "test"
    class_names = load_class_names(train_dir, args.class_names_path)

    model_path = args.models_dir / "plant_model.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = load_keras_model(model_path)

    history_path = resolve_history_path(args.history_path, args.models_dir)
    history = None
    if history_path is not None:
        with history_path.open("r", encoding="utf-8") as f:
            history = json.load(f)
    else:
        print(
            "Warning: training history file not found. "
            "Skipping accuracy/loss plots."
        )

    counts = dataset_distribution(args.splits_dir, class_names)
    save_dataset_distribution_chart(counts, output_dir / "dataset_class_distribution.png")
    if history and {"accuracy", "val_accuracy", "loss", "val_loss"} <= set(history):
        save_training_accuracy_plot(history, output_dir / "training_validation_accuracy.png")
        save_training_loss_plot(history, output_dir / "training_validation_loss.png")
    elif history is not None:
        print(
            f"Warning: {history_path.name} does not contain training curve data. "
            "Skipping accuracy/loss plots."
        )

    if test_dir.exists():
        save_confusion_matrix_plot(
            model=model,
            test_dir=test_dir,
            class_names=class_names,
            img_size=(args.img_width, args.img_height),
            batch_size=args.batch_size,
            out_path=output_dir / "confusion_matrix_test.png",
        )

        sample_image = args.sample_image if args.sample_image else find_default_sample(test_dir)
        save_sample_prediction_plot(
            model=model,
            sample_image_path=sample_image,
            class_names=class_names,
            img_size=(args.img_width, args.img_height),
            out_path=output_dir / "sample_prediction.png",
        )
    else:
        print(f"Warning: test directory not found at {test_dir}. Skipping test visualizations.")

    print(f"Saved figures in: {output_dir}")
    print(" - dataset_class_distribution.png")
    if history and {"accuracy", "val_accuracy", "loss", "val_loss"} <= set(history):
        print(" - training_validation_accuracy.png")
        print(" - training_validation_loss.png")
    if test_dir.exists():
        print(" - confusion_matrix_test.png")
        print(" - sample_prediction.png")


if __name__ == "__main__":
    main()
