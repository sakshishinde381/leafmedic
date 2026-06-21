import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers
from tensorflow.keras.models import load_model

ML_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ML_ROOT / "models" / "plant_model.keras"
LEGACY_MODEL_PATH = ML_ROOT / "models" / "plant_model.h5"
LABELS_PATH = ML_ROOT / "models" / "class_names.txt"
IMG_SIZE = (224, 224)


def load_class_names(path: Path) -> list[str]:
    if not path.exists():
        return []
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def extract_leaf_bbox(image: Image.Image) -> tuple[int, int, int, int] | None:
    hsv = np.asarray(image.convert("HSV"), dtype=np.uint8)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    mask = (hue >= 25) & (hue <= 120) & (sat >= 25) & (val >= 25)
    if not np.any(mask):
        return None

    ys, xs = np.where(mask)
    y0 = int(ys.min())
    y1 = int(ys.max())
    x0 = int(xs.min())
    x1 = int(xs.max())
    h, w = mask.shape

    pad_x = max(4, int((x1 - x0 + 1) * 0.1))
    pad_y = max(4, int((y1 - y0 + 1) * 0.1))
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(w - 1, x1 + pad_x)
    y1 = min(h - 1, y1 + pad_y)

    if (x1 - x0) < max(20, int(w * 0.1)) or (y1 - y0) < max(20, int(h * 0.1)):
        return None
    return x0, y0, x1 + 1, y1 + 1


def prepare_image_tensor(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as im:
        image = ImageOps.exif_transpose(im).convert("RGB")
        bbox = extract_leaf_bbox(image)
        if bbox is not None:
            image = image.crop(bbox)
        image = image.resize(IMG_SIZE, Image.Resampling.LANCZOS)
    arr = np.asarray(image, dtype=np.float32)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def predict_image(model, class_names: list[str], image_path: Path) -> dict:
    if not class_names:
        return {"plant": "unknown", "confidence": 0.0, "index": -1}

    arr = prepare_image_tensor(image_path)

    preds = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(preds))
    conf = float(preds[idx])
    return {
        "plant": class_names[idx],
        "confidence": conf,
        "index": idx,
    }


def main() -> None:
    selected_model_path = MODEL_PATH if MODEL_PATH.exists() else LEGACY_MODEL_PATH
    if not selected_model_path.exists():
        print(f"Model not found: {MODEL_PATH} or {LEGACY_MODEL_PATH}")
        print("Run ml/src/train.py first.")
        return

    if not LABELS_PATH.exists():
        print(f"Labels file not found: {LABELS_PATH}")
        return

    if len(sys.argv) > 1:
        image_paths = [Path(p) for p in sys.argv[1:] if Path(p).exists()]
    else:
        test_dir = ML_ROOT / "data" / "splits" / "test"
        image_paths = []
        if test_dir.exists():
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                image_paths.extend(test_dir.rglob(ext))
        if not image_paths:
            print("No images provided and none found in ml/data/splits/test/")
            print("Usage: python ml/src/test_model.py [image1.jpg] [image2.png]")
            return

    class DenseCompat(layers.Dense):
        def __init__(self, *args, **kwargs):
            kwargs.pop("quantization_config", None)
            super().__init__(*args, **kwargs)

    class InputLayerCompat(layers.InputLayer):
        def __init__(self, *args, **kwargs):
            kwargs.pop("optional", None)
            if "batch_shape" in kwargs and "batch_input_shape" not in kwargs:
                kwargs["batch_input_shape"] = kwargs.pop("batch_shape")
            super().__init__(*args, **kwargs)

    model = load_model(
        selected_model_path,
        custom_objects={"Dense": DenseCompat, "InputLayer": InputLayerCompat},
        compile=False,
    )
    class_names = load_class_names(LABELS_PATH)

    print(f"Model: {selected_model_path}")
    print(f"Class names: {class_names}")
    print("-" * 50)

    for img_path in image_paths[:10]:
        result = predict_image(model, class_names, img_path)
        print(f"{img_path.name} -> {result['plant']} ({result['confidence']*100:.1f}%)")


if __name__ == "__main__":
    main()
