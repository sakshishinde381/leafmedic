import os
import logging
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
import numpy as np
from PIL import Image, ImageOps
from flask import Flask, request, jsonify
from flask_cors import CORS
from keras.applications.mobilenet_v2 import preprocess_input
from werkzeug.utils import secure_filename

BACKEND_ROOT = Path(__file__).resolve().parent
MODEL_PATH = BACKEND_ROOT / "model" / "plant_model.keras"
LABELS_PATH = BACKEND_ROOT / "model" / "class_names.txt"
IMG_SIZE = (224, 224)
UNKNOWN_CLASS_NAME = "unknown"
UNKNOWN_INFO = "This leaf is not recognized as one of the supported medicinal plant classes."
MIN_CONFIDENCE = float(os.getenv("PREDICTION_CONFIDENCE_THRESHOLD", "0.6"))
SUPPORTED_CLASS_NAMES = {"aloe", "guava", "hibiscus", "mango", "neem", "tulsi", UNKNOWN_CLASS_NAME}
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "*").split(",")
    if origin.strip()
]

PLANT_INFO = {
    "aloe": [
        "Soothes minor burns and skin irritation",
        "Supports wound healing",
        "Hydrates dry skin",
        "Has anti-inflammatory properties",
        "Traditionally used to support digestion"
    ],
    "guava": [
        "Rich in antioxidants and Vitamin C",
        "Supports digestion and helps relieve diarrhea",
        "Has antibacterial properties that help fight infections",
        "Helps regulate blood sugar levels",
        "Traditionally used to improve oral and skin health"
    ],
    "hibiscus": [
        "Rich in antioxidants",
        "Traditionally used to support healthy blood pressure",
        "Supports hair and scalp care",
        "May help soothe inflammation",
        "Often used in herbal teas"
    ],
    "mango": [
        "Traditionally used to support digestion",
        "Contains antioxidant compounds",
        "Used in folk remedies for minor skin concerns",
        "May support blood sugar management",
        "Leaves are commonly used in herbal preparations"
    ],
    "neem": [
        "Antibacterial and antifungal",
        "Treats acne and skin infections",
        "Purifies blood",
        "Boosts immunity",
        "Supports oral health"
    ],
    "tulsi": [
        "Improves immunity",
        "Helps in cough and cold",
        "Reduces stress and anxiety",
        "Supports respiratory health",
        "Rich in antioxidants"
    ],
    UNKNOWN_CLASS_NAME: UNKNOWN_INFO,
}


ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
CORS(app, resources={r"/*": {"origins": CORS_ORIGINS}})

_model = None
_class_names = []


class ModelConfigurationError(RuntimeError):
    pass


def load_class_names(path: Path) -> list:
    if not path.exists():
        raise FileNotFoundError(f"Labels file not found: {path}")

    with path.open(encoding="utf-8") as f:
        class_names = [line.strip().lower() for line in f if line.strip()]

    if not class_names:
        raise ModelConfigurationError(f"Labels file is empty: {path}")

    duplicate_names = sorted({name for name in class_names if class_names.count(name) > 1})
    if duplicate_names:
        raise ModelConfigurationError(f"Duplicate class names in labels file: {duplicate_names}")

    if UNKNOWN_CLASS_NAME not in class_names:
        raise ModelConfigurationError(f"Labels file must include '{UNKNOWN_CLASS_NAME}'.")

    unsupported_names = sorted(set(class_names) - SUPPORTED_CLASS_NAMES)
    if unsupported_names:
        raise ModelConfigurationError(f"Unsupported class names in labels file: {unsupported_names}")

    missing_info = sorted(name for name in class_names if name not in PLANT_INFO)
    if missing_info:
        raise ModelConfigurationError(f"Missing PLANT_INFO entries for: {missing_info}")

    return class_names


def get_model_output_size(model) -> int:
    output_shape = model.output_shape
    if isinstance(output_shape, list):
        output_shape = output_shape[0]

    if not output_shape or output_shape[-1] is None:
        raise ModelConfigurationError(f"Could not determine model output size: {output_shape}")

    return int(output_shape[-1])


def get_model():
    global _model, _class_names
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

        _model = keras.saving.load_model(MODEL_PATH, compile=False)
        _class_names = load_class_names(LABELS_PATH)

        output_size = get_model_output_size(_model)
        if output_size != len(_class_names):
            raise ModelConfigurationError(
                f"Model output size ({output_size}) does not match labels count "
                f"({len(_class_names)}). Model: {MODEL_PATH}. Labels: {LABELS_PATH}."
            )
    return _model, _class_names


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


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


def predict_from_path(image_path: Path) -> dict:
    model, class_names = get_model()

    arr = prepare_image_tensor(image_path)
    preds = np.asarray(model.predict(arr, verbose=0)).squeeze()

    if preds.ndim != 1:
        return {
            "plant": UNKNOWN_CLASS_NAME,
            "confidence": 0.0,
            "info": "Model returned an invalid prediction shape.",
        }

    if len(preds) != len(class_names):
        return {
            "plant": UNKNOWN_CLASS_NAME,
            "confidence": 0.0,
            "info": "Model output and class names are out of sync.",
        }

    idx = int(np.argmax(preds))
    conf = float(preds[idx])

    plant = class_names[idx].lower()
    if plant not in PLANT_INFO:
        logging.error("Predicted class has no PLANT_INFO entry: %s", plant)
        plant = UNKNOWN_CLASS_NAME

    if plant.lower() != UNKNOWN_CLASS_NAME and conf < MIN_CONFIDENCE:
        plant = UNKNOWN_CLASS_NAME

    info = PLANT_INFO[plant]
    return {"plant": plant, "confidence": round(conf, 4), "info": info}


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "leafmedic-api"
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    logging.info("Request to /predict received")

    if "image" not in request.files and "file" not in request.files:
        return jsonify({"error": "No image provided. Use form key 'image' or 'file'."}), 400

    file = request.files.get("image") or request.files.get("file")

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid image type. Use jpg, png, bmp, or webp."}), 400

    try:
        get_model()

    except FileNotFoundError as e:
        logging.exception("Model file missing")
        return jsonify({
            "error": "Model not available.",
            "detail": repr(e)
        }), 503

    except Exception as e:
        logging.exception("Model loading failed")
        return jsonify({
            "error": "Model failed to load.",
            "detail": repr(e)
        }), 503

    try:
        filename = secure_filename(file.filename) or "upload.jpg"
        save_path = BACKEND_ROOT / "uploads"
        save_path.mkdir(exist_ok=True)

        full_path = save_path / filename
        file.save(str(full_path))

        result = predict_from_path(full_path)

        try:
            os.remove(full_path)
        except OSError:
            pass

        logging.info(f"Prediction: {result}")
        return jsonify(result)

    except Exception as e:
        logging.exception("Prediction failed")
        return jsonify({
            "error": "Prediction failed.",
            "detail": str(e)
        }), 500

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
