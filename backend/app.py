import os
import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

BACKEND_ROOT = Path(__file__).resolve().parent
MODEL_PATH = BACKEND_ROOT / "model" / "plant_model.keras"
LEGACY_MODEL_PATH = BACKEND_ROOT / "model" / "plant_model.h5"
LABELS_PATH = BACKEND_ROOT / "model" / "class_names.txt"
IMG_SIZE = (224, 224)
UNKNOWN_CLASS_NAME = "unknown"
UNKNOWN_INFO = "This leaf is not recognized as one of the supported medicinal plant classes."
MIN_CONFIDENCE = float(os.getenv("PREDICTION_CONFIDENCE_THRESHOLD", "0.6"))
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "*").split(",")
    if origin.strip()
]

PLANT_INFO = {
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
    "aloe": [
        "Heals burns and wounds",
        "Soothes skin irritation",
        "Improves digestion",
        "Hydrates skin",
        "Anti-inflammatory"
    ],
    
    "guava": [
    "Rich in antioxidants and Vitamin C",
    "Supports digestion and helps relieve diarrhea",
    "Has antibacterial properties that help fight infections",
    "Helps regulate blood sugar levels",
    "Traditionally used to improve oral and skin health"
],
    UNKNOWN_CLASS_NAME: UNKNOWN_INFO
}


ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
CORS(app, resources={r"/*": {"origins": CORS_ORIGINS}})

_model = None
_class_names = []


def load_class_names(path: Path) -> list:
    if not path.exists():
        return []
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def get_model():
    global _model, _class_names
    if _model is None:
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

        model_path = MODEL_PATH if MODEL_PATH.exists() else LEGACY_MODEL_PATH
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH} or {LEGACY_MODEL_PATH}")

        _model = load_model(
            model_path,
            custom_objects={"Dense": DenseCompat, "InputLayer": InputLayerCompat},
            compile=False,
        )
        _class_names = load_class_names(LABELS_PATH)
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
    if not class_names:
        return {"plant": UNKNOWN_CLASS_NAME, "confidence": 0.0, "info": "Class names not loaded."}

    arr = prepare_image_tensor(image_path)
    preds = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(preds))
    conf = float(preds[idx])

    if idx >= len(class_names):
        return {
            "plant": UNKNOWN_CLASS_NAME,
            "confidence": round(conf, 4),
            "info": "Model output and class names are out of sync. Retrain or copy matching labels.",
        }

    plant = class_names[idx]

    if plant.lower() != UNKNOWN_CLASS_NAME and conf < MIN_CONFIDENCE:
        plant = UNKNOWN_CLASS_NAME

    info = PLANT_INFO.get(plant.lower(), "Medicinal plant.")
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
