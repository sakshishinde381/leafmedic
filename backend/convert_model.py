import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras

SOURCE_MODEL_PATH = Path("model/plant_model.keras")
TARGET_MODEL_PATH = Path("model/plant_model.keras")

print("Loading model...")
model = keras.saving.load_model(SOURCE_MODEL_PATH, compile=False)

print("Saving Keras model...")
model.save(TARGET_MODEL_PATH)

print("Done!")
