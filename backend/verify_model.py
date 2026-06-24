import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras

from app import get_model


def main() -> None:
    model, class_names = get_model()
    print(f"Keras: {keras.__version__}")
    print(f"Model output shape: {model.output_shape}")
    print(f"Class names: {class_names}")


if __name__ == "__main__":
    main()
