from tensorflow.keras.models import load_model

print("Loading model...")
model = load_model("model/plant_model.keras", compile=False)

print("Saving H5 model...")
model.save("model/plant_model.h5")

print("Done!")
