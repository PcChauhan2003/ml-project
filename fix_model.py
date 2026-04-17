from tensorflow.keras.models import load_model

print("Loading model...")
model = load_model("cancer_model.h5", compile=False)

print("Saving fixed model...")
model.save("cancer_model.h5")

print("✅ Model fixed successfully!")