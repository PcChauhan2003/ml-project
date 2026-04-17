from flask import Flask, render_template, request
import cv2
import numpy as np
import os
import json

# Ensure we use the correct Keras version if available
os.environ['TF_USE_LEGACY_KERAS'] = '1'

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    print(f"✅ TensorFlow version: {tf.__version__}")
except ImportError:
    from keras.models import load_model
    print("⚠️ Using standalone Keras import")

app = Flask(__name__)

# 🔥 DEBUG: check files in Render
print("FILES IN DIR:", os.listdir())

def fix_and_load_model(model_path):
    """
    Attempts to load the model. If it fails due to Keras 3 metadata 
    (batch_shape/optional), it attempts to load without compilation.
    """
    try:
        # Try loading normally first
        return load_model(model_path, compile=False)
    except Exception as e:
        print(f"⚠️ Initial load failed: {e}")
        print("🔄 Attempting legacy compatibility load...")
        try:
            # Setting safe_mode=False can sometimes bypass metadata strictness in newer TF versions
            return load_model(model_path, compile=False, safe_mode=False)
        except Exception as e2:
            print(f"❌ Comprehensive load failure: {e2}")
            return None

# ✅ LOAD MODEL
MODEL_PATH = "cancer_model_v2.h5"
model = fix_and_load_model(MODEL_PATH)

if model:
    print("✅ Model loaded successfully")
else:
    print("❌ Model could not be initialized. Check your TensorFlow/Keras versions.")

# ✅ HOME
@app.route('/')
def home():
    return render_template('index.html')

# ✅ PREDICT
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "❌ Server Error: Model not loaded. Check Render logs."

    file = request.files.get('file')

    if file is None or file.filename == "":
        return render_template(
            'index.html',
            warning="⚠️ Please upload an image"
        )

    try:
        # Read image
        img_array = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            return render_template(
                'index.html',
                warning="⚠️ Invalid image file"
            )

        # Preprocess
        # Ensure these dimensions (64, 64) match your model's training input
        img = cv2.resize(img, (64, 64))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)

        # Prediction
        preds = model.predict(img)
        # Handle both single-output (sigmoid) and multi-output (softmax) models
        pred = preds[0][0] if preds.shape[-1] == 1 else np.max(preds[0])

        if pred > 0.5:
            result = "🛑 Cancer Detected"
            prob = round(float(pred) * 100, 2)
            color = "red"
        else:
            result = "✅ No Cancer"
            prob = round(float(1 - pred) * 100, 2)
            color = "lime"

        return render_template(
            'index.html',
            prediction=result,
            prob=prob,
            color=color
        )

    except Exception as e:
        print(f"❌ Prediction Error: {e}")
        return f"❌ Prediction error: {str(e)}"

# ✅ RUN APP
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)