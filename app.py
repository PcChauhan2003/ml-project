from flask import Flask, render_template, request
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# 🔥 DEBUG: check files in Render
print("FILES IN DIR:", os.listdir())

# ✅ FINAL MODEL LOAD (NO custom_objects)
try:
    model = load_model("cancer_model_v2.h5", compile=False)
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Model loading failed:", e)
    model = None


# ✅ HOME
@app.route('/')
def home():
    return render_template('index.html')


# ✅ PREDICT
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "❌ Model not loaded properly"

    file = request.files.get('file')

    if file is None or file.filename == "":
        return render_template(
            'index.html',
            warning="⚠️ Please upload an image"
        )

    # read image
    img_array = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return render_template(
            'index.html',
            warning="⚠️ Invalid image file"
        )

    # preprocess
    img = cv2.resize(img, (64, 64))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    # prediction
    try:
        pred = model.predict(img)[0][0]
    except Exception as e:
        return f"❌ Prediction error: {str(e)}"

    if pred > 0.5:
        result = "🛑 Cancer Detected"
        prob = round(pred * 100, 2)
        color = "red"
    else:
        result = "✅ No Cancer"
        prob = round((1 - pred) * 100, 2)
        color = "lime"

    return render_template(
        'index.html',
        prediction=result,
        prob=prob,
        color=color
    )


# ✅ RUN APP
if __name__ == "__main__":
    print("🚀 Starting Flask app...")
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)