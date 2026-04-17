from flask import Flask, render_template, request
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model
model = load_model("cancer_model.h5")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')

    # ❌ No file
    if file is None or file.filename == "":
        return render_template('index.html',
                               warning="⚠️ Please upload an image")

    # Read image
    img_array = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # ❌ Invalid image
    if img is None:
        return render_template('index.html',
                               warning="⚠️ Invalid image file")

    # ==========================
    # 🔥 SMART VALIDATION (WARNING ONLY)
    # ==========================
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    variance = np.var(gray)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.mean(edges)

    warning_msg = None

    if variance < 100 or edge_density > 120:
        warning_msg = "⚠️ This may not be a medical image. Prediction may be inaccurate."

    # ==========================
    # ✅ PREPROCESS
    # ==========================
    img = cv2.resize(img, (64, 64))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    # ==========================
    # 🔮 PREDICT
    # ==========================
    pred = model.predict(img)[0][0]

    # ==========================
    # ✅ CORRECT PROBABILITY
    # ==========================
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
        color=color,
        warning=warning_msg
    )


# 🌐 Run for mobile + network
if __name__ == "__main__":
    print("🚀 Starting Flask server...")
    app.run(host="0.0.0.0", port=8000)