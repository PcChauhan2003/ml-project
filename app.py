from flask import Flask, render_template, request
import cv2
import numpy as np
import os
import tensorflow as tf

# Initialize Flask app
app = Flask(__name__)

# ✅ CONFIGURATION
MODEL_PATH = "cancer_model_v2.h5"
IMG_SIZE = 64  # Ensure this matches your model's training size

# ✅ LOAD MODEL
# TensorFlow 2.16+ uses Keras 3 by default, which understands 'batch_shape'
try:
    print(f"🔄 Loading model from {MODEL_PATH}...")
    # compile=False avoids errors if you used custom loss functions or optimizers
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ CRITICAL ERROR: Could not load model: {e}")
    model = None

# ✅ HOME ROUTE
@app.route('/')
def home():
    return render_template('index.html')

# ✅ PREDICTION ROUTE
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "❌ Error: Model not initialized on server."

    file = request.files.get('file')
    if not file or file.filename == "":
        return render_template('index.html', warning="⚠️ Please upload an image")

    try:
        # Read and decode image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            return render_template('index.html', warning="⚠️ Invalid image format")

        # Preprocessing
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_normalized = img_resized.astype('float32') / 255.0
        img_final = np.expand_dims(img_normalized, axis=0)

        # Run Prediction
        prediction_output = model.predict(img_final)
        
        # Determine probability based on output shape
        # (Works for both Sigmoid [0,1] and Softmax [prob1, prob2])
        if prediction_output.shape[-1] == 1:
            raw_prob = float(prediction_output[0][0])
        else:
            raw_prob = float(np.max(prediction_output[0]))

        # Logic for results
        if raw_prob > 0.5:
            result_text = "🛑 Cancer Detected"
            display_prob = round(raw_prob * 100, 2)
            result_color = "red"
        else:
            result_text = "✅ No Cancer"
            display_prob = round((1 - raw_prob) * 100, 2)
            result_color = "lime"

        return render_template(
            'index.html',
            prediction=result_text,
            prob=display_prob,
            color=result_color
        )

    except Exception as e:
        print(f"❌ Prediction Error: {e}")
        return f"❌ Internal Prediction Error: {str(e)}"

# ✅ START SERVER
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)