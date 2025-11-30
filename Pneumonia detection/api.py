from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
import base64

# --- 1. CONFIGURATION AND MODEL LOADING ---
MODEL_PATH = 'pneumonia_detection_model.h5'
IMAGE_SIZE = (224, 224)

# Initialize Flask app
app = Flask(__name__)

# Load the model globally when the app starts
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"ERROR: Could not load model: {e}")
    # Exit or handle gracefully if model is critical
    model = None

# --- 2. PREDICTION CORE FUNCTION ---

def preprocess_and_predict(img):
    """Processes PIL image, runs prediction, and returns results."""
    
    if img.mode != 'RGB':
        img = img.convert('RGB')
        
    # Resize the PIL Image object
    img_resized = img.resize(IMAGE_SIZE)
    
    # Convert image to numpy array
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    
    # Expand dimensions to match the model's expected shape: (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize the image (as done in training)
    img_array /= 255.0
    
    # Make the prediction
    prediction = model.predict(img_array)[0][0]
    
    # Determine the class and confidence
    if prediction > 0.5:
        class_label = "PNEUMONIA"
        confidence = float(prediction * 100)
    else:
        class_label = "NORMAL"
        confidence = float((1 - prediction) * 100)
    
    return class_label, confidence

# --- 3. FLASK API ENDPOINT ---

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint to receive image data and return prediction."""
    
    if model is None:
        return jsonify({"error": "Model not loaded. Check server logs."}), 500

    # Check for image file upload
    if 'file' in request.files:
        try:
            image_file = request.files['file']
            img = Image.open(image_file.stream)
        except Exception as e:
            return jsonify({"error": "Error processing image file.", "details": str(e)}), 400
            
    # Check for base64 JSON payload (for client-side base64 sending)
    elif request.is_json and 'image_base64' in request.json:
        try:
            image_data = request.json['image_base64']
            # Decode base64 string
            img_bytes = base64.b64decode(image_data)
            img = Image.open(BytesIO(img_bytes))
        except Exception as e:
            return jsonify({"error": "Error processing base64 image data.", "details": str(e)}), 400
    
    else:
        return jsonify({"error": "No image file ('file') or base64 data ('image_base64') provided."}), 400
        
    
    # Run Prediction
    try:
        label, confidence = preprocess_and_predict(img)
        
        return jsonify({
            "status": "success",
            "prediction": label,
            "confidence_percent": round(confidence, 2)
        })
    except Exception as e:
        return jsonify({"error": "Prediction failed.", "details": str(e)}), 500

# --- 4. RUN SERVER ---
if __name__ == '__main__':
    # Use 0.0.0.0 to make it accessible externally (if needed)
    # Use host='127.0.0.1' for local access only
    app.run(host='127.0.0.1', port=5000, debug=False)