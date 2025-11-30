import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
# --- 1. CONFIGURATION ---
MODEL_PATH = 'pneumonia_detection_model.h5'
IMAGE_SIZE = (224, 224)
# --- CHANGE THIS PATH TO YOUR NEW IMAGE ---
NEW_IMAGE_PATH = r'C:\Projects\Pneumonia detection\Data\test\NORMAL\IM-0003-0001.jpeg' 

# --- 2. LOAD MODEL ---
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Successfully loaded model from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please make sure you have run train_model.py successfully.")
    exit()

# --- 3. PREDICTION FUNCTION ---
def predict_pneumonia(img_path):
    # Load the image and resize it to the expected size
    img = image.load_img(img_path, target_size=IMAGE_SIZE)
    # Convert image to numpy array
    img_array = image.img_to_array(img)
    # Expand dimensions to match the model's expected shape: (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    # Normalize the image (as done in training)
    img_array /= 255.0
    
    # Make the prediction
    prediction = model.predict(img_array)[0][0]
    
    # Determine the class and confidence
    if prediction > 0.5:
        # Prediction > 0.5 means class 1 (Pneumonia) based on flow_from_directory sorting
        class_label = "PNEUMONIA"
        confidence = prediction * 100
    else:
        # Prediction <= 0.5 means class 0 (Normal)
        class_label = "NORMAL"
        confidence = (1 - prediction) * 100

    print("\n--- Prediction Result ---")
    print(f"Image: {os.path.basename(img_path)}")
    print(f"Predicted Class: **{class_label}**")
    print(f"Confidence: {confidence:.2f}%")
    print("-------------------------")
    
# --- 4. EXECUTE PREDICTION ---
predict_pneumonia(NEW_IMAGE_PATH)