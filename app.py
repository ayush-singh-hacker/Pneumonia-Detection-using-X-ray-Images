import streamlit as st
from PIL import Image
from io import BytesIO
import requests
import os

# --- 1. CONFIGURATION ---

# !!! IMPORTANT: UPDATE THESE TWO VARIABLES !!!
WEBSITE_NAME = "Pneumonia Detection using X-ray Images" 
# Replace with the public URL of your deployed Flask API (e.g., on Render)
API_URL = "https://chest-xray-diag.onrender.com/predict" 
# If running locally, keep it as 'http://127.0.0.1:5000/predict' 
# If deployed, it must be the public link: "https://your-public-api-name.render.com/predict"

# --- 2. PREDICTION CORE FUNCTION (Uses API) ---

def predict_pneumonia_via_api(uploaded_file):
    """
    Sends the image file to the Flask API endpoint for prediction.
    """
    st.info(f"Sending request to API at: {API_URL}")
    
    # Prepare the file for sending as multipart/form-data
    files = {
        'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
    }

    try:
        # Send the POST request
        response = requests.post(API_URL, files=files)
        
        # Check for successful response status code
        if response.status_code == 200:
            return response.json()
        else:
            # Handle API errors 
            error_data = response.json() if response.content else {"error": "Unknown API error."}
            st.error(f"API Error ({response.status_code}): {error_data.get('error', 'Check server logs.')}")
            return None
            
    except requests.exceptions.ConnectionError:
        st.error(
            "🔴 **Connection Error:** Could not connect to the Flask API. "
            "Please ensure **`python api.py`** is running in a separate terminal window."
        )
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during API communication: {e}")
        return None

# --- 3. STREAMLIT UI SETUP (Branding Applied Here) ---

# This MUST be the first Streamlit command. It sets the browser tab title.
st.set_page_config(
    page_title=f"{WEBSITE_NAME}",
    page_icon="⚕️",
    layout="centered"
)

# Main application title
st.title(f"⚕️ {WEBSITE_NAME}")
st.markdown("Upload a chest X-ray image to get an instant diagnosis via the **Flask API** service.")

# File Uploader Widget
uploaded_file = st.file_uploader(
    "Choose a chest X-ray image (.jpg, .jpeg, .png)", 
    type=['jpg', 'jpeg', 'png']
)

if uploaded_file is not None:
    # Display Uploaded Image (using the corrected parameter)
    # Corrected: use_column_width=True -> use_container_width=True
    st.image(uploaded_file, caption='Uploaded X-ray Image', use_container_width=True)
    st.write("")
    
    # Make Prediction
    with st.spinner('Waiting for prediction from Flask API...'):
        result = predict_pneumonia_via_api(uploaded_file)
        
    # --- Display Results ---
    if result:
        label = result['prediction']
        confidence = result['confidence_percent']
        
        st.subheader("✅ Prediction Received:")
        
        # Use colored metrics for visual distinction
        if label == "PNEUMONIA":
            st.error(f"**Result:** {label} (High Risk)")
        else:
            st.success(f"**Result:** {label}")
            
        st.markdown(f"**Confidence:** `{confidence:.2f}%`")
        
        st.info("Disclaimer: This is an AI-assisted diagnostic tool and should not replace professional medical advice.")

# --- Instructions for Sharing ---
st.markdown(
    f"""
    ---
    ### 🔗 Shareable Link & Deployment
    To share this tool with others, you must deploy both the API and the Streamlit app.
    
    1. **Deploy API (`api.py`):** Get a public URL (e.g., `https://my-pneumonia-api.render.com`). Update the `API_URL` variable above.
    2. **Deploy Client (`app.py`):** Deploy this script (e.g., via Streamlit Community Cloud) to get your final website link.
    """

)
