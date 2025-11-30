import requests

# --- CONFIGURATION ---
API_URL = "http://127.0.0.1:5000/predict"
# Replace this with the path to one of your test images
TEST_IMAGE_PATH = r"C:\Projects\Pneumonia detection\Data\test\Pneumonia\test_image_1.jpeg" 

# --- SEND REQUEST ---
print(f"Sending image {TEST_IMAGE_PATH} to API...")

try:
    with open(TEST_IMAGE_PATH, 'rb') as f:
        files = {'file': (TEST_IMAGE_PATH, f, 'image/jpeg')}
        
        response = requests.post(API_URL, files=files)
        
        # Check if the request was successful
        if response.status_code == 200:
            print("\nAPI Response:")
            print(response.json())
        else:
            print(f"\nAPI Error: Status Code {response.status_code}")
            print(response.text)

except FileNotFoundError:
    print(f"\nERROR: Test image not found at {TEST_IMAGE_PATH}. Please update the path.")
except requests.exceptions.ConnectionError:
    print("\nERROR: Could not connect to the API. Is 'python api.py' running?")
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")