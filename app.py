import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import tempfile
import os

# ----------------------------------
# Load model once with error handling
# ----------------------------------
@st.cache_resource
def load_model():
    model_path = "trained_plant_disease_model.h5"
    
    try:
        # Check if model exists locally
        if not os.path.exists(model_path):
            st.info("📥 Model file not found locally. Attempting to download...")
            
            # Google Drive direct download link (replace FILE_ID with your actual file ID)
            # To get FILE_ID: Share your file -> Get link -> Extract ID from URL
            # URL format: https://drive.google.com/file/d/FILE_ID/view?usp=sharing
            google_drive_file_id = "YOUR_FILE_ID_HERE"
            
            if google_drive_file_id == "YOUR_FILE_ID_HERE":
                raise FileNotFoundError(
                    f"⚠️ Model file '{model_path}' not found!\n\n"
                    f"**How to fix this:**\n\n"
                    f"**Option A - Upload to GitHub:**\n"
                    f"1. Place 'trained_plant_disease_model.h5' in your repo root\n"
                    f"2. For files >100MB, install Git LFS:\n"
                    f"   ```\n"
                    f"   git lfs install\n"
                    f"   git lfs track '*.h5'\n"
                    f"   git add .gitattributes trained_plant_disease_model.h5\n"
                    f"   git commit -m 'Add model file'\n"
                    f"   git push\n"
                    f"   ```\n\n"
                    f"**Option B - Host on Google Drive:**\n"
                    f"1. Upload model to Google Drive\n"
                    f"2. Right-click -> Share -> Anyone with link can view\n"
                    f"3. Copy the FILE_ID from the URL\n"
                    f"4. Update line 16 in app.py with your FILE_ID\n\n"
                    f"**Option C - Use Hugging Face:**\n"
                    f"Upload to Hugging Face Hub and download via their API"
                )
            
            # Download from Google Drive
            import gdown
            url = f"https://drive.google.com/uc?id={google_drive_file_id}"
            
            with st.spinner("Downloading model... This may take a few minutes."):
                gdown.download(url, model_path, quiet=False)
            
            st.success("✅ Model downloaded successfully!")
        
        # Load the model
        model = tf.keras.models.load_model(model_path, compile=False)
        st.success(f"✅ Model loaded successfully!")
        return model
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

model = load_model()

# ----------------------------------
# Prediction Function
# ----------------------------------
def model_prediction(image_file):
    if model is None:
        raise ValueError("Model is not loaded. Please check the model file path.")
    
    # Reset file pointer to beginning
    image_file.seek(0)
    
    # Save uploaded file to a temporary path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(image_file.read())
        temp_path = tmp.name

    try:
        # Preprocess image
        image = tf.keras.preprocessing.image.load_img(temp_path, target_size=(128, 128))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = input_arr / 255.0  # normalize
        input_arr = np.expand_dims(input_arr, axis=0)

        # Predict
        predictions = model.predict(input_arr)
        return np.argmax(predictions)
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ----------------------------------
# Sidebar
# ----------------------------------
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])


# ----------------------------------
# Home Page
# ----------------------------------
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    
    # Check if home image exists
    if os.path.exists("home.png"):
        st.image("home.png", use_column_width=True)
    else:
        st.info("Home image not found. Please add 'home.png' to your project directory.")
    
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    ### Features:
    - Upload plant leaf images
    - Get instant disease diagnosis
    - Supports 38 different plant disease classes
    
    Use the sidebar to navigate between pages.
    """)


# ----------------------------------
# About Page
# ----------------------------------
elif app_mode == "About":
    st.header("About")
    st.markdown("""
    #### About Dataset
    This system uses a trained deep learning model to identify plant diseases from leaf images.
    
    #### Supported Plants:
    - Apple
    - Blueberry
    - Cherry
    - Corn (Maize)
    - Grape
    - Orange
    - Peach
    - Pepper (Bell)
    - Potato
    - Raspberry
    - Soybean
    - Squash
    - Strawberry
    - Tomato
    
    #### How to Use:
    1. Navigate to the "Disease Recognition" page
    2. Upload a clear image of a plant leaf
    3. Click "Predict" to get the diagnosis
    """)


# ----------------------------------
# Disease Recognition Page
# ----------------------------------
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    
    # Check if model is loaded
    if model is None:
        st.error("⚠️ Model could not be loaded. Please ensure 'trained_plant_disease_model.h5' is in the correct directory.")
        st.info(f"Current working directory: {os.getcwd()}")
        st.info(f"Files in current directory: {os.listdir('.')}")
        st.stop()

    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])

    if test_image is not None:
        # Display uploaded image
        img = Image.open(test_image)
        st.image(img, use_column_width=True)

        if st.button("Predict"):
            st.snow()
            st.write("Model is predicting...")

            try:
                result_index = model_prediction(test_image)

                class_name = [
                    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                    'Tomato___healthy'
                ]

                # Format the result nicely
                predicted_class = class_name[result_index]
                plant_type = predicted_class.split('___')[0]
                disease = predicted_class.split('___')[1]
                
                st.success(f"### Prediction Result")
                st.write(f"**Plant Type:** {plant_type}")
                st.write(f"**Condition:** {disease}")
                
                if disease.lower() == 'healthy':
                    st.balloons()
                    st.info("✅ Great news! The plant appears to be healthy.")
                else:
                    st.warning(f"⚠️ Disease detected: {disease}")

            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                st.info("Please try uploading a different image or check if the model file is corrupted.")
