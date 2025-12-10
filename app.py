import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import tempfile

# ----------------------------------
# Load model once (IMPORTANT FIX)
# ----------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("trained_plant_disease_model.h5", compile=False)
    return model

model = load_model()

# ----------------------------------
# Prediction Function
# ----------------------------------
def model_prediction(image_file):
    # Save uploaded file to a temporary path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(image_file.read())       # write bytes
        temp_path = tmp.name               # get path

    # Preprocess image
    image = tf.keras.preprocessing.image.load_img(temp_path, target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = input_arr / 255.0          # normalize
    input_arr = np.expand_dims(input_arr, axis=0)

    # Predict
    predictions = model.predict(input_arr)
    return np.argmax(predictions)


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
    st.image("home.png", use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    """)


# ----------------------------------
# About Page
# ----------------------------------
elif app_mode == "About":
    st.header("About")
    st.markdown("""
    #### About Dataset
    Dataset details...
    """)


# ----------------------------------
# Disease Recognition Page
# ----------------------------------
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")

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

                st.success(f"The model predicts: **{class_name[result_index]}**")

            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
