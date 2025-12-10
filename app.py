import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import tempfile
import os

# ----------------------------------
# Page Configuration
# ----------------------------------
st.set_page_config(
    page_title="Plant Disease Recognition",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------
# Custom CSS for Better UI/UX
# ----------------------------------
st.markdown("""
<style>
    /* Main background gradient */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2d5016 0%, #1a3409 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Header styling */
    h1 {
        color: #2d5016;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 700;
        text-align: center;
        padding: 20px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    h2, h3 {
        color: #3d6b1f;
    }
    
    /* Card-like containers */
    .css-1r6slb0 {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(90deg, #56ab2f 0%, #a8e063 100%);
        color: white;
        border: none;
        padding: 15px 40px;
        font-size: 18px;
        font-weight: bold;
        border-radius: 50px;
        box-shadow: 0 4px 15px rgba(86, 171, 47, 0.4);
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(86, 171, 47, 0.6);
    }
    
    /* File uploader styling */
    [data-testid="stFileUploader"] {
        background: white;
        padding: 30px;
        border-radius: 15px;
        border: 2px dashed #56ab2f;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 10px;
        border-left: 5px solid #56ab2f;
    }
    
    /* Feature cards */
    .feature-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        margin: 10px 0;
        transition: transform 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }
    
    .feature-icon {
        font-size: 48px;
        margin-bottom: 15px;
    }
    
    .feature-title {
        color: #2d5016;
        font-size: 22px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    
    .feature-desc {
        color: #666;
        font-size: 16px;
        line-height: 1.6;
    }
    
    /* Result card */
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 30px;
        border-radius: 20px;
        margin: 20px 0;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    /* Stats container */
    .stats-container {
        display: flex;
        justify-content: space-around;
        margin: 30px 0;
    }
    
    .stat-box {
        background: white;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        flex: 1;
        margin: 0 10px;
    }
    
    .stat-number {
        font-size: 48px;
        font-weight: bold;
        color: #56ab2f;
    }
    
    .stat-label {
        color: #666;
        font-size: 16px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------
# Load model once with error handling
# ----------------------------------
@st.cache_resource
def load_model():
    model_path = "trained_plant_disease_model.h5"
    
    try:
        if not os.path.exists(model_path):
            st.info("📥 Model file not found locally. Attempting to download...")
            
            google_drive_file_id = "YOUR_FILE_ID_HERE"
            
            if google_drive_file_id == "YOUR_FILE_ID_HERE":
                raise FileNotFoundError(
                    f"⚠️ Model file '{model_path}' not found!\n\n"
                    f"**How to fix this:**\n\n"
                    f"**Option A - Upload to GitHub:**\n"
                    f"1. Place 'trained_plant_disease_model.h5' in your repo root\n"
                    f"2. For files >100MB, install Git LFS\n\n"
                    f"**Option B - Host on Google Drive:**\n"
                    f"1. Upload model to Google Drive\n"
                    f"2. Right-click -> Share -> Anyone with link can view\n"
                    f"3. Copy the FILE_ID from the URL\n"
                    f"4. Update line 16 in app.py with your FILE_ID\n\n"
                    f"**Option C - Use Hugging Face:**\n"
                    f"Upload to Hugging Face Hub and download via their API"
                )
            
            import gdown
            url = f"https://drive.google.com/uc?id={google_drive_file_id}"
            
            with st.spinner("Downloading model... This may take a few minutes."):
                gdown.download(url, model_path, quiet=False)
            
            st.success("✅ Model downloaded successfully!")
        
        model = tf.keras.models.load_model(model_path, compile=False)
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
    
    image_file.seek(0)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(image_file.read())
        temp_path = tmp.name

    try:
        image = tf.keras.preprocessing.image.load_img(temp_path, target_size=(128, 128))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = input_arr / 255.0
        input_arr = np.expand_dims(input_arr, axis=0)

        predictions = model.predict(input_arr)
        return np.argmax(predictions), np.max(predictions)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ----------------------------------
# Sidebar
# ----------------------------------
st.sidebar.markdown("# 🌿 Navigation")
st.sidebar.markdown("---")
app_mode = st.sidebar.selectbox("Select Page", ["🏠 Home", "ℹ️ About", "🔍 Disease Recognition"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Quick Stats")
st.sidebar.metric("Supported Diseases", "38")
st.sidebar.metric("Plant Types", "14")
st.sidebar.metric("Accuracy", "95%+")

st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Tips")
st.sidebar.info("""
- Use clear, well-lit images
- Focus on the affected leaf area
- Avoid blurry photos
- Upload images in JPG/PNG format
""")


# ----------------------------------
# Home Page
# ----------------------------------
if app_mode == "🏠 Home":
    # Hero Section
    st.markdown("<h1>🌿 Plant Disease Recognition System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 20px; color: #666; margin-bottom: 40px;'>AI-Powered Plant Health Diagnosis in Seconds</p>", unsafe_allow_html=True)
    
    # Display home image if exists
    if os.path.exists("home.png"):
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.image("home.png", use_column_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Feature Cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📸</div>
            <div class="feature-title">Easy Upload</div>
            <div class="feature-desc">Simply upload a photo of your plant leaf and get instant results</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🤖</div>
            <div class="feature-title">AI-Powered</div>
            <div class="feature-desc">Advanced deep learning model trained on thousands of plant images</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <div class="feature-title">Instant Results</div>
            <div class="feature-desc">Get accurate disease diagnosis in seconds, not days</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Statistics Section
    st.markdown("### 📊 System Capabilities")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-number">38</div>
            <div class="stat-label">Disease Types</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-number">14</div>
            <div class="stat-label">Plant Species</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-number">95%</div>
            <div class="stat-label">Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-number">⚡</div>
            <div class="stat-label">Real-time</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Getting Started
    st.markdown("### 🚀 Getting Started")
    st.info("""
    1. Navigate to the **Disease Recognition** page using the sidebar
    2. Upload a clear image of your plant leaf
    3. Click the **Predict** button
    4. View your instant diagnosis and recommendations
    """)


# ----------------------------------
# About Page
# ----------------------------------
elif app_mode == "ℹ️ About":
    st.markdown("<h1>ℹ️ About This System</h1>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🔬 Technology")
        st.write("""
        This system leverages state-of-the-art deep learning technology to identify plant diseases 
        from leaf images. Our convolutional neural network has been trained on thousands of images 
        to provide accurate, real-time diagnoses.
        """)
        
        st.markdown("### 🌱 Supported Plants")
        plants_col1, plants_col2 = st.columns(2)
        
        with plants_col1:
            st.markdown("""
            - 🍎 Apple
            - 🫐 Blueberry
            - 🍒 Cherry
            - 🌽 Corn (Maize)
            - 🍇 Grape
            - 🍊 Orange
            - 🍑 Peach
            """)
        
        with plants_col2:
            st.markdown("""
            - 🌶️ Pepper (Bell)
            - 🥔 Potato
            - 🍓 Raspberry
            - 🫘 Soybean
            - 🎃 Squash
            - 🍓 Strawberry
            - 🍅 Tomato
            """)
    
    with col2:
        st.markdown("### 📈 Model Stats")
        st.metric("Training Images", "54,000+")
        st.metric("Model Accuracy", "95.2%")
        st.metric("Processing Time", "< 2 sec")
        st.metric("Classes", "38")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("### 📖 How to Use")
    
    step_col1, step_col2, step_col3 = st.columns(3)
    
    with step_col1:
        st.info("""
        **Step 1: Navigate**
        
        Go to the Disease Recognition page from the sidebar menu
        """)
    
    with step_col2:
        st.info("""
        **Step 2: Upload**
        
        Choose a clear image of the plant leaf you want to analyze
        """)
    
    with step_col3:
        st.info("""
        **Step 3: Predict**
        
        Click the Predict button and get instant results
        """)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.success("""
    💡 **Pro Tip:** For best results, ensure your images are well-lit, in focus, and clearly show 
    any symptoms or abnormalities on the leaf surface.
    """)


# ----------------------------------
# Disease Recognition Page
# ----------------------------------
elif app_mode == "🔍 Disease Recognition":
    st.markdown("<h1>🔍 Disease Recognition</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666; font-size: 18px; margin-bottom: 30px;'>Upload a plant leaf image for instant AI-powered diagnosis</p>", unsafe_allow_html=True)
    
    if model is None:
        st.error("⚠️ Model could not be loaded. Please ensure 'trained_plant_disease_model.h5' is in the correct directory.")
        st.info(f"📁 Current working directory: {os.getcwd()}")
        st.stop()

    # Upload Section
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 📤 Upload Image")
        test_image = st.file_uploader("", type=["jpg", "jpeg", "png"], help="Upload a clear image of the plant leaf")
    
    if test_image is not None:
        # Display uploaded image
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🖼️ Uploaded Image")
            img = Image.open(test_image)
            st.image(img, use_column_width=True, caption="Your uploaded image")
        
        with col2:
            st.markdown("### 🎯 Ready for Analysis")
            st.info("""
            ✅ Image uploaded successfully!
            
            Click the **Predict** button below to analyze the plant and detect any diseases.
            """)
            
            # Predict Button
            if st.button("🔬 Predict Disease", use_container_width=True):
                with st.spinner("🤖 AI is analyzing your plant..."):
                    try:
                        result_index, confidence = model_prediction(test_image)
                        
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
                        
                        predicted_class = class_name[result_index]
                        plant_type = predicted_class.split('___')[0].replace('_', ' ')
                        disease = predicted_class.split('___')[1].replace('_', ' ')
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                        # Results Section
                        if disease.lower() == 'healthy':
                            st.balloons()
                            st.success("### ✅ Great News! Your Plant is Healthy")
                            st.markdown(f"""
                            <div style='background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                                        padding: 30px; border-radius: 20px; color: white; margin: 20px 0;'>
                                <h2 style='color: white; margin: 0;'>🌿 {plant_type}</h2>
                                <h3 style='color: white; margin: 10px 0;'>Status: {disease}</h3>
                                <p style='font-size: 18px; margin: 10px 0;'>Confidence: {confidence*100:.2f}%</p>
                                <p style='margin-top: 20px;'>Your plant looks healthy! Continue your current care routine.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.snow()
                            st.warning("### ⚠️ Disease Detected")
                            st.markdown(f"""
                            <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                                        padding: 30px; border-radius: 20px; color: white; margin: 20px 0;'>
                                <h2 style='color: white; margin: 0;'>🌿 {plant_type}</h2>
                                <h3 style='color: white; margin: 10px 0;'>Disease: {disease}</h3>
                                <p style='font-size: 18px; margin: 10px 0;'>Confidence: {confidence*100:.2f}%</p>
                                <p style='margin-top: 20px;'>⚠️ Disease detected. Consider consulting an agricultural expert for treatment options.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Additional Info
                        st.markdown("### 📋 Recommendation")
                        if disease.lower() == 'healthy':
                            st.info("""
                            - Continue regular watering and fertilizing schedule
                            - Monitor for any changes in leaf color or texture
                            - Ensure adequate sunlight and air circulation
                            """)
                        else:
                            st.warning("""
                            - Isolate affected plants to prevent spread
                            - Remove severely infected leaves
                            - Consider appropriate fungicide or treatment
                            - Consult with a local agricultural extension office
                            """)
                        
                    except Exception as e:
                        st.error(f"❌ Error during prediction: {str(e)}")
                        st.info("Please try uploading a different image or check if the model file is working correctly.")
    
    else:
        # Empty state
        st.markdown("""
        <div style='text-align: center; padding: 60px 20px; background: white; border-radius: 20px; margin: 40px 0;'>
            <h2 style='color: #666;'>📸 No Image Uploaded Yet</h2>
            <p style='color: #999; font-size: 18px;'>Upload a plant leaf image above to get started with the analysis</p>
        </div>
        """, unsafe_allow_html=True)
