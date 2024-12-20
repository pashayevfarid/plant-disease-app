import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd


# Disease classes
class_name = ['Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 
              'Cherry_(including_sour)___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
              'Corn_(maize)___Common_rust_', 'Corn_(maize)___healthy', 'Corn_(maize)___Northern_Leaf_Blight', 
              'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
              'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper_bell___Bacterial_spot', 'Pepper_bell___healthy', 
              'Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight', 'Strawberry___healthy', 
              'Strawberry___Leaf_scorch', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy', 
              'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
              'Tomato___Target_Spot', 'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus']

# Load dataset for disease information
data = pd.read_csv('Plant_symptoms_treatment.csv', encoding='Windows-1252')

# Function to extract disease information from dataset
def get_disease_info(disease_name):
    result = data[data['Disease'] == disease_name]
    if not result.empty:
        return result.iloc[0]['Symptoms'], result.iloc[0]['Treatment']
    else:
        # Return "Healthy" if disease info is not found
        return "Healthy", "No treatment needed"

# Tensorflow Model Prediction
def model_prediction(test_image_path):
    model = tf.keras.models.load_model('VGG_16_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image_path, target_size=(224, 224))
    input_arr = tf.keras.preprocessing.image.img_to_array(image) / 255.0
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    return np.argmax(predictions), np.max(predictions) 

def format_disease_name(disease_name):
    # Replace underscores with spaces and capitalize each word
    formatted_name = disease_name.replace('_', ' ').title()
    return formatted_name



# Streamlit layout design
st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿", layout="centered")

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Main Page
if app_mode == "Home":
    st.header("🌿 Plant Disease Detection Tool 🌿")
    image_url = "https://media.licdn.com/dms/image/v2/D4D22AQHDxqMMS4KVYA/feedshare-shrink_800/feedshare-shrink_800/0/1682330831854?e=2147483647&v=beta&t=1cFDdyhOY2CoFKXM-H1De2qeraQUv3zPbYFhArXEkQA"
    st.image(image_url, use_container_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 
    
    Our goal is to assist in the early identification of plant diseases, enabling you to take swift action to protect your crops and ensure a healthier harvest.
                
    ### How It Works
    1. **Upload Image:** Navigate to the **Disease Recognition** page and either upload an image  of a plant or capture one directly using your camera (coming soon).
    2. **Analysis:** Our advanced deep learning model processes the image to detect signs of diseases based on trained algorithms.
    3. **Results:** Once the analysis is complete, you’ll see the predicted disease along with its symptoms and recommended treatment.
                
    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and begin identifying plant diseases efficiently. 
    """)

# About Project
elif app_mode == "About":
    st.header("About")
    st.markdown("""
                #### About Model
                Model is based on VGG-16 architecture.
                #### About Dataset
                This dataset consists of about 77K rgb images of healthy and diseased crop leaves which is categorized into 33 different classes.
                The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                #### Content
                1. train (60930 images)
                2. test (33 images)
                3. validation (17572 images)
                """)

# Prediction Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    
    # Initialize uploaded_image as None to avoid the NameError
    uploaded_image = None

    # Upload Image
    uploaded_image = st.file_uploader("Upload an image of the plant leaf (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])

     
    # Show the uploaded image
    if uploaded_image is not None:
        st.image(uploaded_image, use_container_width=True)
        test_image_path = "uploaded_image.jpg"
        with open(test_image_path, "wb") as f:
            f.write(uploaded_image.getbuffer())
    else:
        st.warning("Please upload an image to proceed.")

    # Prediction Button
    if uploaded_image and st.button("Predict"):
        st.write("Analyzing the image...")
        
        # Use the path for the uploaded
        test_image_path = "uploaded_image.jpg"
        
        result_index, predicted_probability = model_prediction(test_image_path)
        
        # Disease name
        disease_name = class_name[result_index]

        # Format the disease name
        formatted_disease_name = format_disease_name(disease_name)
        
        # Fetch disease information
        symptoms, treatment = get_disease_info(disease_name)
        
        # Display the result
        st.success(f"Model Prediction: {formatted_disease_name}")
        st.write(f"**Confidence**: {predicted_probability * 100:.2f}%")
        st.write(f"**Symptoms:** {symptoms}")
        st.write(f"**Treatment:** {treatment}")





