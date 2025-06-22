import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Get absolute path to model
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, '..', 'models', 'mnist_model.h5')

@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error(f"Model path: {model_path}")
        st.error("Please ensure the model exists and path is correct")
        st.stop()

model = load_model()

st.title('MNIST Digit Classifier')
st.write('Upload an image of a handwritten digit (0-9) for classification')

uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess image
    img = image.convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28
    img_array = np.array(img)
    img_array = img_array.reshape(1, 28, 28, 1) / 255.0  # Normalize
    
    # Make prediction
    if st.button('Classify'):
        try:
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class]
            
            st.success(f'**Predicted Digit:** {predicted_class}')
            st.info(f'**Confidence:** {confidence:.2%}')
            
            # Show probabilities
            st.subheader('Prediction Probabilities:')
            for i in range(10):
                st.write(f'Digit {i}: {prediction[0][i]:.2%}')
                st.progress(float(prediction[0][i]))
        except Exception as e:
            st.error(f"Prediction error: {e}")