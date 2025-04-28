# Import necessary libraries
import streamlit as st  # Web application framework
import torch  # PyTorch deep learning framework
import numpy as np  # Numerical computing
from PIL import Image  # Image processing
import torchvision.transforms as transforms  # Image transformations
from simp_transform_setup import MNISTTransformer  # Our custom transformer model
import cv2  # Computer vision library for image processing
from streamlit_drawable_canvas import st_canvas  # Drawing canvas component

# Configure the Streamlit page
# This sets up the page title, icon, and layout
st.set_page_config(
    page_title="MNIST Digit Classifier",  # Title shown in browser tab
    page_icon="✍️",  # Emoji icon
    layout="centered"  # Center the content
)

# Add title and description to the web interface
st.title("MNIST Digit Classifier")  # Main heading
st.write("Draw a digit (0-9) in the box below and see the model's prediction!")  # Instructions

# Define a function to load the trained model
# @st.cache_resource ensures the model is loaded only once and cached
@st.cache_resource
def load_model():
    # Initialize the model with the same architecture as during training
    model = MNISTTransformer(
        img_size=28,
        patch_size=7,
        emb_dim=64,
        num_heads=4,
        num_layers=2,
        num_classes=10
    )
    # Load the trained weights from the saved file
    # Using weights_only=True for security
    model.load_state_dict(torch.load('mnist_transformer.pth', 
                                   map_location=torch.device('cpu'),
                                   weights_only=True))
    model.eval()  # Set model to evaluation mode
    return model

# Try to load the model, show error if model file not found
try:
    model = load_model()
except:
    st.error("Model file not found. Please train the model first using simp_train.py")
    st.stop()  # Stop the application if model can't be loaded

# Create a drawing canvas
st.write("Draw a digit in the box below:")
canvas_result = st_canvas(
    fill_color="black",  # Fill color
    stroke_width=20,  # Stroke width
    stroke_color="white",  # Stroke color
    background_color="black",  # Background color
    height=280,  # Canvas height
    width=280,  # Canvas width
    drawing_mode="freedraw",  # Drawing mode
    key="canvas",  # Unique key
    update_streamlit=True,  # Update on drawing
)

# Process the drawn image when the canvas has content
if canvas_result.image_data is not None:
    # Convert the canvas data to a numpy array
    img_array = np.array(canvas_result.image_data)
    
    # Convert the RGBA image to grayscale
    # This matches the MNIST format which uses single-channel images
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY)
    
    # Resize the image to 28x28 pixels
    # This matches the MNIST image size
    img_resized = cv2.resize(img_gray, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Invert the colors
    # MNIST has white digits on black background, so we need to invert
    img_inverted = 255 - img_resized
    
    # Normalize the pixel values to [0, 1] and ensure float32
    img_normalized = (img_inverted / 255.0).astype(np.float32)
    
    # Define the same transformations used during training
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to PyTorch tensor
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize using MNIST statistics
    ])
    
    # Apply transformations and add batch dimension
    img_tensor = transform(img_normalized).unsqueeze(0)
    
    # Ensure the tensor is float32
    img_tensor = img_tensor.float()
    
    # Make prediction using the model
    with torch.no_grad():  # Disable gradient computation for inference
        output = model(img_tensor)  # Get model output
        probabilities = torch.nn.functional.softmax(output, dim=1)  # Convert to probabilities
        predicted_class = torch.argmax(output, dim=1).item()  # Get predicted class
        confidence = probabilities[0][predicted_class].item()  # Get confidence score
    
    # Display prediction results
    st.subheader("Prediction Results")
    
    # Create two columns for displaying results side by side
    col1, col2 = st.columns(2)
    
    # Display predicted digit in the first column
    with col1:
        st.write("Predicted Digit:")
        st.markdown(f"<h1 style='text-align: center; color: #FF4B4B;'>{predicted_class}</h1>", unsafe_allow_html=True)
    
    # Display confidence score in the second column
    with col2:
        st.write("Confidence:")
        st.markdown(f"<h1 style='text-align: center; color: #FF4B4B;'>{confidence:.2%}</h1>", unsafe_allow_html=True)
    
    # Display probability distribution across all classes
    st.subheader("Probability Distribution")
    probs = probabilities[0].numpy()  # Convert to numpy array
    st.bar_chart(probs)  # Create bar chart of probabilities
    
    # Display the processed image that was fed to the model
    st.subheader("Processed Image")
    st.image(img_normalized, width=100)  # Show the normalized image

# Add a clear button to reset the canvas
if st.button("Clear Canvas"):
    st.experimental_rerun()  # Refresh the page to clear the canvas
