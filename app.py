import streamlit as st
import torch
from PIL import Image
import numpy as np
import config
from model import create_model
from data import get_val_test_transforms
from utils import load_model

# Page configuration
st.set_page_config(
    page_title=config.STREAMLIT_TITLE,
    page_icon="🌿",
    layout="wide"
)

@st.cache_resource
def load_trained_model(model_type: str = 'resnet'):
    try:
        # Create model
        model = create_model(model_type, device=config.DEVICE)
        
        # Load best checkpoint
        checkpoint_path = config.CHECKPOINT_DIR / f"{model_type}_model_best.pth"
        
        if checkpoint_path.exists():
            model, _, _ = load_model(model, checkpoint_path, config.DEVICE)
            model.eval()
            return model, True
        else:
            st.warning(f"Checkpoint not found at {checkpoint_path}. Using untrained model for demo.")
            model.eval()
            return model, False
    
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, False


def predict_image(model, image: Image.Image):
    # Get transforms
    transform = get_val_test_transforms()
    
    # Preprocess image
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    image_tensor = image_tensor.to(config.DEVICE)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = probabilities.max(1)
    
    # Get class name
    predicted_class = config.CLASS_NAMES[predicted_idx.item()]
    confidence_score = confidence.item()
    all_probs = probabilities.cpu().numpy()[0]
    
    return predicted_class, confidence_score, all_probs


def display_prediction_results(predicted_class: str, confidence: float, all_probs: np.ndarray):
    st.markdown("### Prediction Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Color code based on disease
        if predicted_class == "Healthy":
            color = "green"
        elif "Blight" in predicted_class:
            color = "red"
        else:
            color = "orange"
        
        st.markdown(
            f"<h2 style='color: {color};'>{predicted_class}</h2>",
            unsafe_allow_html=True
        )
        st.metric("Confidence", f"{confidence * 100:.2f}%")
    
    with col2:
        st.markdown("#### Class Probabilities:")
        for i, class_name in enumerate(config.CLASS_NAMES):
            prob = all_probs[i]
            st.progress(float(prob), text=f"{class_name}: {prob * 100:.2f}%")


def display_disease_info(predicted_class: str):
    disease_info = {
        "Healthy": {
            "description": "The plant appears healthy with no visible signs of disease.",
            "recommendation": "Continue regular care and monitoring.",
            "color": "green"
        },
        "Early_Blight": {
            "description": "Early blight is a common fungal disease caused by Alternaria solani. "
                         "It affects tomato and potato plants, causing dark spots with concentric rings on leaves.",
            "recommendation": "Remove infected leaves, improve air circulation, apply fungicides, "
                            "and practice crop rotation.",
            "color": "orange"
        },
        "Late_Blight": {
            "description": "Late blight is a devastating disease caused by Phytophthora infestans. "
                         "It can quickly destroy entire crops if not controlled.",
            "recommendation": "Remove and destroy infected plants immediately, apply copper-based fungicides, "
                            "ensure good drainage, and avoid overhead watering.",
            "color": "red"
        }
    }
    
    if predicted_class in disease_info:
        info = disease_info[predicted_class]
        
        # Format the class name for display (replace underscore with space)
        display_name = predicted_class.replace("_", " ")
        
        st.markdown(f"### About: {display_name}")
        st.markdown(f"**Description:** {info['description']}")
        st.markdown(f"**Recommendation:** {info['recommendation']}")

def show_sample_images():
    st.markdown("### Sample Training Images by Class")
    
    cols = st.columns(len(config.CLASS_NAMES))
    
    for idx, class_name in enumerate(config.CLASS_NAMES):
        # Ganti TEST_DIR dengan TRAIN_DIR
        class_dir = config.TRAIN_DIR / class_name  # Asumsi TRAIN_DIR ada di config
        
        if class_dir.exists():
            # Get first image from training class directory
            images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
            
            if images:
                with cols[idx]:
                    st.markdown(f"**{class_name}**")
                    sample_image = Image.open(images[0])
                    st.image(sample_image, use_container_width=True)
                    # Optional: Tampilkan jumlah gambar di folder
                    st.caption(f"{len(images)} images")
            else:
                with cols[idx]:
                    st.markdown(f"**{class_name}**")
                    st.info("No training images found")
        else:
            with cols[idx]:
                st.markdown(f"**{class_name}**")
                st.warning("Training directory not found")


def main():
    st.title("🌿 " + config.STREAMLIT_TITLE)
    st.markdown(config.STREAMLIT_DESCRIPTION)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        # Model selection
        model_type = st.selectbox(
            "Select Model",
            options=['resnet', 'cnn'],
            format_func=lambda x: "ResNet50 (Transfer Learning)" if x == 'resnet' else "Custom CNN",
            help="Choose which model to use for prediction"
        )
        
        st.markdown("---")
        
        # About section
        st.markdown("## ℹ️ About")
        st.markdown(
            """
            This application uses deep learning to classify plant diseases 
            from leaf images. It can detect:
            
            - 🟢 **Healthy** plants
            - 🟠 **Early Blight**
            - 🔴 **Late Blight**
            
            Upload a clear image of a plant leaf for analysis.
            """
        )
    
    # Load model
    with st.spinner(f"Loading {model_type.upper()} model..."):
        model, is_trained = load_trained_model(model_type)
    
    if model is None:
        st.error("Failed to load model. Please check the configuration.")
        return
    
    if not is_trained:
        st.warning("Using untrained model for demonstration. Train the model first for accurate predictions.")
    
    # Main content tabs
    tab1, tab2 = st.tabs(["📸 Upload & Predict", "🖼️ Sample Images"])
    
    with tab1:
        st.markdown("### Upload Potatoes Plant Leaf Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of potatoes plant leaf"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### Original Image")
                image = Image.open(uploaded_file)
                st.image(image, use_container_width=True, caption="Uploaded Image", clamp=True, channels="RGB")
            
            with col2:
                # Make prediction
                with st.spinner("Analyzing image..."):
                    predicted_class, confidence, all_probs = predict_image(model, image)
                
                # Display results
                display_prediction_results(predicted_class, confidence, all_probs)
            
            # Display disease information
            st.markdown("---")
            display_disease_info(predicted_class)
        
        else:
            st.info("Please upload an image to get started")
    
    with tab2:
        show_sample_images()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            <p>🌿 Potatoes Disease Classification System | Built with PyTorch & Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
