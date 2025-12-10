"""
Streamlit web application for plant disease classification.
Upload images and get real-time predictions.
"""
import streamlit as st
import torch
from PIL import Image
import numpy as np
from pathlib import Path
import json
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
    """
    Load trained model from checkpoint.
    Cached to avoid reloading on every interaction.
    
    Args:
        model_type: Type of model ('cnn' or 'resnet')
        
    Returns:
        Loaded model
    """
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
    """
    Predict disease class for an image.
    
    Args:
        model: Trained PyTorch model
        image: PIL Image
        
    Returns:
        Tuple of (predicted_class_name, confidence, all_probabilities)
    """
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
    """
    Display prediction results with confidence scores.
    
    Args:
        predicted_class: Predicted class name
        confidence: Confidence score
        all_probs: All class probabilities
    """
    # Display main prediction
    st.markdown("### 🎯 Prediction Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Color code based on disease
        if predicted_class == "Healthy":
            color = "green"
            emoji = "✅"
        elif "Blight" in predicted_class:
            color = "red"
            emoji = "⚠️"
        else:
            color = "orange"
            emoji = "ℹ️"
        
        st.markdown(
            f"<h2 style='color: {color};'>{emoji} {predicted_class}</h2>",
            unsafe_allow_html=True
        )
        st.metric("Confidence", f"{confidence * 100:.2f}%")
    
    with col2:
        # Display all class probabilities
        st.markdown("#### Class Probabilities:")
        for i, class_name in enumerate(config.CLASS_NAMES):
            prob = all_probs[i]
            st.progress(float(prob), text=f"{class_name}: {prob * 100:.2f}%")


def display_disease_info(predicted_class: str):
    """
    Display information about the predicted disease.
    
    Args:
        predicted_class: Predicted class name
    """
    disease_info = {
        "Healthy": {
            "description": "The plant appears healthy with no visible signs of disease.",
            "recommendation": "Continue regular care and monitoring.",
            "color": "green"
        },
        "Early Blight": {
            "description": "Early blight is a common fungal disease caused by Alternaria solani. "
                         "It affects tomato and potato plants, causing dark spots with concentric rings on leaves.",
            "recommendation": "Remove infected leaves, improve air circulation, apply fungicides, "
                            "and practice crop rotation.",
            "color": "orange"
        },
        "Late Blight": {
            "description": "Late blight is a devastating disease caused by Phytophthora infestans. "
                         "It can quickly destroy entire crops if not controlled.",
            "recommendation": "Remove and destroy infected plants immediately, apply copper-based fungicides, "
                            "ensure good drainage, and avoid overhead watering.",
            "color": "red"
        }
    }
    
    if predicted_class in disease_info:
        info = disease_info[predicted_class]
        
        st.markdown(f"### About : {predicted_class}")
        st.markdown(f"**Description:** {info['description']}")
        st.markdown(f"**Recommendation:** {info['recommendation']}")


def display_model_performance():
    """Display model performance metrics if available."""
    st.markdown("### 📊 Model Performance")
    
    # Check if evaluation results exist
    results_file = config.RESULTS_DIR / "evaluation_results.json"
    
    if results_file.exists():
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Overall Accuracy", f"{results.get('accuracy', 0):.2f}%")
            
            with col2:
                st.metric("Validation Loss", f"{results.get('val_loss', 0):.4f}")
            
            with col3:
                st.metric("Test Samples", results.get('test_samples', 'N/A'))
            
            # Display per-class metrics if available
            if 'per_class_metrics' in results:
                st.markdown("#### Per-Class Performance")
                metrics_data = []
                for class_name, metrics in results['per_class_metrics'].items():
                    metrics_data.append({
                        'Class': class_name,
                        'Precision': f"{metrics['precision']:.4f}",
                        'Recall': f"{metrics['recall']:.4f}",
                        'F1-Score': f"{metrics['f1_score']:.4f}"
                    })
                
                st.table(metrics_data)
        
        except Exception as e:
            st.warning(f"Could not load performance metrics: {e}")
    else:
        st.info("Train and evaluate the model to see performance metrics here.")


def show_sample_images():
    """Display sample images from each class."""
    st.markdown("### 🖼️ Sample Images by Class")
    
    cols = st.columns(len(config.CLASS_NAMES))
    
    for idx, class_name in enumerate(config.CLASS_NAMES):
        class_dir = config.TEST_DIR / class_name
        
        if class_dir.exists():
            # Get first image from class directory
            images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
            
            if images:
                with cols[idx]:
                    st.markdown(f"**{class_name}**")
                    sample_image = Image.open(images[0])
                    st.image(sample_image, use_container_width=True)
            else:
                with cols[idx]:
                    st.markdown(f"**{class_name}**")
                    st.info("No sample image")
        else:
            with cols[idx]:
                st.markdown(f"**{class_name}**")
                st.warning("Directory not found")


def main():
    """Main application function."""
    
    # Title and description
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
        
        st.markdown("---")
        
        # Model info
        st.markdown("## 🔧 Model Info")
        st.markdown(f"**Device:** {config.DEVICE}")
        st.markdown(f"**Image Size:** {config.IMAGE_SIZE}x{config.IMAGE_SIZE}")
        st.markdown(f"**Classes:** {config.NUM_CLASSES}")
    
    # Load model
    with st.spinner(f"Loading {model_type.upper()} model..."):
        model, is_trained = load_trained_model(model_type)
    
    if model is None:
        st.error("Failed to load model. Please check the configuration.")
        return
    
    if not is_trained:
        st.warning("⚠️ Using untrained model for demonstration. Train the model first for accurate predictions.")
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📸 Upload & Predict", "📊 Model Performance", "🖼️ Sample Images"])
    
    with tab1:
        st.markdown("## Upload Plant Leaf Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of a plant leaf"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### Original Image")
                image = Image.open(uploaded_file)
                st.image(image, use_container_width=True)
            
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
            st.info("👆 Please upload an image to get started")
    
    with tab2:
        display_model_performance()
    
    with tab3:
        show_sample_images()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            <p>🌿 Plant Disease Classification System | Built with PyTorch & Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
