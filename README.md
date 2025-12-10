# Plant Disease Classification using Deep Learning

A production-ready PyTorch implementation for classifying plant diseases from leaf images using both custom CNN and transfer learning approaches.

## Features

- **Two Model Architectures**: Custom CNN and ResNet50 transfer learning
- **Complete Training Pipeline**: With early stopping, learning rate scheduling, and checkpointing
- **Comprehensive Evaluation**: Confusion matrix, classification reports, and visualizations
- **Interactive Web App**: Streamlit-based UI for real-time predictions
- **Production-Ready**: Error handling, logging, type hints, and modular code structure

## Dataset Structure

```
Dataset/
├── Training/
│   ├── Early_Blight/
│   ├── Healthy/
│   └── Late_Blight/
├── Validation/
│   ├── Early_Blight/
│   ├── Healthy/
│   └── Late_Blight/
└── Testing/
    ├── Early_Blight/
    ├── Healthy/
    └── Late_Blight/
```

## Installation

1. Clone the repository and navigate to the project directory

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

Train both models (Custom CNN and ResNet50):
```bash
python main.py --mode train --model all
```

Train specific model:
```bash
python main.py --mode train --model cnn
python main.py --mode train --model resnet
```

### Evaluation

Evaluate trained models:
```bash
python main.py --mode evaluate --model all
```

### Streamlit Application

Launch the web interface:
```bash
streamlit run app.py
```

## Project Structure

- `config.py` - Centralized configuration and hyperparameters
- `data.py` - Dataset class and data loading utilities
- `model.py` - Model architectures (Custom CNN and ResNet50)
- `train.py` - Training pipeline with all training logic
- `evaluate.py` - Evaluation metrics and visualization
- `utils.py` - Utility functions for plotting and model management
- `app.py` - Streamlit web application
- `main.py` - Main execution script

## Model Architectures

### Custom CNN
- 4 convolutional layers with batch normalization
- MaxPooling and Dropout for regularization
- Fully connected layers with adaptive pooling

### ResNet50 Transfer Learning
- Pretrained on ImageNet
- Fine-tuning last 3 layers
- Custom classifier head

## Performance
Results and trained models are saved in:
- `checkpoints/` - Model weights
- `logs/` - TensorBoard logs
- `results/` - Evaluation plots and metrics