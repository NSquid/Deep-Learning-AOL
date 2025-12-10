import os
import torch
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "Dataset"
TRAIN_DIR = DATASET_DIR / "Training"
VAL_DIR = DATASET_DIR / "Validation"
TEST_DIR = DATASET_DIR / "Testing"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
LOGS_DIR = BASE_DIR / "logs"
RESULTS_DIR = BASE_DIR / "results"

# Create directories if they don't exist
CHECKPOINT_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Class names
CLASS_NAMES = ['Early_Blight', 'Healthy', 'Late_Blight']
NUM_CLASSES = len(CLASS_NAMES)

# Image settings
IMAGE_SIZE = 224
IMAGE_CHANNELS = 3

# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Data augmentation parameters
ROTATION_DEGREES = 20
COLOR_JITTER_BRIGHTNESS = 0.2
COLOR_JITTER_CONTRAST = 0.2
COLOR_JITTER_SATURATION = 0.2
COLOR_JITTER_HUE = 0.1

# DataLoader settings
BATCH_SIZE = 32
NUM_WORKERS = 4
SHUFFLE_TRAIN = True
SHUFFLE_VAL = False
SHUFFLE_TEST = False

# Training hyperparameters
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5

# Learning rate scheduler
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 3
LR_SCHEDULER_MIN_LR = 1e-7

# Model settings
DROPOUT_RATE = 0.5
RESNET_PRETRAINED = True
RESNET_FINE_TUNE_LAYERS = 3  # Fine-tune last 3 layers

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# TensorBoard settings
TENSORBOARD_LOG_DIR = LOGS_DIR / "tensorboard"
TENSORBOARD_LOG_DIR.mkdir(exist_ok=True)

# Model checkpoint settings
SAVE_BEST_ONLY = True
CHECKPOINT_FILENAME = "best_model.pth"

# Random seed for reproducibility
RANDOM_SEED = 42

# Streamlit settings
STREAMLIT_TITLE = "Potatoes Disease Classification"
STREAMLIT_DESCRIPTION = """
Upload a potatoes plant leaf image to detect diseases. 
This model can classify three conditions: Early Blight, Healthy, and Late Blight.
"""
CONFIDENCE_THRESHOLD = 0.5
