"""
Model architectures for plant disease classification.
Includes Custom CNN and ResNet50 transfer learning models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, Tuple
import config


class CustomCNN(nn.Module):
    """
    Custom CNN architecture with 4 convolutional layers.
    
    Architecture:
        - 4 Conv blocks (Conv2D -> BatchNorm -> ReLU -> MaxPool)
        - Adaptive Average Pooling
        - Fully connected layers with Dropout
    
    Args:
        num_classes: Number of output classes
        dropout_rate: Dropout probability
    """
    
    def __init__(self, num_classes: int = config.NUM_CLASSES, dropout_rate: float = config.DROPOUT_RATE):
        super(CustomCNN, self).__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Convolutional Block 1: 3 -> 32 channels
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolutional Block 2: 32 -> 64 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolutional Block 3: 64 -> 128 channels
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolutional Block 4: 128 -> 256 channels
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 7 * 7, 512)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(256, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Conv Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Conv Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Conv Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # Conv Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool4(x)
        
        # Adaptive pooling
        x = self.adaptive_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x
    
    def get_model_summary(self) -> str:
        """Get a string summary of the model architecture."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        summary = f"""
Custom CNN Model Summary:
========================
Total parameters: {total_params:,}
Trainable parameters: {trainable_params:,}
Non-trainable parameters: {total_params - trainable_params:,}

Architecture:
- Conv Block 1: 3 -> 32 channels
- Conv Block 2: 32 -> 64 channels
- Conv Block 3: 64 -> 128 channels
- Conv Block 4: 128 -> 256 channels
- FC Layer 1: 12544 -> 512
- FC Layer 2: 512 -> 256
- Output Layer: 256 -> {self.num_classes}
- Dropout Rate: {self.dropout_rate}
========================
"""
        return summary


class ResNet50Transfer(nn.Module):
    """
    ResNet50 transfer learning model.
    
    Uses pretrained ResNet50 on ImageNet with option to fine-tune last N layers.
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        fine_tune_layers: Number of last layers to fine-tune (0 = freeze all)
    """
    
    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        pretrained: bool = config.RESNET_PRETRAINED,
        fine_tune_layers: int = config.RESNET_FINE_TUNE_LAYERS
    ):
        super(ResNet50Transfer, self).__init__()
        
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.fine_tune_layers = fine_tune_layers
        
        # Load pretrained ResNet50
        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            self.resnet = models.resnet50(weights=weights)
        else:
            self.resnet = models.resnet50(weights=None)
        
        # Get the number of features from the last layer
        num_features = self.resnet.fc.in_features
        
        # Replace the final fully connected layer
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(512, num_classes)
        )
        
        # Freeze layers if specified
        self._freeze_layers()
    
    def _freeze_layers(self):
        """Freeze all layers except the last N layers for fine-tuning."""
        if self.fine_tune_layers == 0:
            # Freeze all layers except the classifier
            for param in self.resnet.parameters():
                param.requires_grad = False
            # Unfreeze the new classifier
            for param in self.resnet.fc.parameters():
                param.requires_grad = True
        else:
            # Freeze all layers first
            for param in self.resnet.parameters():
                param.requires_grad = False
            
            # Unfreeze the classifier
            for param in self.resnet.fc.parameters():
                param.requires_grad = True
            
            # Unfreeze last N layers based on fine_tune_layers parameter
            if self.fine_tune_layers >= 1:
                for param in self.resnet.layer4.parameters():
                    param.requires_grad = True
            if self.fine_tune_layers >= 2:
                for param in self.resnet.layer3.parameters():
                    param.requires_grad = True
            if self.fine_tune_layers >= 3:
                for param in self.resnet.layer2.parameters():
                    param.requires_grad = True
            if self.fine_tune_layers >= 4:
                for param in self.resnet.layer1.parameters():
                    param.requires_grad = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        return self.resnet(x)
    
    def get_model_summary(self) -> str:
        """Get a string summary of the model architecture."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        summary = f"""
ResNet50 Transfer Learning Model Summary:
=========================================
Total parameters: {total_params:,}
Trainable parameters: {trainable_params:,}
Non-trainable parameters: {total_params - trainable_params:,}

Configuration:
- Pretrained: {self.pretrained}
- Fine-tune layers: {self.fine_tune_layers}
- Output classes: {self.num_classes}

Fine-tuning Strategy:
"""
        if self.fine_tune_layers == 0:
            summary += "- Only training the classifier head\n"
        else:
            summary += f"- Training classifier + last {self.fine_tune_layers} layer(s)\n"
        
        summary += "========================================="
        
        return summary


def create_model(model_type: str = 'cnn', device: torch.device = config.DEVICE) -> nn.Module:
    """
    Factory function to create and initialize models.
    
    Args:
        model_type: Type of model ('cnn' or 'resnet')
        device: Device to place the model on
        
    Returns:
        Initialized model
    """
    if model_type.lower() == 'cnn':
        model = CustomCNN(num_classes=config.NUM_CLASSES, dropout_rate=config.DROPOUT_RATE)
        print("Created Custom CNN model")
    elif model_type.lower() == 'resnet':
        model = ResNet50Transfer(
            num_classes=config.NUM_CLASSES,
            pretrained=config.RESNET_PRETRAINED,
            fine_tune_layers=config.RESNET_FINE_TUNE_LAYERS
        )
        print("Created ResNet50 transfer learning model")
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'cnn' or 'resnet'")
    
    # Move model to device
    model = model.to(device)
    
    # Print model summary
    print(model.get_model_summary())
    
    return model


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


if __name__ == "__main__":
    # Test model creation
    print("Testing model architectures...\n")
    
    # Test Custom CNN
    print("=" * 60)
    print("Testing Custom CNN")
    print("=" * 60)
    cnn_model = create_model('cnn', device='cpu')
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 224, 224)
    output = cnn_model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test ResNet50
    print("\n" + "=" * 60)
    print("Testing ResNet50 Transfer Learning")
    print("=" * 60)
    resnet_model = create_model('resnet', device='cpu')
    
    # Test forward pass
    output = resnet_model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    print("\nModel architecture tests completed successfully!")
