"""
Utility functions for plotting, model management, and benchmarking.
"""
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import config


def plot_training_history(
    history: Dict,
    save_path: Optional[Path] = None,
    model_name: str = "model"
):
    """
    Plot training and validation loss/accuracy curves.
    
    Args:
        history: Dictionary containing training history
        save_path: Path to save the plot
        model_name: Name of the model for the title
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_title(f'{model_name} - Loss Curves', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[1].set_title(f'{model_name} - Accuracy Curves', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.show()


def plot_learning_rate(
    history: Dict,
    save_path: Optional[Path] = None,
    model_name: str = "model"
):
    """
    Plot learning rate schedule.
    
    Args:
        history: Dictionary containing training history
        save_path: Path to save the plot
        model_name: Name of the model for the title
    """
    if 'learning_rates' not in history:
        print("No learning rate information in history")
        return
    
    plt.figure(figsize=(10, 5))
    epochs = range(1, len(history['learning_rates']) + 1)
    
    plt.plot(epochs, history['learning_rates'], 'g-', linewidth=2)
    plt.title(f'{model_name} - Learning Rate Schedule', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Learning rate plot saved to {save_path}")
    
    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[Path] = None,
    model_name: str = "model",
    normalize: bool = False
):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save the plot
        model_name: Name of the model for the title
        normalize: Whether to normalize the confusion matrix
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
        title = f'{model_name} - Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = f'{model_name} - Confusion Matrix'
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
    )
    
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_sample_predictions(
    images: torch.Tensor,
    true_labels: torch.Tensor,
    pred_labels: torch.Tensor,
    class_names: List[str],
    num_samples: int = 9,
    save_path: Optional[Path] = None
):
    """
    Plot sample images with their true and predicted labels.
    
    Args:
        images: Batch of images
        true_labels: True labels
        pred_labels: Predicted labels
        class_names: List of class names
        num_samples: Number of samples to plot
        save_path: Path to save the plot
    """
    num_samples = min(num_samples, len(images))
    cols = 3
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    # Denormalize images
    mean = torch.tensor(config.IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(config.IMAGENET_STD).view(3, 1, 1)
    
    for idx in range(num_samples):
        # Denormalize
        img = images[idx].cpu() * std + mean
        img = torch.clamp(img, 0, 1)
        img = img.permute(1, 2, 0).numpy()
        
        true_label = class_names[true_labels[idx]]
        pred_label = class_names[pred_labels[idx]]
        
        # Determine color based on correctness
        color = 'green' if true_label == pred_label else 'red'
        
        axes[idx].imshow(img)
        axes[idx].set_title(
            f'True: {true_label}\nPred: {pred_label}',
            color=color,
            fontweight='bold',
            fontsize=10
        )
        axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sample predictions saved to {save_path}")
    
    plt.show()


def save_model(
    model: nn.Module,
    save_path: Path,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    additional_info: Optional[Dict] = None
):
    """
    Save model checkpoint with optional additional information.
    
    Args:
        model: PyTorch model to save
        save_path: Path to save the model
        optimizer: Optional optimizer to save
        epoch: Optional epoch number
        additional_info: Optional dictionary with additional information
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    
    if additional_info is not None:
        checkpoint.update(additional_info)
    
    torch.save(checkpoint, save_path)
    print(f"Model saved to {save_path}")


def load_model(
    model: nn.Module,
    load_path: Path,
    device: torch.device = config.DEVICE,
    load_optimizer: bool = False,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> Tuple[nn.Module, Optional[torch.optim.Optimizer], Optional[int]]:
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model to load weights into
        load_path: Path to load the model from
        device: Device to load the model on
        load_optimizer: Whether to load optimizer state
        optimizer: Optimizer to load state into
        
    Returns:
        Tuple of (model, optimizer, epoch)
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Model file not found: {load_path}")
    
    checkpoint = torch.load(load_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    loaded_optimizer = None
    if load_optimizer and optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loaded_optimizer = optimizer
    
    epoch = checkpoint.get('epoch', None)
    
    print(f"Model loaded from {load_path}")
    if epoch is not None:
        print(f"Loaded from epoch {epoch}")
    
    return model, loaded_optimizer, epoch


def benchmark_model(
    model: nn.Module,
    dataloader,
    device: torch.device = config.DEVICE,
    num_iterations: int = 100
) -> Dict[str, float]:
    """
    Benchmark model inference performance.
    
    Args:
        model: PyTorch model to benchmark
        dataloader: DataLoader for data
        device: Device to run on
        num_iterations: Number of iterations to benchmark
        
    Returns:
        Dictionary with benchmark results
    """
    model.eval()
    model = model.to(device)
    
    # Warmup
    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            if i >= 10:
                break
            images = images.to(device)
            _ = model(images)
    
    # Benchmark
    times = []
    
    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            if i >= num_iterations:
                break
            
            images = images.to(device)
            
            # Synchronize for accurate timing
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start = time.time()
            _ = model(images)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end = time.time()
            times.append(end - start)
    
    times = np.array(times)
    
    results = {
        'mean_time': times.mean(),
        'std_time': times.std(),
        'min_time': times.min(),
        'max_time': times.max(),
        'median_time': np.median(times),
        'throughput': config.BATCH_SIZE / times.mean(),  # images/second
    }
    
    return results


def print_benchmark_results(results: Dict[str, float], model_name: str = "Model"):
    """
    Print benchmark results in a formatted table.
    
    Args:
        results: Dictionary with benchmark results
        model_name: Name of the model
    """
    print(f"\n{'='*60}")
    print(f"{model_name} Benchmark Results")
    print(f"{'='*60}")
    print(f"Mean inference time:   {results['mean_time']*1000:.2f} ms")
    print(f"Std inference time:    {results['std_time']*1000:.2f} ms")
    print(f"Min inference time:    {results['min_time']*1000:.2f} ms")
    print(f"Max inference time:    {results['max_time']*1000:.2f} ms")
    print(f"Median inference time: {results['median_time']*1000:.2f} ms")
    print(f"Throughput:            {results['throughput']:.2f} images/sec")
    print(f"{'='*60}\n")


def compare_models(
    results1: Dict[str, float],
    results2: Dict[str, float],
    model1_name: str = "Model 1",
    model2_name: str = "Model 2"
):
    """
    Compare benchmark results of two models.
    
    Args:
        results1: Benchmark results for first model
        results2: Benchmark results for second model
        model1_name: Name of first model
        model2_name: Name of second model
    """
    print(f"\n{'='*80}")
    print(f"Model Comparison: {model1_name} vs {model2_name}")
    print(f"{'='*80}")
    print(f"{'Metric':<25} {model1_name:<20} {model2_name:<20} {'Difference':<15}")
    print(f"{'-'*80}")
    
    metrics = [
        ('Mean time (ms)', 'mean_time', 1000),
        ('Throughput (img/s)', 'throughput', 1),
    ]
    
    for metric_name, key, scale in metrics:
        val1 = results1[key] * scale
        val2 = results2[key] * scale
        diff = ((val2 - val1) / val1) * 100
        
        print(f"{metric_name:<25} {val1:<20.2f} {val2:<20.2f} {diff:>+.2f}%")
    
    print(f"{'='*80}\n")


if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    # Test plotting with dummy data
    dummy_history = {
        'train_loss': [0.8, 0.6, 0.5, 0.4, 0.35, 0.3],
        'val_loss': [0.85, 0.65, 0.55, 0.5, 0.45, 0.42],
        'train_acc': [60, 70, 75, 80, 82, 85],
        'val_acc': [58, 68, 72, 76, 78, 80],
        'learning_rates': [0.001, 0.001, 0.001, 0.0005, 0.0005, 0.00025]
    }
    
    print("\nPlotting training history...")
    plot_training_history(dummy_history, model_name="Test Model")
    
    print("\nPlotting learning rate schedule...")
    plot_learning_rate(dummy_history, model_name="Test Model")
    
    print("\nUtility functions test completed!")
