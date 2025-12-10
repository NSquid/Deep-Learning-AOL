"""
Evaluation script for plant disease classification models.
Generates confusion matrix, classification reports, and visualizations.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, List
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
    accuracy_score
)
from tqdm import tqdm
import config
from utils import (
    plot_confusion_matrix,
    plot_sample_predictions,
    load_model,
    benchmark_model,
    print_benchmark_results
)


def evaluate_model(
    model: nn.Module,
    dataloader,
    device: torch.device = config.DEVICE,
    return_predictions: bool = False
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: PyTorch model to evaluate
        dataloader: DataLoader for evaluation data
        device: Device to evaluate on
        return_predictions: Whether to return all predictions
        
    Returns:
        Tuple of (accuracy, loss, true_labels, predicted_labels)
        If return_predictions=True, also returns (all_images, all_probs)
    """
    model.eval()
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    all_labels = []
    all_predictions = []
    all_probs = []
    all_images_list = []
    total_loss = 0.0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
        
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Get predictions
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            # Store results
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            if return_predictions:
                all_images_list.extend(images.cpu())
            
            total_loss += loss.item() * images.size(0)
    
    # Calculate metrics
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probs = np.array(all_probs)
    
    accuracy = accuracy_score(all_labels, all_predictions) * 100
    avg_loss = total_loss / len(all_labels)
    
    if return_predictions:
        return accuracy, avg_loss, all_labels, all_predictions, all_probs, torch.stack(all_images_list)
    
    return accuracy, avg_loss, all_labels, all_predictions


def generate_classification_report(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    class_names: List[str] = config.CLASS_NAMES
) -> str:
    """
    Generate classification report with per-class metrics.
    
    Args:
        true_labels: True labels
        pred_labels: Predicted labels
        class_names: List of class names
        
    Returns:
        Classification report string
    """
    report = classification_report(
        true_labels,
        pred_labels,
        target_names=class_names,
        digits=4
    )
    
    return report


def calculate_per_class_metrics(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    class_names: List[str] = config.CLASS_NAMES
) -> Dict[str, Dict[str, float]]:
    """
    Calculate per-class precision, recall, and F1-score.
    
    Args:
        true_labels: True labels
        pred_labels: Predicted labels
        class_names: List of class names
        
    Returns:
        Dictionary with per-class metrics
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels,
        pred_labels,
        labels=range(len(class_names)),
        zero_division=0
    )
    
    metrics = {}
    for i, class_name in enumerate(class_names):
        metrics[class_name] = {
            'precision': precision[i],
            'recall': recall[i],
            'f1_score': f1[i],
            'support': support[i]
        }
    
    return metrics


def print_evaluation_results(
    accuracy: float,
    loss: float,
    metrics: Dict[str, Dict[str, float]],
    model_name: str = "Model"
):
    """
    Print evaluation results in a formatted table.
    
    Args:
        accuracy: Overall accuracy
        loss: Average loss
        metrics: Per-class metrics dictionary
        model_name: Name of the model
    """
    print(f"\n{'='*80}")
    print(f"{model_name} - Evaluation Results")
    print(f"{'='*80}")
    print(f"Overall Accuracy: {accuracy:.2f}%")
    print(f"Average Loss: {loss:.4f}")
    print(f"\n{'='*80}")
    print(f"Per-Class Metrics:")
    print(f"{'='*80}")
    print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print(f"{'-'*80}")
    
    for class_name, class_metrics in metrics.items():
        print(
            f"{class_name:<20} "
            f"{class_metrics['precision']:<12.4f} "
            f"{class_metrics['recall']:<12.4f} "
            f"{class_metrics['f1_score']:<12.4f} "
            f"{int(class_metrics['support']):<10}"
        )
    
    print(f"{'='*80}\n")


def comprehensive_evaluation(
    model: nn.Module,
    dataloader,
    model_name: str = "model",
    device: torch.device = config.DEVICE,
    save_plots: bool = True,
    run_benchmark: bool = True
) -> Dict:
    """
    Perform comprehensive evaluation of a model.
    
    Args:
        model: PyTorch model to evaluate
        dataloader: DataLoader for evaluation data
        model_name: Name of the model
        device: Device to evaluate on
        save_plots: Whether to save plots
        run_benchmark: Whether to run performance benchmark
        
    Returns:
        Dictionary with all evaluation results
    """
    print(f"\n{'='*80}")
    print(f"Comprehensive Evaluation: {model_name}")
    print(f"{'='*80}\n")
    
    # Evaluate model
    print("Evaluating model...")
    accuracy, loss, true_labels, pred_labels, probs, images = evaluate_model(
        model, dataloader, device, return_predictions=True
    )
    
    # Calculate metrics
    print("Calculating metrics...")
    per_class_metrics = calculate_per_class_metrics(true_labels, pred_labels)
    
    # Generate classification report
    report = generate_classification_report(true_labels, pred_labels)
    
    # Print results
    print_evaluation_results(accuracy, loss, per_class_metrics, model_name)
    print("\nDetailed Classification Report:")
    print(report)
    
    # Generate confusion matrix
    print("Generating confusion matrix...")
    cm = confusion_matrix(true_labels, pred_labels)
    
    # Plot confusion matrix
    save_path = config.RESULTS_DIR / f"{model_name}_confusion_matrix.png" if save_plots else None
    plot_confusion_matrix(cm, config.CLASS_NAMES, save_path, model_name)
    
    # Plot normalized confusion matrix
    save_path_norm = config.RESULTS_DIR / f"{model_name}_confusion_matrix_normalized.png" if save_plots else None
    plot_confusion_matrix(cm, config.CLASS_NAMES, save_path_norm, model_name, normalize=True)
    
    # Plot sample predictions
    print("Plotting sample predictions...")
    num_samples = min(9, len(images))
    indices = np.random.choice(len(images), num_samples, replace=False)
    
    sample_images = images[indices]
    sample_true = true_labels[indices]
    sample_pred = pred_labels[indices]
    
    save_path_samples = config.RESULTS_DIR / f"{model_name}_sample_predictions.png" if save_plots else None
    plot_sample_predictions(
        sample_images,
        torch.tensor(sample_true),
        torch.tensor(sample_pred),
        config.CLASS_NAMES,
        num_samples,
        save_path_samples
    )
    
    # Benchmark model
    benchmark_results = None
    if run_benchmark:
        print("Running performance benchmark...")
        benchmark_results = benchmark_model(model, dataloader, device)
        print_benchmark_results(benchmark_results, model_name)
    
    # Compile results
    results = {
        'accuracy': accuracy,
        'loss': loss,
        'confusion_matrix': cm,
        'per_class_metrics': per_class_metrics,
        'classification_report': report,
        'benchmark': benchmark_results
    }
    
    print(f"{'='*80}")
    print(f"Evaluation completed for {model_name}!")
    print(f"{'='*80}\n")
    
    return results


def evaluate_from_checkpoint(
    model_type: str,
    checkpoint_path,
    dataloader,
    device: torch.device = config.DEVICE
) -> Dict:
    """
    Load model from checkpoint and evaluate.
    
    Args:
        model_type: Type of model ('cnn' or 'resnet')
        checkpoint_path: Path to model checkpoint
        dataloader: DataLoader for evaluation data
        device: Device to evaluate on
        
    Returns:
        Dictionary with evaluation results
    """
    from model import create_model
    
    # Create model
    model = create_model(model_type, device)
    
    # Load checkpoint
    model, _, epoch = load_model(model, checkpoint_path, device)
    
    # Comprehensive evaluation
    results = comprehensive_evaluation(
        model,
        dataloader,
        model_name=f"{model_type}_model",
        device=device
    )
    
    return results


if __name__ == "__main__":
    # Test evaluation script
    print("Testing evaluation script...")
    
    try:
        from data import create_dataloaders
        from model import create_model
        
        # Create dataloaders
        print("Creating dataloaders...")
        _, _, test_loader = create_dataloaders()
        
        # Create a model (for testing)
        print("\nCreating test model...")
        model = create_model('cnn', device=config.DEVICE)
        
        # Run comprehensive evaluation
        print("\nRunning comprehensive evaluation...")
        results = comprehensive_evaluation(
            model,
            test_loader,
            model_name="test_cnn",
            device=config.DEVICE,
            save_plots=False,
            run_benchmark=False  # Skip benchmark for quick test
        )
        
        print("\nEvaluation script test completed successfully!")
        
    except Exception as e:
        print(f"Error during evaluation test: {e}")
        import traceback
        traceback.print_exc()
