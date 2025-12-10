"""
Main execution script for plant disease classification project.
Orchestrates training, evaluation, and model management.
"""
import argparse
import sys
import json
from pathlib import Path
import torch
import config
from data import create_dataloaders, print_dataset_info
from train import train_model
from evaluate import comprehensive_evaluation
from model import create_model
from utils import (
    load_model,
    plot_training_history,
    plot_learning_rate,
    compare_models,
    benchmark_model,
    print_benchmark_results
)


def setup_environment():
    """Set up the training environment."""
    # Set random seeds for reproducibility
    torch.manual_seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"\n{'='*80}")
    print("ENVIRONMENT SETUP")
    print(f"{'='*80}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Device: {config.DEVICE}")
    print(f"Random seed: {config.RANDOM_SEED}")
    print(f"{'='*80}\n")


def train_models(args):
    """
    Train models based on arguments.
    
    Args:
        args: Command line arguments
    """
    print("\n" + "="*80)
    print("TRAINING MODE")
    print("="*80 + "\n")
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Print dataset info
    print_dataset_info()
    
    models_to_train = []
    if args.model == 'all':
        models_to_train = ['cnn', 'resnet']
    else:
        models_to_train = [args.model]
    
    results = {}
    
    for model_type in models_to_train:
        print(f"\n{'='*80}")
        print(f"Training {model_type.upper()} Model")
        print(f"{'='*80}\n")
        
        # Train model
        model, history = train_model(
            model_type=model_type,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate,
            device=config.DEVICE
        )
        
        results[model_type] = history
        
        # Plot training history
        if args.plot:
            print(f"\nGenerating plots for {model_type}...")
            plot_path = config.RESULTS_DIR / f"{model_type}_training_history.png"
            plot_training_history(history, plot_path, model_type.upper())
            
            lr_plot_path = config.RESULTS_DIR / f"{model_type}_learning_rate.png"
            plot_learning_rate(history, lr_plot_path, model_type.upper())
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETED")
    print(f"{'='*80}\n")
    
    return results


def evaluate_models(args):
    """
    Evaluate models based on arguments.
    
    Args:
        args: Command line arguments
    """
    print("\n" + "="*80)
    print("EVALUATION MODE")
    print("="*80 + "\n")
    
    # Create dataloaders
    print("Creating dataloaders...")
    _, _, test_loader = create_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    models_to_evaluate = []
    if args.model == 'all':
        models_to_evaluate = ['cnn', 'resnet']
    else:
        models_to_evaluate = [args.model]
    
    evaluation_results = {}
    benchmark_results = {}
    
    for model_type in models_to_evaluate:
        print(f"\n{'='*80}")
        print(f"Evaluating {model_type.upper()} Model")
        print(f"{'='*80}\n")
        
        # Load model
        model = create_model(model_type, config.DEVICE)
        checkpoint_path = config.CHECKPOINT_DIR / f"{model_type}_model_best.pth"
        
        if not checkpoint_path.exists():
            print(f"Warning: Checkpoint not found at {checkpoint_path}")
            print("Skipping evaluation for this model.")
            continue
        
        model, _, epoch = load_model(model, checkpoint_path, config.DEVICE)
        print(f"Loaded model from epoch {epoch}")
        
        # Comprehensive evaluation
        results = comprehensive_evaluation(
            model=model,
            dataloader=test_loader,
            model_name=f"{model_type}_model",
            device=config.DEVICE,
            save_plots=args.plot,
            run_benchmark=args.benchmark
        )
        
        evaluation_results[model_type] = results
        
        if args.benchmark and results['benchmark']:
            benchmark_results[model_type] = results['benchmark']
    
    # Compare models if both were evaluated
    if len(benchmark_results) == 2 and args.benchmark:
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)
        compare_models(
            benchmark_results['cnn'],
            benchmark_results['resnet'],
            "Custom CNN",
            "ResNet50"
        )
    
    # Save evaluation results
    save_evaluation_results(evaluation_results)
    
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETED")
    print(f"{'='*80}\n")
    
    return evaluation_results


def save_evaluation_results(evaluation_results: dict):
    """
    Save evaluation results to JSON file.
    
    Args:
        evaluation_results: Dictionary containing evaluation results
    """
    results_to_save = {}
    
    for model_type, results in evaluation_results.items():
        results_to_save[model_type] = {
            'accuracy': float(results['accuracy']),
            'loss': float(results['loss']),
            'per_class_metrics': {
                class_name: {
                    k: float(v) if isinstance(v, (int, float)) else int(v)
                    for k, v in metrics.items()
                }
                for class_name, metrics in results['per_class_metrics'].items()
            }
        }
        
        if results['benchmark']:
            results_to_save[model_type]['benchmark'] = {
                k: float(v) for k, v in results['benchmark'].items()
            }
    
    # Save to file
    results_file = config.RESULTS_DIR / "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results_to_save, f, indent=4)
    
    print(f"\nEvaluation results saved to {results_file}")


def main():
    """Main function to parse arguments and execute."""
    parser = argparse.ArgumentParser(
        description="Plant Disease Classification - Training and Evaluation"
    )
    
    # Mode selection
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'evaluate', 'both'],
        default='both',
        help='Execution mode: train, evaluate, or both'
    )
    
    # Model selection
    parser.add_argument(
        '--model',
        type=str,
        choices=['cnn', 'resnet', 'all'],
        default='all',
        help='Model to train/evaluate: cnn, resnet, or all'
    )
    
    # Training parameters
    parser.add_argument(
        '--epochs',
        type=int,
        default=config.NUM_EPOCHS,
        help=f'Number of training epochs (default: {config.NUM_EPOCHS})'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=config.BATCH_SIZE,
        help=f'Batch size (default: {config.BATCH_SIZE})'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=config.LEARNING_RATE,
        help=f'Learning rate (default: {config.LEARNING_RATE})'
    )
    
    parser.add_argument(
        '--num-workers',
        type=int,
        default=config.NUM_WORKERS,
        help=f'Number of data loading workers (default: {config.NUM_WORKERS})'
    )
    
    # Plotting and benchmarking
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Disable plotting'
    )
    
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Run performance benchmark during evaluation'
    )
    
    args = parser.parse_args()
    args.plot = not args.no_plot
    
    # Setup environment
    setup_environment()
    
    # Execute based on mode
    if args.mode in ['train', 'both']:
        train_models(args)
    
    if args.mode in ['evaluate', 'both']:
        evaluate_models(args)
    
    print("\n" + "="*80)
    print("ALL TASKS COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nNext steps:")
    print("1. Review results in the 'results/' directory")
    print("2. Check TensorBoard logs: tensorboard --logdir logs/tensorboard")
    print("3. Launch the Streamlit app: streamlit run app.py")
    print("="*80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
