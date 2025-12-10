import torch
import config
from data import create_dataloaders, print_dataset_info
from model import create_model, count_parameters
from train import train_model
from evaluate import comprehensive_evaluation
from utils import plot_training_history

# Example usage of the Plant Disease Classification project components.
def example_data_pipeline():
    print("\n" + "="*80)
    print("EXAMPLE 1: Data Pipeline")
    print("="*80 + "\n")
    
    # Print dataset information
    print_dataset_info()
    
    # Create DataLoaders
    print("\nCreating DataLoaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=32,
        num_workers=4
    )
    
    print(f"\nDataLoader Information:")
    print(f"- Training batches: {len(train_loader)}")
    print(f"- Validation batches: {len(val_loader)}")
    print(f"- Test batches: {len(test_loader)}")
    
    # Load a sample batch
    images, labels = next(iter(train_loader))
    print(f"\nSample Batch:")
    print(f"- Images shape: {images.shape}")
    print(f"- Labels shape: {labels.shape}")
    print(f"- Image value range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"- Unique labels in batch: {labels.unique().tolist()}")
    print(f"- Class names: {[config.CLASS_NAMES[i] for i in labels.unique().tolist()]}")

# Example: Model creation and parameter counting.
def example_model_creation():
    print("\n" + "="*80)
    print("EXAMPLE 2: Model Creation")
    print("="*80 + "\n")
    
    # Create Custom CNN
    print("Creating Custom CNN...")
    cnn_model = create_model('cnn', device='cpu')
    total_params, trainable_params = count_parameters(cnn_model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create ResNet50
    print("\n" + "-"*80 + "\n")
    print("Creating ResNet50 Transfer Learning Model...")
    resnet_model = create_model('resnet', device='cpu')
    total_params, trainable_params = count_parameters(resnet_model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    print("\n" + "-"*80 + "\n")
    print("Testing forward pass...")
    dummy_input = torch.randn(4, 3, 224, 224)
    
    with torch.no_grad():
        cnn_output = cnn_model(dummy_input)
        resnet_output = resnet_model(dummy_input)
    
    print(f"CNN output shape: {cnn_output.shape}")
    print(f"ResNet output shape: {resnet_output.shape}")
    print(f"\nBoth models produce outputs for {config.NUM_CLASSES} classes ✓")

# Example: Quick training demonstration (2 epochs).
def example_quick_training():
    print("\n" + "="*80)
    print("EXAMPLE 3: Quick Training (2 epochs)")
    print("="*80 + "\n")
    
    print("Creating DataLoaders...")
    train_loader, val_loader, _ = create_dataloaders(batch_size=32)
    
    print("\nTraining Custom CNN for 2 epochs (demo)...")
    model, history = train_model(
        model_type='cnn',
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=2,
        learning_rate=0.001,
        device=config.DEVICE
    )
    
    print("\nTraining History:")
    print(f"- Train Loss: {history['train_loss']}")
    print(f"- Train Accuracy: {history['train_acc']}")
    print(f"- Val Loss: {history['val_loss']}")
    print(f"- Val Accuracy: {history['val_acc']}")
    
    # Plot training history
    print("\nPlotting training curves...")
    plot_training_history(history, model_name="Quick Demo CNN")

# Example: Model evaluation on test set.
def example_model_evaluation():
    print("\n" + "="*80)
    print("EXAMPLE 4: Model Evaluation")
    print("="*80 + "\n")
    
    # Check if trained model exists
    checkpoint_path = config.CHECKPOINT_DIR / "cnn_model_best.pth"
    
    if not checkpoint_path.exists():
        print("⚠️  No trained model found.")
        print("Please train a model first using:")
        print("  python main.py --mode train --model cnn --epochs 5")
        return
    
    print("Creating test DataLoader...")
    _, _, test_loader = create_dataloaders()
    
    print("\nLoading trained model...")
    from utils import load_model
    model = create_model('cnn', device=config.DEVICE)
    model, _, epoch = load_model(model, checkpoint_path, config.DEVICE)
    
    print(f"Model loaded from epoch {epoch}")
    
    print("\nRunning comprehensive evaluation...")
    results = comprehensive_evaluation(
        model=model,
        dataloader=test_loader,
        model_name="CNN_Evaluation_Demo",
        device=config.DEVICE,
        save_plots=True,
        run_benchmark=True
    )
    
    print("\nEvaluation completed!")
    print(f"Accuracy: {results['accuracy']:.2f}%")
    print(f"Loss: {results['loss']:.4f}")

# Example: Single image prediction.
def example_prediction():
    print("\n" + "="*80)
    print("EXAMPLE 5: Single Image Prediction")
    print("="*80 + "\n")
    
    # Check if trained model exists
    checkpoint_path = config.CHECKPOINT_DIR / "resnet_model_best.pth"
    
    if not checkpoint_path.exists():
        print("⚠️  No trained model found.")
        print("Please train a model first using:")
        print("  python main.py --mode train --model resnet --epochs 5")
        return
    
    print("Loading trained model...")
    from utils import load_model
    from data import get_val_test_transforms
    from PIL import Image
    
    model = create_model('resnet', device=config.DEVICE)
    model, _, _ = load_model(model, checkpoint_path, config.DEVICE)
    model.eval()
    
    # Get a test image
    test_image_path = None
    for class_name in config.CLASS_NAMES:
        class_dir = config.TEST_DIR / class_name
        if class_dir.exists():
            images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
            if images:
                test_image_path = images[0]
                true_class = class_name
                break
    
    if test_image_path is None:
        print("⚠️  No test images found in Dataset/Testing/")
        return
    
    print(f"Loading test image: {test_image_path}")
    print(f"True class: {true_class}")
    
    # Preprocess and predict
    transform = get_val_test_transforms()
    image = Image.open(test_image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(config.DEVICE)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = probabilities.max(1)
    
    predicted_class = config.CLASS_NAMES[predicted_idx.item()]
    
    print(f"\n{'='*60}")
    print("Prediction Results:")
    print(f"{'='*60}")
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence.item()*100:.2f}%")
    print(f"True class: {true_class}")
    print(f"Correct: {'✓' if predicted_class == true_class else '✗'}")
    print(f"\nAll class probabilities:")
    for i, class_name in enumerate(config.CLASS_NAMES):
        prob = probabilities[0][i].item()
        print(f"  {class_name}: {prob*100:.2f}%")
    print(f"{'='*60}")


def main():
    print("\n" + "="*80)
    print("PLANT DISEASE CLASSIFICATION - EXAMPLE USAGE")
    print("="*80)
    
    # Run examples
    try:
        example_data_pipeline()
        input("\nPress Enter to continue to next example...")
        
        example_model_creation()
        input("\nPress Enter to continue to next example...")
        
        print("\nThe next example will train a model for 2 epochs (demo).")
        response = input("Continue? (y/n): ")
        if response.lower() == 'y':
            example_quick_training()
        
        input("\nPress Enter to continue to next example...")
        example_model_evaluation()
        
        input("\nPress Enter to continue to final example...")
        example_prediction()
        
    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("EXAMPLES COMPLETED")
    print("="*80)
    print("\nFor full training, run:")
    print("  python main.py --mode train --model all --epochs 50")
    print("\nTo launch the web app:")
    print("  streamlit run app.py")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
