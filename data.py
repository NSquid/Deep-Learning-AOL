import os
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import config

# Custom Dataset for Plant Disease Classification
class PlantDiseaseDataset(Dataset): 
    def __init__(
        self,
        root_dir: Path,
        transform: Optional[transforms.Compose] = None,
        class_names: List[str] = None
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.class_names = class_names or config.CLASS_NAMES
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}
        
        # Load all image paths and labels
        self.samples = self._load_samples()
        
        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {root_dir}")
    
    def _load_samples(self) -> List[Tuple[Path, int]]:
        samples = []
        
        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            
            if not class_dir.exists():
                print(f"Warning: Directory {class_dir} does not exist")
                continue
            
            class_idx = self.class_to_idx[class_name]
            
            # Get all image files (jpg, jpeg, png)
            for img_path in class_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    samples.append((img_path, class_idx))
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    # Get a sample from the dataset.
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        
        try:
            # Load image
            image = Image.open(img_path).convert('RGB')
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            
            return image, label
        
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image if loading fails
            if self.transform:
                blank_image = self.transform(Image.new('RGB', (config.IMAGE_SIZE, config.IMAGE_SIZE)))
            else:
                blank_image = torch.zeros(3, config.IMAGE_SIZE, config.IMAGE_SIZE)
            return blank_image, label
    
    # Calculate the distribution of samples across classes.
    def get_class_distribution(self) -> Dict[str, int]:
        distribution = {class_name: 0 for class_name in self.class_names}
        
        for _, label in self.samples:
            class_name = self.class_names[label]
            distribution[class_name] += 1
        
        return distribution

# Get training data transformations with augmentation.
def get_train_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.RandomRotation(config.ROTATION_DEGREES),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=config.COLOR_JITTER_BRIGHTNESS,
            contrast=config.COLOR_JITTER_CONTRAST,
            saturation=config.COLOR_JITTER_SATURATION,
            hue=config.COLOR_JITTER_HUE
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
    ])

# Get validation/test data transformations without augmentation.
def get_val_test_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
    ])


def create_dataloaders(
    train_dir: Path = config.TRAIN_DIR,
    val_dir: Path = config.VAL_DIR,
    test_dir: Path = config.TEST_DIR,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # Create datasets
    train_dataset = PlantDiseaseDataset(
        root_dir=train_dir,
        transform=get_train_transforms()
    )
    
    val_dataset = PlantDiseaseDataset(
        root_dir=val_dir,
        transform=get_val_test_transforms()
    )
    
    test_dataset = PlantDiseaseDataset(
        root_dir=test_dir,
        transform=get_val_test_transforms()
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=config.SHUFFLE_TRAIN,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=config.SHUFFLE_VAL,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=config.SHUFFLE_TEST,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader, test_loader

def get_dataset_statistics(
    train_dir: Path = config.TRAIN_DIR,
    val_dir: Path = config.VAL_DIR,
    test_dir: Path = config.TEST_DIR
) -> Dict[str, any]:
    
    # Create datasets without transforms to get raw counts
    train_dataset = PlantDiseaseDataset(train_dir, transform=None)
    val_dataset = PlantDiseaseDataset(val_dir, transform=None)
    test_dataset = PlantDiseaseDataset(test_dir, transform=None)
    
    stats = {
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'test_size': len(test_dataset),
        'total_size': len(train_dataset) + len(val_dataset) + len(test_dataset),
        'num_classes': config.NUM_CLASSES,
        'class_names': config.CLASS_NAMES,
        'train_distribution': train_dataset.get_class_distribution(),
        'val_distribution': val_dataset.get_class_distribution(),
        'test_distribution': test_dataset.get_class_distribution()
    }
    
    return stats

# Print comprehensive dataset information.
def print_dataset_info():
    try:
        stats = get_dataset_statistics()
        
        print("=" * 60)
        print("DATASET INFORMATION")
        print("=" * 60)
        print(f"\nTotal samples: {stats['total_size']}")
        print(f"Training samples: {stats['train_size']}")
        print(f"Validation samples: {stats['val_size']}")
        print(f"Testing samples: {stats['test_size']}")
        print(f"\nNumber of classes: {stats['num_classes']}")
        print(f"Class names: {', '.join(stats['class_names'])}")
        
        print("\nClass Distribution:")
        print("-" * 60)
        print(f"{'Class':<20} {'Train':<10} {'Val':<10} {'Test':<10} {'Total':<10}")
        print("-" * 60)
        
        for class_name in stats['class_names']:
            train_count = stats['train_distribution'][class_name]
            val_count = stats['val_distribution'][class_name]
            test_count = stats['test_distribution'][class_name]
            total = train_count + val_count + test_count
            print(f"{class_name:<20} {train_count:<10} {val_count:<10} {test_count:<10} {total:<10}")
        
        print("-" * 60)
        print("=" * 60)
        
    except Exception as e:
        print(f"Error getting dataset statistics: {e}")


if __name__ == "__main__":
    print("Testing data pipeline...")
    print_dataset_info()
    
    print("\nCreating DataLoaders...")
    try:
        train_loader, val_loader, test_loader = create_dataloaders()
        
        print(f"Train batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")

        print("\nLoading a sample batch...")
        images, labels = next(iter(train_loader))
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
        print("\nData pipeline test completed successfully!")
        
    except Exception as e:
        print(f"Error creating DataLoaders: {e}")
