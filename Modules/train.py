"""
Training pipeline for plant disease classification models.
Includes training loop, validation, early stopping, and checkpointing.
"""
import os
import time
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import config
from model import create_model


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.
    
    Args:
        patience: Number of epochs to wait before stopping
        min_delta: Minimum change to qualify as improvement
        verbose: Whether to print early stopping messages
    """
    
    def __init__(self, patience: int = 5, min_delta: float = 0.0, verbose: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, val_loss: float, epoch: int) -> bool:
        """
        Check if training should be stopped.
        
        Args:
            val_loss: Current validation loss
            epoch: Current epoch number
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"Early stopping triggered! Best epoch: {self.best_epoch}")
                return True
        else:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
        
        return False


class Trainer:
    """
    Trainer class for managing the training pipeline.
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        model_name: Name for saving checkpoints
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler,
        device: torch.device,
        model_name: str = "model"
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.model_name = model_name
        
        # Initialize TensorBoard writer
        log_dir = config.TENSORBOARD_LOG_DIR / model_name
        self.writer = SummaryWriter(log_dir=str(log_dir))
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_epoch = 0
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)
        
        for images, labels in progress_bar:
            # Move data to device
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': 100. * correct / total
            })
        
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self) -> Tuple[float, float]:
        """
        Validate the model.
        
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc="Validation", leave=False)
            
            for images, labels in progress_bar:
                # Move data to device
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': loss.item(),
                    'acc': 100. * correct / total
                })
        
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        checkpoint_path = config.CHECKPOINT_DIR / f"{self.model_name}_checkpoint.pth"
        best_path = config.CHECKPOINT_DIR / f"{self.model_name}_best.pth"
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'history': self.history
        }
        
        # Save regular checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")
    
    def train(
        self,
        num_epochs: int = config.NUM_EPOCHS,
        early_stopping_patience: int = config.EARLY_STOPPING_PATIENCE
    ) -> Dict:
        """
        Complete training loop.
        
        Args:
            num_epochs: Number of epochs to train
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Training history dictionary
        """
        early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True)
        
        print(f"\n{'='*60}")
        print(f"Starting training: {self.model_name}")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Learning rate: {config.LEARNING_RATE}")
        print(f"Batch size: {config.BATCH_SIZE}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update learning rate
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)
            
            # Log to TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            self.writer.add_scalar('Learning_rate', current_lr, epoch)
            
            # Check if best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                self.best_epoch = epoch
            
            # Save checkpoint
            if config.SAVE_BEST_ONLY:
                if is_best:
                    self.save_checkpoint(epoch, is_best=True)
            else:
                self.save_checkpoint(epoch, is_best=is_best)
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start
            print(f"Epoch [{epoch}/{num_epochs}] - {epoch_time:.2f}s")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print(f"  LR: {current_lr:.6f}")
            if is_best:
                print("  *** New best model! ***")
            print()
            
            # Early stopping check
            if early_stopping(val_loss, epoch):
                print(f"Early stopping triggered at epoch {epoch}")
                break
        
        total_time = time.time() - start_time
        
        # Print training summary
        print(f"\n{'='*60}")
        print("Training completed!")
        print(f"{'='*60}")
        print(f"Total time: {total_time/60:.2f} minutes")
        print(f"Best epoch: {self.best_epoch}")
        print(f"Best val loss: {self.best_val_loss:.4f}")
        print(f"Best val accuracy: {self.best_val_acc:.2f}%")
        print(f"{'='*60}\n")
        
        self.writer.close()
        
        return self.history


def train_model(
    model_type: str = 'cnn',
    train_loader = None,
    val_loader = None,
    num_epochs: int = config.NUM_EPOCHS,
    learning_rate: float = config.LEARNING_RATE,
    device: torch.device = config.DEVICE
) -> Tuple[nn.Module, Dict]:
    """
    High-level function to train a model.
    
    Args:
        model_type: Type of model ('cnn' or 'resnet')
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        num_epochs: Number of epochs to train
        learning_rate: Learning rate
        device: Device to train on
        
    Returns:
        Tuple of (trained model, training history)
    """
    # Create model
    model = create_model(model_type, device)
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Define optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config.LR_SCHEDULER_FACTOR,
        patience=config.LR_SCHEDULER_PATIENCE,
        min_lr=config.LR_SCHEDULER_MIN_LR
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        model_name=f"{model_type}_model"
    )
    
    # Train
    history = trainer.train(num_epochs=num_epochs)
    
    return model, history


if __name__ == "__main__":
    # Test training pipeline
    print("Testing training pipeline...")
    
    try:
        from data import create_dataloaders
        
        # Create dataloaders
        print("Creating dataloaders...")
        train_loader, val_loader, test_loader = create_dataloaders()
        
        # Train Custom CNN (just 2 epochs for testing)
        print("\nTesting Custom CNN training...")
        model, history = train_model(
            model_type='cnn',
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=2,
            device=config.DEVICE
        )
        
        print("\nTraining pipeline test completed successfully!")
        
    except Exception as e:
        print(f"Error during training test: {e}")
        import traceback
        traceback.print_exc()
