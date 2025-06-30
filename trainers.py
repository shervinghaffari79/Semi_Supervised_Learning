"""
Training and evaluation utilities for SSL experiments.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from typing import Tuple, Dict, Any, Optional

from config import TrainingConfig
from utils import EarlyStopping, save_checkpoint


class AutoencoderTrainer:
    """Trainer for autoencoder pre-training phase."""
    
    def __init__(
        self, 
        model: nn.Module, 
        config: TrainingConfig, 
        device: torch.device,
        checkpoint_dir: str = "./checkpoints"
    ):
        self.model = model
        self.config = config
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Setup training components
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=config.autoencoder_lr,
            weight_decay=config.autoencoder_weight_decay
        )
        
        # Setup scheduler
        if config.scheduler_type == "ReduceLROnPlateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                patience=config.scheduler_patience,
                factor=config.scheduler_factor,
                verbose=config.verbose
            )
        else:
            self.scheduler = None
        
        # Setup early stopping
        self.early_stopping = EarlyStopping(
            patience=config.autoencoder_patience,
            min_delta=config.min_delta,
            mode='min'
        ) if config.early_stopping else None
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': []
        }
    
    def train(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader
    ) -> Dict[str, Any]:
        """
        Train the autoencoder.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            Training history and results
        """
        print("=== Pre-training Autoencoder ===")
        
        best_val_loss = float('inf')
        best_epoch = 0
        
        for epoch in range(self.config.autoencoder_epochs):
            # Training phase
            train_loss = self._train_epoch(train_loader)
            
            # Validation phase
            val_loss = self._validate_epoch(val_loader)
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            # Check for best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                if self.config.save_best_model:
                    self._save_best_encoder()
            
            # Logging
            if epoch % self.config.log_interval == 0 or epoch == self.config.autoencoder_epochs - 1:
                print(f'Epoch [{epoch+1}/{self.config.autoencoder_epochs}], '
                      f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
            
            # Early stopping
            if self.early_stopping and self.early_stopping(val_loss):
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        if self.config.save_best_model:
            self._load_best_encoder()
        
        results = {
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'final_train_loss': self.history['train_loss'][-1],
            'final_val_loss': self.history['val_loss'][-1],
            'history': self.history
        }
        
        print(f"Autoencoder training completed. Best val loss: {best_val_loss:.6f} at epoch {best_epoch+1}")
        return results
    
    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for data, _ in train_loader:
            data = data.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, data)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def _validate_epoch(self, val_loader: DataLoader) -> float:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, data)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def _save_best_encoder(self) -> None:
        """Save the best encoder state."""
        encoder_path = os.path.join(self.checkpoint_dir, 'best_encoder.pth')
        torch.save(self.model.encoder.state_dict(), encoder_path)
    
    def _load_best_encoder(self) -> None:
        """Load the best encoder state."""
        encoder_path = os.path.join(self.checkpoint_dir, 'best_encoder.pth')
        if os.path.exists(encoder_path):
            self.model.encoder.load_state_dict(torch.load(encoder_path, map_location=self.device))


class ClassifierTrainer:
    """Trainer for classifier fine-tuning phase."""
    
    def __init__(
        self, 
        model: nn.Module, 
        config: TrainingConfig, 
        device: torch.device,
        checkpoint_dir: str = "./checkpoints"
    ):
        self.model = model
        self.config = config
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Setup training components
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=config.classifier_lr,
            weight_decay=config.classifier_weight_decay
        )
        
        # Setup scheduler
        if config.scheduler_type == "ReduceLROnPlateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                patience=config.scheduler_patience,
                factor=config.scheduler_factor,
                verbose=config.verbose
            )
        else:
            self.scheduler = None
        
        # Setup early stopping
        self.early_stopping = EarlyStopping(
            patience=config.classifier_patience,
            min_delta=config.min_delta,
            mode='max'  # For accuracy
        ) if config.early_stopping else None
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def train(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader
    ) -> Dict[str, Any]:
        """
        Train the classifier.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            Training history and results
        """
        print("=== Fine-tuning Classifier ===")
        
        best_val_acc = 0.0
        best_epoch = 0
        
        for epoch in range(self.config.classifier_epochs):
            # Unfreeze encoder if specified
            if (hasattr(self.model, 'unfreeze_encoder') and 
                epoch == self.config.unfreeze_after_epoch):
                print(f"Unfreezing encoder at epoch {epoch+1}")
                self.model.unfreeze_encoder()
                # Adjust learning rate for fine-tuning
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.config.fine_tune_lr
            
            # Training phase
            train_loss, train_acc = self._train_epoch(train_loader)
            
            # Validation phase
            val_loss, val_acc = self._validate_epoch(val_loader)
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Check for best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                if self.config.save_best_model:
                    self._save_best_classifier()
            
            # Logging
            if epoch % self.config.log_interval == 0 or epoch == self.config.classifier_epochs - 1:
                print(f'Epoch [{epoch+1}/{self.config.classifier_epochs}], '
                      f'Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
            
            # Early stopping
            if self.early_stopping and self.early_stopping(val_acc):
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        if self.config.save_best_model:
            self._load_best_classifier()
        
        results = {
            'best_val_acc': best_val_acc,
            'best_epoch': best_epoch,
            'final_train_acc': self.history['train_acc'][-1],
            'final_val_acc': self.history['val_acc'][-1],
            'history': self.history
        }
        
        print(f"Classifier training completed. Best val acc: {best_val_acc:.2f}% at epoch {best_epoch+1}")
        return results
    
    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for data, targets in train_loader:
            data, targets = data.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        accuracy = 100.0 * correct / total
        return total_loss / len(train_loader), accuracy
    
    def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        accuracy = 100.0 * correct / total
        return total_loss / len(val_loader), accuracy
    
    def _save_best_classifier(self) -> None:
        """Save the best classifier state."""
        classifier_path = os.path.join(self.checkpoint_dir, 'best_classifier.pth')
        torch.save(self.model.state_dict(), classifier_path)
    
    def _load_best_classifier(self) -> None:
        """Load the best classifier state."""
        classifier_path = os.path.join(self.checkpoint_dir, 'best_classifier.pth')
        if os.path.exists(classifier_path):
            self.model.load_state_dict(torch.load(classifier_path, map_location=self.device))


def evaluate_model(model: nn.Module, test_loader: DataLoader, device: torch.device) -> float:
    """
    Evaluate model on test set.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run evaluation on
        
    Returns:
        Test accuracy percentage
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100.0 * correct / total
    return accuracy
