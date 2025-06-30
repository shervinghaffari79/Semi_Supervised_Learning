"""
Utility functions for SSL experiments.
"""

import random
import numpy as np
import torch
import os
from typing import Optional
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_device(device_str: str = "auto") -> torch.device:
    """Setup and return the appropriate device for training."""
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return device


def create_directories(paths: list) -> None:
    """Create directories if they don't exist."""
    for path in paths:
        os.makedirs(path, exist_ok=True)


def count_parameters(model: torch.nn.Module) -> tuple:
    """Count total and trainable parameters in a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def visualize_features_tsne(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_samples: int = 5000,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize learned features using t-SNE.
    
    Args:
        model: Trained model with encoder
        data_loader: DataLoader for the data to visualize
        device: Device to run inference on
        max_samples: Maximum number of samples to use for t-SNE
        save_path: Path to save the plot (optional)
    """
    model.eval()
    features = []
    labels = []
    
    with torch.no_grad():
        for i, (imgs, targets) in enumerate(data_loader):
            if len(features) * data_loader.batch_size >= max_samples:
                break
                
            imgs = imgs.to(device)
            
            # Extract features from encoder
            if hasattr(model, 'encoder'):
                encoded = model.encoder(imgs)
            else:
                # Assume model is the encoder itself
                encoded = model(imgs)
            
            # Flatten features
            encoded = encoded.view(encoded.size(0), -1)
            features.append(encoded.cpu().numpy())
            labels.append(targets.numpy())
    
    if not features:
        print("No features extracted for visualization")
        return
    
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    # Limit to max_samples
    if len(features) > max_samples:
        indices = np.random.choice(len(features), max_samples, replace=False)
        features = features[indices]
        labels = labels[indices]
    
    print(f"Running t-SNE on {len(features)} samples...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_results = tsne.fit_transform(features)
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        tsne_results[:, 0], 
        tsne_results[:, 1], 
        c=labels, 
        cmap='tab10', 
        s=10,
        alpha=0.7
    )
    plt.colorbar(scatter, ticks=range(10), label='CIFAR-10 Classes')
    plt.title("t-SNE Visualization of Learned Features")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def print_experiment_summary(config, results: dict) -> None:
    """Print a summary of the experiment configuration and results."""
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    print(f"Experiment: {config.experiment_name}")
    print(f"Description: {config.description}")
    print()
    
    print("DATA CONFIGURATION:")
    print(f"  Dataset: {config.data.dataset_name}")
    print(f"  Labeled samples: {config.data.labeled_size:,}")
    print(f"  Validation samples: {config.data.validation_size:,}")
    print(f"  Batch size: {config.data.batch_size}")
    print()
    
    print("MODEL CONFIGURATION:")
    print(f"  Backbone: {config.model.backbone}")
    print(f"  Pretrained: {config.model.pretrained}")
    print(f"  Number of classes: {config.model.num_classes}")
    print()
    
    print("TRAINING CONFIGURATION:")
    print(f"  Autoencoder epochs: {config.training.autoencoder_epochs}")
    print(f"  Classifier epochs: {config.training.classifier_epochs}")
    print(f"  Autoencoder LR: {config.training.autoencoder_lr}")
    print(f"  Classifier LR: {config.training.classifier_lr}")
    print()
    
    if results:
        print("RESULTS:")
        for key, value in results.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}%")
            else:
                print(f"  {key}: {value}")
    
    print("="*60)


class EarlyStopping:
    """Early stopping utility class."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = 'min'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """
        Check if training should be stopped.
        
        Args:
            score: Current metric value (loss or accuracy)
            
        Returns:
            True if training should be stopped
        """
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
        return self.early_stop
    
    def _is_improvement(self, score: float) -> bool:
        """Check if the current score is an improvement."""
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        else:  # mode == 'max'
            return score > self.best_score + self.min_delta


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    filepath: str
) -> None:
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    filepath: str,
    device: torch.device
) -> tuple:
    """Load model checkpoint."""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss
