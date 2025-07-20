"""
Configuration classes for SSL (Semi-Supervised Learning) experiments.
Provides centralized, class-based configuration management.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any
import torch
import torchvision.transforms as transforms


@dataclass
class DataConfig:
    """Configuration for dataset and data loading."""
    
    # Dataset settings
    dataset_name: str = "CIFAR10"
    dataset_path: str = "./data"
    use_kaggle_csv: bool = True
    kaggle_dataset_id: str = "fedesoriano/cifar10-python-in-csv"
    
    # Data splits
    labeled_size: int = 5000
    validation_size: int = 5000
    test_size: Optional[int] = None  # Use full test set
    
    # Data loading
    batch_size: int = 128
    num_workers: int = 16
    pin_memory: bool = True
    
    # Image preprocessing
    image_size: Tuple[int, int] = (32, 32)
    normalize_mean: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
    normalize_std: Tuple[float, float, float] = (0.2023, 0.1994, 0.2010)
    
    # Data augmentation for training
    augmentation_config: Dict[str, Any] = field(default_factory=lambda: {
        'rotation_degrees': 15,
        'horizontal_flip_prob': 0.5,
        'translate': (0.1, 0.1),
        'scale_range': (0.9, 1.0)
    })
    
    # Random seed
    random_seed: int = 42
    
    def get_train_transform(self) -> transforms.Compose:
        """Get training data transforms with augmentation."""
        return transforms.Compose([
            transforms.RandomRotation(degrees=self.augmentation_config['rotation_degrees']),
            transforms.RandomHorizontalFlip(p=self.augmentation_config['horizontal_flip_prob']),
            transforms.RandomAffine(
                degrees=0, 
                translate=self.augmentation_config['translate']
            ),
            transforms.RandomResizedCrop(
                size=self.image_size[0], 
                scale=self.augmentation_config['scale_range']
            ),
            transforms.ToTensor(),
            transforms.Normalize(self.normalize_mean, self.normalize_std)
        ])
    
    def get_test_transform(self) -> transforms.Compose:
        """Get test/validation data transforms without augmentation."""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.normalize_mean, self.normalize_std)
        ])


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    
    # Backbone settings
    backbone: str = "resnet18"
    pretrained: bool = True
    pretrained_weights: str = "IMAGENET1K_V1"
    
    # Architecture settings
    num_classes: int = 10
    feature_dim: int = 512
    
    # Autoencoder settings
    decoder_channels: Tuple[int, ...] = (512, 256, 128, 64, 32)
    use_batch_norm: bool = True
    activation: str = "relu"
    
    # Classifier settings
    classifier_hidden_dim: int = 256
    dropout_rate: float = 0.5
    use_batch_norm_classifier: bool = True
    
    # Transfer learning settings
    freeze_encoder_initially: bool = True
    unfreeze_after_epoch: int = 10


@dataclass
class TrainingConfig:
    """Configuration for training process."""
    
    # Autoencoder pre-training
    autoencoder_epochs: int = 100
    autoencoder_lr: float = 0.001
    autoencoder_weight_decay: float = 0.0
    autoencoder_patience: int = 20
    
    # Classifier fine-tuning
    classifier_epochs: int = 50
    classifier_lr: float = 0.001
    classifier_weight_decay: float = 1e-4
    classifier_patience: int = 15
    fine_tune_lr: float = 0.0005  # LR after unfreezing encoder
    
    # Scheduler settings
    scheduler_type: str = "ReduceLROnPlateau"
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    
    # Early stopping
    early_stopping: bool = True
    min_delta: float = 0.001
    
    # Checkpointing
    save_best_model: bool = True
    checkpoint_dir: str = "./checkpoints"
    
    # Logging
    log_interval: int = 10  # Log every N epochs
    verbose: bool = True


@dataclass
class ExperimentConfig:
    """Main experiment configuration combining all sub-configs."""
    
    # Sub-configurations
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Experiment metadata
    experiment_name: str = "ssl_cifar10_experiment"
    description: str = "Semi-supervised learning on CIFAR-10 with autoencoder pre-training"
    
    # Device settings
    device: str = "auto"  # "auto", "cuda", "cpu"
    mixed_precision: bool = False
    
    # Reproducibility
    deterministic: bool = True
    
    def get_device(self) -> torch.device:
        """Get the appropriate device for training."""
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        assert self.data.labeled_size > 0, "Labeled size must be positive"
        assert self.data.validation_size > 0, "Validation size must be positive"
        assert self.model.num_classes > 0, "Number of classes must be positive"
        assert self.training.autoencoder_epochs >= 0, "Autoencoder epochs must be non-negative"
        assert self.training.classifier_epochs > 0, "Classifier epochs must be positive"
        
        if self.model.unfreeze_after_epoch >= self.training.classifier_epochs:
            print("Warning: unfreeze_after_epoch is >= classifier_epochs")


# Predefined configurations for different experiment settings
def get_small_experiment_config() -> ExperimentConfig:
    """Configuration for quick testing with smaller dataset."""
    config = ExperimentConfig()
    config.data.labeled_size = 1000
    config.data.validation_size = 1000
    config.data.batch_size = 64
    config.training.autoencoder_epochs = 20
    config.training.classifier_epochs = 20
    config.experiment_name = "ssl_cifar10_small"
    return config


def get_full_experiment_config() -> ExperimentConfig:
    """Configuration for full-scale experiment."""
    config = ExperimentConfig()
    config.data.labeled_size = 5000
    config.data.validation_size = 5000
    config.data.batch_size = 128
    config.training.autoencoder_epochs = 100
    config.training.classifier_epochs = 50
    config.experiment_name = "ssl_cifar10_full"
    return config


def get_baseline_config() -> ExperimentConfig:
    """Configuration for baseline (no pre-training) experiment."""
    config = ExperimentConfig()
    config.model.pretrained = True
    config.model.freeze_encoder_initially = False
    config.model.unfreeze_after_epoch = 0
    config.training.autoencoder_epochs = 0  # Skip autoencoder training
    config.experiment_name = "ssl_cifar10_baseline"
    return config
