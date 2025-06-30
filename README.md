# Semi-Supervised Learning with Organized Configuration

This repository contains a clean, organized implementation of semi-supervised learning (SSL) on CIFAR-10 using autoencoder pre-training and classifier fine-tuning.

## 🎯 Key Features

- **Class-based Configuration System**: Centralized, type-safe configuration management
- **Modular Architecture**: Clean separation of concerns across different modules
- **Reproducible Experiments**: Consistent seed setting and configuration management
- **Easy Experimentation**: Simple configuration changes for different experimental setups
- **Professional Code Structure**: Follows software engineering best practices

## 📁 Project Structure

```
├── config.py              # Configuration classes for all experiment settings
├── data_utils.py          # Dataset loading, preprocessing, and splitting utilities
├── models.py              # Model definitions (autoencoder, classifier)
├── trainers.py            # Training and evaluation logic
├── utils.py               # Utility functions (seed setting, visualization, etc.)
├── main.py                # Main script for running experiments
├── SSL_organized.ipynb    # Clean, organized Jupyter notebook
├── SSL.ipynb              # Original notebook (for reference)
└── README.md              # This file
```

## 🚀 Quick Start

### Option 1: Using the Organized Notebook
```bash
jupyter notebook SSL_organized.ipynb
```

### Option 2: Using the Main Script
```bash
python main.py
```

### Option 3: Programmatic Usage
```python
from config import get_full_experiment_config
from main import run_ssl_experiment

# Run full SSL experiment
config = get_full_experiment_config()
results = run_ssl_experiment(config)
```

## ⚙️ Configuration System

The configuration system is built around four main classes:

### 1. DataConfig
Manages dataset and data loading settings:
```python
@dataclass
class DataConfig:
    dataset_name: str = "CIFAR10"
    labeled_size: int = 5000
    validation_size: int = 5000
    batch_size: int = 128
    num_workers: int = 16
    # ... and more
```

### 2. ModelConfig
Manages model architecture settings:
```python
@dataclass
class ModelConfig:
    backbone: str = "resnet18"
    pretrained: bool = True
    num_classes: int = 10
    feature_dim: int = 512
    dropout_rate: float = 0.5
    # ... and more
```

### 3. TrainingConfig
Manages training process settings:
```python
@dataclass
class TrainingConfig:
    autoencoder_epochs: int = 100
    classifier_epochs: int = 50
    autoencoder_lr: float = 0.001
    classifier_lr: float = 0.001
    early_stopping: bool = True
    # ... and more
```

### 4. ExperimentConfig
Combines all configurations:
```python
@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    experiment_name: str = "ssl_cifar10_experiment"
    # ... and more
```

## 🔧 Predefined Configurations

The framework includes several predefined configurations:

```python
# Quick testing with smaller dataset
config = get_small_experiment_config()

# Full-scale SSL experiment
config = get_full_experiment_config()

# Baseline without pre-training
config = get_baseline_config()
```

## 🎛️ Custom Configuration Example

```python
from config import ExperimentConfig

# Create custom configuration
config = ExperimentConfig()
config.experiment_name = "my_custom_experiment"
config.data.labeled_size = 3000
config.data.batch_size = 64
config.model.backbone = "resnet34"
config.model.dropout_rate = 0.3
config.training.autoencoder_epochs = 50
config.training.classifier_epochs = 30

# Run experiment
results = run_ssl_experiment(config)
```

## 📊 Experiment Types

### 1. Full SSL Experiment
- Pre-trains autoencoder on unlabeled data
- Fine-tunes classifier on labeled data
- Includes feature visualization

### 2. Baseline Experiment
- Trains classifier directly on labeled data
- No autoencoder pre-training
- Useful for comparison

### 3. Comparison Study
- Runs both SSL and baseline approaches
- Compares performance improvements

## 🔍 Key Components

### Data Management
- **CIFAR10CSVDataset**: Custom dataset class for Kaggle CSV format
- **DataManager**: Handles dataset loading, splitting, and data loader creation
- **Stratified splitting**: Ensures balanced class distribution

### Model Architecture
- **ResNetAutoencoder**: ResNet-based autoencoder for self-supervised pre-training
- **ResNetClassifier**: ResNet-based classifier with optional pre-trained encoder
- **Flexible backbone**: Support for ResNet18/34/50

### Training Framework
- **AutoencoderTrainer**: Handles autoencoder pre-training phase
- **ClassifierTrainer**: Handles classifier fine-tuning phase
- **Early stopping**: Prevents overfitting
- **Learning rate scheduling**: Adaptive learning rate adjustment

### Utilities
- **Reproducibility**: Consistent seed setting across all libraries
- **Visualization**: t-SNE visualization of learned features
- **Checkpointing**: Automatic saving of best models
- **Logging**: Comprehensive experiment tracking

## 📈 Results and Visualization

The framework automatically generates:
- Training/validation accuracy and loss curves
- t-SNE visualization of learned features
- Comprehensive experiment summaries
- Model parameter counts

## 🔄 Migration from Original Code

The original `SSL.ipynb` has been refactored into this organized structure:

**Before (Original):**
- All code in one notebook
- Hardcoded parameters scattered throughout
- Duplicate code sections
- Difficult to modify and experiment

**After (Organized):**
- Clean modular structure
- Centralized configuration management
- No code duplication
- Easy to modify and extend

## 🛠️ Extending the Framework

### Adding New Models
```python
# In models.py
class NewBackbone(nn.Module):
    def __init__(self, config: ModelConfig):
        # Implementation
        pass

# In config.py
config.model.backbone = "new_backbone"
```

### Adding New Datasets
```python
# In data_utils.py
class NewDataset(Dataset):
    def __init__(self, config: DataConfig):
        # Implementation
        pass

# In config.py
config.data.dataset_name = "new_dataset"
```

### Adding New Training Strategies
```python
# In trainers.py
class NewTrainer:
    def __init__(self, model, config, device):
        # Implementation
        pass
```

## 📋 Requirements

- Python 3.7+
- PyTorch 1.9+
- torchvision
- scikit-learn
- pandas
- numpy
- matplotlib
- kagglehub

## 🤝 Benefits of This Organization

1. **Maintainability**: Clean, readable code structure
2. **Reproducibility**: Consistent configuration management
3. **Extensibility**: Easy to add new components
4. **Experimentation**: Simple parameter modifications
5. **Professionalism**: Industry-standard code organization
6. **Debugging**: Easier to isolate and fix issues
7. **Collaboration**: Clear module boundaries for team development

## 📝 Usage Examples

See `SSL_organized.ipynb` for detailed usage examples and `main.py` for programmatic usage patterns.

This organized structure transforms the original experimental code into a professional, maintainable framework suitable for research and production use.
