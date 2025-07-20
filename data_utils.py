"""
Data utilities for SSL experiments.
Handles dataset loading, preprocessing, and splitting.
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets
from sklearn.model_selection import train_test_split
from PIL import Image
import kagglehub
from typing import Tuple, Optional
import copy

from config import DataConfig


class CIFAR10CSVDataset(Dataset):
    """Custom Dataset for CIFAR-10 CSV format from Kaggle."""
    
    def __init__(self, csv_file: str, transform=None):
        """
        Args:
            csv_file: Path to the CSV file with CIFAR-10 data
            transform: Optional transform to be applied on samples
        """
        self.csv_file = csv_file
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform
        
        # Separate features and labels
        if 'label' in self.data_frame.columns:
            self.labels = self.data_frame['label'].values
            self.pixel_data = self.data_frame.drop('label', axis=1).values
        else:
            # Assume last column is the label
            self.labels = self.data_frame.iloc[:, -1].values
            self.pixel_data = self.data_frame.iloc[:, :-1].values
        
        # Add targets attribute for compatibility with torchvision datasets
        self.targets = self.labels.tolist()
    
    def __len__(self) -> int:
        return len(self.data_frame)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        pixel_values = self.pixel_data[idx]
        
        # Reshape from flat array to 32x32x3 image
        image = pixel_values.reshape(3, 32, 32).transpose(1, 2, 0)
        image = Image.fromarray(image.astype(np.uint8))
        
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class DataManager:
    """Manages data loading and preprocessing for SSL experiments."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.train_dataset = None
        self.test_dataset = None
        self.labeled_dataset = None
        self.unlabeled_dataset = None
        self.val_dataset = None
        
    def setup_datasets(self) -> None:
        """Setup all datasets based on configuration."""
        if self.config.use_kaggle_csv:
            self._setup_kaggle_datasets()
        else:
            self._setup_torchvision_datasets()
        
        self._create_splits()
    
    def _setup_kaggle_datasets(self) -> None:
        """Setup datasets from Kaggle CSV format."""
        print("Downloading CIFAR-10 dataset from Kaggle...")
        dataset_path = kagglehub.dataset_download(self.config.kaggle_dataset_id)
        
        train_csv_path = os.path.join(dataset_path, 'train.csv')
        test_csv_path = os.path.join(dataset_path, 'test.csv')
        
        print("Files in dataset directory:", os.listdir(dataset_path))
        
        self.train_dataset = CIFAR10CSVDataset(
            train_csv_path, 
            transform=self.config.get_train_transform()
        )
        self.test_dataset = CIFAR10CSVDataset(
            test_csv_path, 
            transform=self.config.get_test_transform()
        )
        
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Test samples: {len(self.test_dataset)}")
    
    def _setup_torchvision_datasets(self) -> None:
        """Setup datasets using torchvision CIFAR-10."""
        self.train_dataset = datasets.CIFAR10(
            root=self.config.dataset_path,
            train=True,
            download=True,
            transform=self.config.get_train_transform()
        )
        self.test_dataset = datasets.CIFAR10(
            root=self.config.dataset_path,
            train=False,
            download=True,
            transform=self.config.get_test_transform()
        )
        
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Test samples: {len(self.test_dataset)}")
    
    def _create_splits(self) -> None:
        """Create labeled/unlabeled/validation splits."""
        labels = np.array(self.train_dataset.targets)
        
        # Split training data: train + validation
        train_indices, val_indices = train_test_split(
            np.arange(len(self.train_dataset)),
            test_size=self.config.validation_size,
            stratify=labels,
            random_state=self.config.random_seed
        )
        
        train_labels = labels[train_indices]
        
        # From training indices: labeled + unlabeled
        labeled_indices, unlabeled_indices = train_test_split(
            train_indices,
            train_size=self.config.labeled_size,
            stratify=train_labels,
            random_state=self.config.random_seed
        )
        
        # Create subset datasets
        self.labeled_dataset = Subset(self.train_dataset, labeled_indices)
        self.unlabeled_dataset = Subset(self.train_dataset, unlabeled_indices)
        self.val_dataset = Subset(self.train_dataset, val_indices)
        
        # Apply test transform to validation dataset without affecting training data
        if hasattr(self.val_dataset.dataset, 'transform'):
            val_dataset_copy = copy.deepcopy(self.train_dataset)
            val_dataset_copy.transform = self.config.get_test_transform()
            self.val_dataset = Subset(val_dataset_copy, val_indices)
        
        print(f"Data splits created:")
        print(f"  Labeled: {len(self.labeled_dataset)}")
        print(f"  Unlabeled: {len(self.unlabeled_dataset)}")
        print(f"  Validation: {len(self.val_dataset)}")
        print(f"  Test: {len(self.test_dataset)}")
    
    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
        """
        Get data loaders for all splits.
        
        Returns:
            Tuple of (labeled_loader, unlabeled_loader, val_loader, test_loader)
        """
        if self.labeled_dataset is None:
            raise ValueError("Datasets not setup. Call setup_datasets() first.")
        
        labeled_loader = DataLoader(
            self.labeled_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        unlabeled_loader = DataLoader(
            self.unlabeled_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        return labeled_loader, unlabeled_loader, val_loader, test_loader
    
    def get_class_distribution(self, dataset_type: str = "labeled") -> dict:
        """
        Get class distribution for a specific dataset split.
        
        Args:
            dataset_type: One of "labeled", "unlabeled", "validation", "test"
            
        Returns:
            Dictionary with class counts and proportions
        """
        if dataset_type == "labeled":
            dataset = self.labeled_dataset
        elif dataset_type == "unlabeled":
            dataset = self.unlabeled_dataset
        elif dataset_type == "validation":
            dataset = self.val_dataset
        elif dataset_type == "test":
            dataset = self.test_dataset
        else:
            raise ValueError(f"Unknown dataset_type: {dataset_type}")
        
        if isinstance(dataset, Subset):
            labels = [dataset.dataset.targets[i] for i in dataset.indices]
        else:
            labels = dataset.targets
        
        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)
        
        distribution = {
            'counts': dict(zip(unique.tolist(), counts.tolist())),
            'proportions': dict(zip(unique.tolist(), (counts / total).tolist())),
            'total': total
        }
        
        return distribution
    
    def print_data_summary(self) -> None:
        """Print a summary of all data splits and their distributions."""
        print("\n" + "="*50)
        print("DATA SUMMARY")
        print("="*50)
        
        for split_name in ["labeled", "unlabeled", "validation", "test"]:
            dist = self.get_class_distribution(split_name)
            print(f"\n{split_name.upper()} SET:")
            print(f"  Total samples: {dist['total']}")
            print(f"  Class distribution:")
            for class_id, count in dist['counts'].items():
                prop = dist['proportions'][class_id]
                print(f"    Class {class_id}: {count} ({prop:.1%})")
        
        print("="*50)
