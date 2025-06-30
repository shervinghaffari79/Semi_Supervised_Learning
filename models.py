"""
Model definitions for SSL experiments.
Contains autoencoder and classifier architectures.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Tuple

from config import ModelConfig


class ResNetAutoencoder(nn.Module):
    """ResNet-based autoencoder for self-supervised pre-training."""
    
    def __init__(self, config: ModelConfig):
        super(ResNetAutoencoder, self).__init__()
        self.config = config
        
        # Encoder: ResNet backbone without classifier
        if config.pretrained:
            if config.backbone == "resnet18":
                resnet = models.resnet18(weights=config.pretrained_weights)
            elif config.backbone == "resnet34":
                resnet = models.resnet34(weights=config.pretrained_weights)
            elif config.backbone == "resnet50":
                resnet = models.resnet50(weights=config.pretrained_weights)
            else:
                raise ValueError(f"Unsupported backbone: {config.backbone}")
        else:
            if config.backbone == "resnet18":
                resnet = models.resnet18(weights=None)
            elif config.backbone == "resnet34":
                resnet = models.resnet34(weights=None)
            elif config.backbone == "resnet50":
                resnet = models.resnet50(weights=None)
            else:
                raise ValueError(f"Unsupported backbone: {config.backbone}")
        
        # Remove the final classification layer
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        
        # Build decoder
        self.decoder = self._build_decoder()
    
    def _build_decoder(self) -> nn.Module:
        """Build the decoder network."""
        layers = []
        in_channels = self.config.feature_dim
        
        # Upsampling layers
        for out_channels in self.config.decoder_channels:
            layers.extend([
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0),
                nn.BatchNorm2d(out_channels) if self.config.use_batch_norm else nn.Identity(),
                self._get_activation()
            ])
            in_channels = out_channels
        
        # Final layer to reconstruct RGB image
        layers.extend([
            nn.Conv2d(in_channels, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # Output in [-1, 1] range
        ])
        
        return nn.Sequential(*layers)
    
    def _get_activation(self) -> nn.Module:
        """Get activation function based on config."""
        if self.config.activation.lower() == "relu":
            return nn.ReLU(inplace=True)
        elif self.config.activation.lower() == "leaky_relu":
            return nn.LeakyReLU(0.2, inplace=True)
        elif self.config.activation.lower() == "gelu":
            return nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {self.config.activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through autoencoder."""
        # Encode
        encoded = self.encoder(x)
        encoded = encoded.view(encoded.size(0), self.config.feature_dim, 1, 1)
        
        # Decode
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to feature representation."""
        with torch.no_grad():
            encoded = self.encoder(x)
            return encoded.view(encoded.size(0), -1)


class ResNetClassifier(nn.Module):
    """ResNet-based classifier for supervised fine-tuning."""
    
    def __init__(self, config: ModelConfig, pretrained_encoder: Optional[nn.Module] = None):
        super(ResNetClassifier, self).__init__()
        self.config = config
        
        if pretrained_encoder is not None:
            # Use pre-trained encoder
            self.encoder = pretrained_encoder
            # Freeze encoder initially if specified
            if config.freeze_encoder_initially:
                for param in self.encoder.parameters():
                    param.requires_grad = False
        else:
            # Create fresh encoder
            if config.pretrained:
                if config.backbone == "resnet18":
                    resnet = models.resnet18(weights=config.pretrained_weights)
                elif config.backbone == "resnet34":
                    resnet = models.resnet34(weights=config.pretrained_weights)
                elif config.backbone == "resnet50":
                    resnet = models.resnet50(weights=config.pretrained_weights)
                else:
                    raise ValueError(f"Unsupported backbone: {config.backbone}")
            else:
                if config.backbone == "resnet18":
                    resnet = models.resnet18(weights=None)
                elif config.backbone == "resnet34":
                    resnet = models.resnet34(weights=None)
                elif config.backbone == "resnet50":
                    resnet = models.resnet50(weights=None)
                else:
                    raise ValueError(f"Unsupported backbone: {config.backbone}")
            
            self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        
        # Build classifier head
        self.classifier = self._build_classifier()
    
    def _build_classifier(self) -> nn.Module:
        """Build the classifier head."""
        layers = []
        
        # Dropout
        if self.config.dropout_rate > 0:
            layers.append(nn.Dropout(self.config.dropout_rate))
        
        # Hidden layer (optional)
        if self.config.classifier_hidden_dim > 0:
            layers.extend([
                nn.Linear(self.config.feature_dim, self.config.classifier_hidden_dim),
                nn.BatchNorm1d(self.config.classifier_hidden_dim) if self.config.use_batch_norm_classifier else nn.Identity(),
                self._get_activation(),
                nn.Dropout(self.config.dropout_rate) if self.config.dropout_rate > 0 else nn.Identity()
            ])
            in_features = self.config.classifier_hidden_dim
        else:
            in_features = self.config.feature_dim
        
        # Output layer
        layers.append(nn.Linear(in_features, self.config.num_classes))
        
        return nn.Sequential(*layers)
    
    def _get_activation(self) -> nn.Module:
        """Get activation function based on config."""
        if self.config.activation.lower() == "relu":
            return nn.ReLU(inplace=True)
        elif self.config.activation.lower() == "leaky_relu":
            return nn.LeakyReLU(0.2, inplace=True)
        elif self.config.activation.lower() == "gelu":
            return nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {self.config.activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through classifier."""
        # Extract features
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        
        # Classify
        output = self.classifier(features)
        return output
    
    def unfreeze_encoder(self) -> None:
        """Unfreeze encoder parameters for fine-tuning."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        print("Encoder unfrozen for fine-tuning")
    
    def freeze_encoder(self) -> None:
        """Freeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("Encoder frozen")
    
    def get_feature_extractor(self) -> nn.Module:
        """Get the encoder part for feature extraction."""
        return self.encoder


def create_autoencoder(config: ModelConfig) -> ResNetAutoencoder:
    """Factory function to create autoencoder model."""
    return ResNetAutoencoder(config)


def create_classifier(
    config: ModelConfig, 
    pretrained_encoder: Optional[nn.Module] = None
) -> ResNetClassifier:
    """Factory function to create classifier model."""
    return ResNetClassifier(config, pretrained_encoder)


def load_pretrained_encoder(
    config: ModelConfig, 
    checkpoint_path: str, 
    device: torch.device
) -> nn.Module:
    """
    Load a pre-trained encoder from checkpoint.
    
    Args:
        config: Model configuration
        checkpoint_path: Path to the encoder checkpoint
        device: Device to load the model on
        
    Returns:
        Loaded encoder module
    """
    # Create a dummy autoencoder to get the encoder structure
    autoencoder = create_autoencoder(config)
    
    # Load the encoder state dict
    encoder_state_dict = torch.load(checkpoint_path, map_location=device)
    autoencoder.encoder.load_state_dict(encoder_state_dict)
    
    return autoencoder.encoder


def count_model_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def print_model_summary(model: nn.Module, model_name: str = "Model") -> None:
    """Print a summary of the model architecture and parameters."""
    total_params, trainable_params = count_model_parameters(model)
    
    print(f"\n{model_name} Summary:")
    print("-" * 40)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print("-" * 40)
    
    # Print model structure
    print(f"\n{model_name} Architecture:")
    print(model)
