"""
Neural network models for cough detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ConvBlock(nn.Module):
    """Convolutional block with BatchNorm and ReLU."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        pool_size: int = 2
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(pool_size) if pool_size > 1 else nn.Identity()
        self.dropout = nn.Dropout2d(0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class CoughDetector(nn.Module):
    """
    CNN-based cough detection model.
    
    Architecture designed for mel spectrogram input:
    - Input: (batch, 1, n_mels, time_frames)
    - Multiple conv blocks with batch normalization
    - Global average pooling for variable-length input support
    - Fully connected layers for classification
    
    Design choices:
    - Small kernel sizes (3x3) to capture local patterns
    - Batch normalization for training stability
    - Dropout for regularization
    - Global average pooling to handle variable input sizes
    """
    
    def __init__(
        self,
        n_mels: int = 64,
        num_classes: int = 2,
        in_channels: int = 1,
        channels: Tuple[int, ...] = (32, 64, 128, 256),
        fc_hidden: int = 128,
        dropout: float = 0.5
    ):
        """
        Initialize the cough detector model.
        
        Args:
            n_mels: Number of mel filterbanks (input height)
            num_classes: Number of output classes (2 for binary cough/non-cough)
            in_channels: Number of input channels (1 for mel only, 90 for mel+mfcc)
            channels: Number of channels in each conv block
            fc_hidden: Hidden units in fully connected layer
            dropout: Dropout rate for FC layers
        """
        super().__init__()
        
        self.n_mels = n_mels
        self.num_classes = num_classes
        
        # Convolutional layers
        conv_layers = []
        curr_channels = in_channels
        for out_channels in channels:
            conv_layers.append(ConvBlock(curr_channels, out_channels))
            curr_channels = out_channels
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(channels[-1], fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, 1, n_mels, time_frames)
            
        Returns:
            Logits of shape (batch, num_classes)
        """
        # Convolutional layers
        x = self.conv_layers(x)
        
        # Global average pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc(x)
        
        return x
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make prediction with probabilities.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (predicted class indices, probabilities)
        """
        logits = self.forward(x)
        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        return preds, probs


class CoughDetectorSmall(nn.Module):
    """
    Lightweight cough detector for real-time inference.
    
    Optimized for low latency:
    - Fewer parameters
    - Depthwise separable convolutions
    - Minimal fully connected layers
    """
    
    def __init__(
        self,
        n_mels: int = 64,
        num_classes: int = 2,
        in_channels: int = 1
    ):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 2 - depthwise separable
            nn.Conv2d(16, 16, 3, padding=1, groups=16),  # Depthwise
            nn.Conv2d(16, 32, 1),  # Pointwise
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 3 - depthwise separable
            nn.Conv2d(32, 32, 3, padding=1, groups=32),
            nn.Conv2d(32, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 4 - depthwise separable
            nn.Conv2d(64, 64, 3, padding=1, groups=64),
            nn.Conv2d(64, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(x)
        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        return preds, probs


class CoughDetectorResidual(nn.Module):
    """
    Residual network for cough detection.
    Better gradient flow for deeper networks.
    """
    
    def __init__(
        self,
        n_mels: int = 64,
        num_classes: int = 2,
        in_channels: int = 1,
        channels: Tuple[int, ...] = (32, 64, 128),
        dropout: float = 0.5
    ):
        super().__init__()
        
        # Initial conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], 7, stride=2, padding=3),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Residual blocks
        self.res_blocks = nn.ModuleList()
        in_ch = channels[0]
        for out_ch in channels[1:]:
            self.res_blocks.append(self._make_res_block(in_ch, out_ch))
            in_ch = out_ch
        
        # Classification head
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1], num_classes)
        )
    
    def _make_res_block(self, in_ch: int, out_ch: int) -> nn.Module:
        """Create a residual block with skip connection."""
        return ResidualBlock(in_ch, out_ch)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        for block in self.res_blocks:
            x = block(x)
        x = self.global_pool(x)
        x = self.fc(x)
        return x
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(x)
        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        return preds, probs


class ResidualBlock(nn.Module):
    """Basic residual block with skip connection."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection with projection if dimensions change
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=stride),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels or stride != 1 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = F.relu(out)
        
        return out


def create_model(model_type: str = "standard", **kwargs) -> nn.Module:
    """
    Factory function to create a cough detection model.
    
    Args:
        model_type: One of "standard", "small", or "residual"
        **kwargs: Additional model arguments
        
    Returns:
        Model instance
    """
    models = {
        "standard": CoughDetector,
        "small": CoughDetectorSmall,
        "residual": CoughDetectorResidual
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(models.keys())}")
    
    return models[model_type](**kwargs)


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def export_to_torchscript(model: nn.Module, example_input: torch.Tensor, path: str):
    """Export model to TorchScript for optimized inference."""
    model.eval()
    traced = torch.jit.trace(model, example_input)
    traced.save(path)
    return traced
