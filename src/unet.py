"""
Standard UNet architecture for cell segmentation baseline.

This module implements a clean UNet baseline that produces a continuous 
segmentation field suitable for later PDE-constrained optimization.

The network:
- Takes single-channel 2D images as input
- Outputs single-channel probability maps u(x) ∈ (0,1)
- Uses standard encoder-decoder structure with skip connections
- No PDE losses, curvature terms, or post-processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Double convolution block: Conv -> Activation -> Dropout -> Conv -> Activation"""
    
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0, activation: str = 'relu'):
        super(DoubleConv, self).__init__()
        
        # Create activation function
        activation_layer = self._create_activation(activation)
        
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            activation_layer
        ]
        
        # Add dropout after first conv if specified
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        
        layers.extend([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            activation_layer
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def _create_activation(self, activation: str) -> nn.Module:
        """Create activation function based on string name."""
        activation_lower = activation.lower()
        
        if activation_lower == 'relu':
            return nn.ReLU(inplace=True)
        elif activation_lower == 'leaky_relu' or activation_lower == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01, inplace=True)
        elif activation_lower == 'elu':
            return nn.ELU(alpha=1.0, inplace=True)
        elif activation_lower == 'gelu':
            return nn.GELU()
        elif activation_lower == 'swish' or activation_lower == 'silu':
            return nn.SiLU()  # SiLU is the same as Swish
        elif activation_lower == 'mish':
            # Mish is not in standard PyTorch, we'll use a custom implementation
            return Mish()
        elif activation_lower == 'prelu':
            return nn.PReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}. Must be one of: relu, leaky_relu, elu, gelu, swish/silu, mish, prelu")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Mish(nn.Module):
    """Mish activation function: x * tanh(softplus(x))"""
    def __init__(self):
        super(Mish, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))


class UNet(nn.Module):
    """
    Standard UNet architecture for semantic segmentation.
    
    Architecture:
    - Encoder: 4 levels with downsampling (max pooling)
    - Bottleneck: Context encoding at lowest resolution
    - Decoder: 4 levels with upsampling and skip connections
    - Output: Single channel with sigmoid activation
    
    Channel progression:
    - Level 1: 64 channels
    - Level 2: 128 channels
    - Level 3: 256 channels
    - Level 4: 512 channels
    - Bottleneck: 512 channels
    
    Args:
        in_channels: Number of input channels (default: 1 for grayscale)
        out_channels: Number of output channels (default: 1 for probability map)
        base_channels: Base number of channels (default: 64)
        dropout: Dropout probability for regularization (default: 0.2)
                Applied in deeper layers (enc3, enc4, bottleneck, dec4) with
                reduced rates in intermediate layers (enc2, dec3, dec2)
        output_activation: Activation function for output layer ('sigmoid' or 'tanh', default: 'sigmoid')
        intermediate_activation: Activation function for intermediate layers 
                ('relu', 'leaky_relu', 'elu', 'gelu', 'swish/silu', 'mish', 'prelu', default: 'relu')
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        dropout: float = 0.2,
        output_activation: str = 'sigmoid',
        intermediate_activation: str = 'relu'
    ):
        super(UNet, self).__init__()
        
        # Encoder (downsampling path) - dropout in deeper layers
        self.enc1 = DoubleConv(in_channels, base_channels, dropout=0.0, activation=intermediate_activation)
        self.enc2 = DoubleConv(base_channels, base_channels * 2, dropout=dropout * 0.5, activation=intermediate_activation)
        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4, dropout=dropout, activation=intermediate_activation)
        self.enc4 = DoubleConv(base_channels * 4, base_channels * 8, dropout=dropout, activation=intermediate_activation)
        
        # Downsampling via max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck (lowest resolution) - higher dropout
        self.bottleneck = DoubleConv(base_channels * 8, base_channels * 8, dropout=dropout, activation=intermediate_activation)
        
        # Decoder (upsampling path) - dropout in deeper layers
        self.up4 = nn.ConvTranspose2d(
            base_channels * 8, base_channels * 8, 
            kernel_size=2, stride=2
        )
        self.dec4 = DoubleConv(base_channels * 16, base_channels * 8, dropout=dropout, activation=intermediate_activation)
        
        self.up3 = nn.ConvTranspose2d(
            base_channels * 8, base_channels * 4,
            kernel_size=2, stride=2
        )
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4, dropout=dropout * 0.5, activation=intermediate_activation)
        
        self.up2 = nn.ConvTranspose2d(
            base_channels * 4, base_channels * 2,
            kernel_size=2, stride=2
        )
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2, dropout=dropout * 0.5, activation=intermediate_activation)
        
        self.up1 = nn.ConvTranspose2d(
            base_channels * 2, base_channels,
            kernel_size=2, stride=2
        )
        self.dec1 = DoubleConv(base_channels * 2, base_channels, dropout=0.0, activation=intermediate_activation)
        
        # Output layer: maps to single channel with configurable activation
        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        
        # Set output activation function
        if output_activation.lower() == 'sigmoid':
            self.output_activation = nn.Sigmoid()
            self.activation_name = 'sigmoid'
        elif output_activation.lower() == 'tanh':
            self.output_activation = nn.Tanh()
            self.activation_name = 'tanh'
        else:
            raise ValueError(f"Unsupported output_activation: {output_activation}. Must be 'sigmoid' or 'tanh'")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through UNet.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Probability map tensor of shape (B, 1, H, W) with values in (0,1)
        """
        # Encoder path
        enc1 = self.enc1(x)  # (B, 64, H, W)
        enc2 = self.enc2(self.pool(enc1))  # (B, 128, H/2, W/2)
        enc3 = self.enc3(self.pool(enc2))  # (B, 256, H/4, W/4)
        enc4 = self.enc4(self.pool(enc3))  # (B, 512, H/8, W/8)
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))  # (B, 512, H/16, W/16)
        
        # Decoder path with skip connections
        dec4 = self.up4(bottleneck)  # (B, 512, H/8, W/8)
        dec4 = torch.cat([dec4, enc4], dim=1)  # (B, 1024, H/8, W/8)
        dec4 = self.dec4(dec4)  # (B, 512, H/8, W/8)
        
        dec3 = self.up3(dec4)  # (B, 256, H/4, W/4)
        dec3 = torch.cat([dec3, enc3], dim=1)  # (B, 512, H/4, W/4)
        dec3 = self.dec3(dec3)  # (B, 256, H/4, W/4)
        
        dec2 = self.up2(dec3)  # (B, 128, H/2, W/2)
        dec2 = torch.cat([dec2, enc2], dim=1)  # (B, 256, H/2, W/2)
        dec2 = self.dec2(dec2)  # (B, 128, H/2, W/2)
        
        dec1 = self.up1(dec2)  # (B, 64, H, W)
        dec1 = torch.cat([dec1, enc1], dim=1)  # (B, 128, H, W)
        dec1 = self.dec1(dec1)  # (B, 64, H, W)
        
        # Output: single channel probability map
        out = self.out_conv(dec1)  # (B, 1, H, W)
        
        # Apply output activation
        if self.activation_name == 'sigmoid':
            out = self.output_activation(out)  # Values in (0, 1)
        elif self.activation_name == 'tanh':
            out = self.output_activation(out)  # Values in (-1, 1)
            # Rescale tanh output to (0, 1) for compatibility with loss functions
            out = (out + 1.0) / 2.0  # Map (-1, 1) -> (0, 1)
        
        return out



def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

