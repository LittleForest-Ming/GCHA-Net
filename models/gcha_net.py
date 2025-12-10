"""
GCHA-Net: Grid-based Channel-wise Hierarchical Attention Network
Main model definition integrating all components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.gcha_attention import GCHAAttention, GCHABlock
from utils.anchors import generate_parameter_grid, generate_anchor_grid


class GCHAEncoder(nn.Module):
    """
    Encoder module with GCHA blocks for feature extraction.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_blocks: List[int] = [2, 2, 6, 2],
        num_heads: List[int] = [2, 4, 8, 16],
        grid_size: Tuple[int, int] = (7, 7),
        N_total: int = 256,
        epsilon: float = 1e-6,
        dropout: float = 0.1
    ):
        """
        Initialize GCHA encoder.
        
        Args:
            in_channels: Number of input channels
            base_channels: Base number of channels
            num_blocks: Number of blocks at each stage
            num_heads: Number of attention heads at each stage
            grid_size: Grid size for GCHA attention
            N_total: Total number of attention parameters
            epsilon: Numerical stability parameter
            dropout: Dropout probability
        """
        super(GCHAEncoder, self).__init__()
        
        self.num_stages = len(num_blocks)
        
        # Stem convolution
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(32, base_channels),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, base_channels),
            nn.GELU()
        )
        
        # Build stages
        self.stages = nn.ModuleList()
        current_channels = base_channels
        
        for stage_idx in range(self.num_stages):
            stage_blocks = []
            out_channels = base_channels * (2 ** stage_idx)
            
            # Downsample if not first stage
            if stage_idx > 0:
                stage_blocks.append(
                    nn.Sequential(
                        nn.Conv2d(current_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
                        nn.GroupNorm(32, out_channels),
                        nn.GELU()
                    )
                )
            
            # Add GCHA blocks
            for _ in range(num_blocks[stage_idx]):
                stage_blocks.append(
                    GCHABlock(
                        in_channels=out_channels,
                        num_heads=num_heads[stage_idx],
                        grid_size=grid_size,
                        N_total=N_total,
                        epsilon=epsilon,
                        dropout=dropout
                    )
                )
            
            self.stages.append(nn.Sequential(*stage_blocks))
            current_channels = out_channels
        
        self.out_channels = current_channels
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through encoder.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            features: List of feature maps from each stage
        """
        features = []
        
        x = self.stem(x)
        
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        
        return features


class GCHADecoder(nn.Module):
    """
    Decoder module for upsampling and feature refinement.
    """
    
    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int = 64,
        num_classes: int = 1000
    ):
        """
        Initialize GCHA decoder.
        
        Args:
            in_channels_list: List of input channels from encoder stages
            out_channels: Number of output channels
            num_classes: Number of output classes
        """
        super(GCHADecoder, self).__init__()
        
        self.num_stages = len(in_channels_list)
        
        # Upsampling blocks
        self.upsample_blocks = nn.ModuleList()
        
        for i in range(self.num_stages - 1, 0, -1):
            self.upsample_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels_list[i], in_channels_list[i-1], kernel_size=2, stride=2),
                    nn.GroupNorm(32, in_channels_list[i-1]),
                    nn.GELU()
                )
            )
        
        # Final output head
        self.head = nn.Sequential(
            nn.Conv2d(in_channels_list[0], out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, num_classes, kernel_size=1)
        )
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through decoder.
        
        Args:
            features: List of feature maps from encoder
            
        Returns:
            output: Decoded output tensor
        """
        x = features[-1]
        
        # Upsample and fuse features
        for i, upsample in enumerate(self.upsample_blocks):
            x = upsample(x)
            # Add skip connection from encoder
            feat_idx = len(features) - 2 - i
            x = x + features[feat_idx]
        
        # Final prediction
        output = self.head(x)
        
        return output


class GCHANet(nn.Module):
    """
    Complete GCHA-Net model for semantic segmentation or classification.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 19,
        base_channels: int = 64,
        num_blocks: List[int] = [2, 2, 6, 2],
        num_heads: List[int] = [2, 4, 8, 16],
        grid_size: Tuple[int, int] = (7, 7),
        N_total: int = 256,
        epsilon: float = 1e-6,
        dropout: float = 0.1,
        task: str = 'segmentation'
    ):
        """
        Initialize GCHA-Net.
        
        Args:
            in_channels: Number of input channels
            num_classes: Number of output classes
            base_channels: Base number of channels
            num_blocks: Number of blocks at each stage
            num_heads: Number of attention heads at each stage
            grid_size: Grid size for GCHA attention
            N_total: Total number of attention parameters
            epsilon: Numerical stability parameter
            dropout: Dropout probability
            task: Task type ('segmentation' or 'classification')
        """
        super(GCHANet, self).__init__()
        
        self.task = task
        self.num_classes = num_classes
        
        # Encoder
        self.encoder = GCHAEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            grid_size=grid_size,
            N_total=N_total,
            epsilon=epsilon,
            dropout=dropout
        )
        
        # Generate parameter grid for attention
        encoder_channels = [base_channels * (2 ** i) for i in range(len(num_blocks))]
        
        if task == 'segmentation':
            # Decoder for segmentation
            self.decoder = GCHADecoder(
                in_channels_list=encoder_channels,
                out_channels=base_channels,
                num_classes=num_classes
            )
        elif task == 'classification':
            # Classification head
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(encoder_channels[-1], 512),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(512, num_classes)
            )
        else:
            raise ValueError(f"Unknown task: {task}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GCHA-Net.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            output: Model output (segmentation map or class logits)
        """
        # Encode
        features = self.encoder(x)
        
        # Task-specific head
        if self.task == 'segmentation':
            output = self.decoder(features)
            # Upsample to original resolution
            output = F.interpolate(output, size=x.shape[2:], mode='bilinear', align_corners=False)
        elif self.task == 'classification':
            output = self.classifier(features[-1])
        
        return output
    
    def get_parameter_grid(self, feature_dim: int, device: str = 'cuda') -> torch.Tensor:
        """
        Get the parameter grid for GCHA attention.
        
        Args:
            feature_dim: Feature dimension
            device: Device to place parameters on
            
        Returns:
            param_grid: Parameter grid tensor
        """
        return generate_parameter_grid(
            N_total=256,
            feature_dim=feature_dim,
            grid_size=(7, 7),
            epsilon=1e-6,
            device=device
        )


def build_gcha_net(config: dict) -> GCHANet:
    """
    Build GCHA-Net model from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        model: GCHA-Net model
    """
    return GCHANet(
        in_channels=config.get('in_channels', 3),
        num_classes=config.get('num_classes', 19),
        base_channels=config.get('base_channels', 64),
        num_blocks=config.get('num_blocks', [2, 2, 6, 2]),
        num_heads=config.get('num_heads', [2, 4, 8, 16]),
        grid_size=tuple(config.get('grid_size', [7, 7])),
        N_total=config.get('N_total', 256),
        epsilon=config.get('epsilon', 1e-6),
        dropout=config.get('dropout', 0.1),
        task=config.get('task', 'segmentation')
    )
