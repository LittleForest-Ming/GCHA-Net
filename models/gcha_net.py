"""
GCHA-Net: Guided Cross-Hierarchical Attention Network.

Main model definition integrating all components including GCHA attention,
anchor generation, and hierarchical feature processing.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.gcha_attention import GCHAAttention, GCHABlock
from utils.anchors import generate_anchor_grid, get_position_encoding


class FeatureExtractor(nn.Module):
    """
    Feature extraction backbone with hierarchical outputs.
    
    This is a simple CNN-based feature extractor that produces multi-scale features.
    Can be replaced with more sophisticated backbones (ResNet, ViT, etc.).
    
    Args:
        in_channels: Number of input channels
        base_channels: Base number of channels
        num_stages: Number of hierarchical stages
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_stages: int = 4
    ):
        super().__init__()
        
        self.num_stages = num_stages
        self.stages = nn.ModuleList()
        
        current_channels = in_channels
        for i in range(num_stages):
            out_channels = base_channels * (2 ** i)
            stage = nn.Sequential(
                nn.Conv2d(current_channels, out_channels, kernel_size=3, 
                         stride=2 if i > 0 else 1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.stages.append(stage)
            current_channels = out_channels
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract hierarchical features.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            List of feature tensors at different scales
        """
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


class GCHANet(nn.Module):
    """
    GCHA-Net: Guided Cross-Hierarchical Attention Network.
    
    Main network architecture that integrates:
    - Multi-scale feature extraction
    - Anchor-based attention guidance
    - Cross-hierarchical attention fusion
    
    Args:
        in_channels: Number of input channels
        num_classes: Number of output classes for segmentation
        embed_dim: Embedding dimension for attention
        num_heads: Number of attention heads
        n_anchors: Number of anchor points (N_total)
        epsilon: Epsilon for numerical stability
        num_layers: Number of GCHA blocks
        mlp_ratio: MLP expansion ratio
        dropout: Dropout probability
        base_channels: Base channels for feature extractor
        num_stages: Number of feature extraction stages
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 19,  # Default for cityscapes/agroscapes
        embed_dim: int = 256,
        num_heads: int = 8,
        n_anchors: int = 64,
        epsilon: float = 1e-6,
        num_layers: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        base_channels: int = 64,
        num_stages: int = 4
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.n_anchors = n_anchors
        self.epsilon = epsilon
        self.num_stages = num_stages
        
        # Feature extraction backbone
        self.backbone = FeatureExtractor(
            in_channels=in_channels,
            base_channels=base_channels,
            num_stages=num_stages
        )
        
        # Channel projection for each stage
        self.stage_projections = nn.ModuleList()
        for i in range(num_stages):
            stage_channels = base_channels * (2 ** i)
            self.stage_projections.append(
                nn.Conv2d(stage_channels, embed_dim, kernel_size=1)
            )
        
        # GCHA transformer blocks
        self.gcha_blocks = nn.ModuleList([
            GCHABlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                n_anchors=n_anchors,
                epsilon=epsilon,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Cross-hierarchical fusion
        self.fusion = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(embed_dim),
                nn.ReLU(inplace=True)
            )
            for _ in range(num_stages)
        ])
        
        # Segmentation head
        self.decode_head = nn.Sequential(
            nn.Conv2d(embed_dim * num_stages, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, num_classes, kernel_size=1)
        )
        
        # Anchor positions (will be initialized in forward)
        self.register_buffer('anchor_positions', None)
    
    def _get_positions(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        """
        Generate 2D position grid for feature map.
        
        Args:
            height: Height of feature map
            width: Width of feature map
            device: Device to place positions on
            
        Returns:
            torch.Tensor: Position grid of shape (H*W, 2)
        """
        y_coords = torch.arange(height, dtype=torch.float32, device=device)
        x_coords = torch.arange(width, dtype=torch.float32, device=device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        positions = torch.stack([yy.flatten(), xx.flatten()], dim=-1)
        return positions
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of GCHA-Net.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            torch.Tensor: Segmentation output (B, num_classes, H, W)
        """
        B, _, H, W = x.shape
        device = x.device
        
        # Extract multi-scale features
        features = self.backbone(x)  # List of (B, C_i, H_i, W_i)
        
        # Process each feature level
        processed_features = []
        
        for i, feat in enumerate(features):
            _, _, H_i, W_i = feat.shape
            
            # Project to embedding dimension
            feat_proj = self.stage_projections[i](feat)  # (B, embed_dim, H_i, W_i)
            
            # Reshape to sequence: (B, embed_dim, H_i, W_i) -> (B, H_i*W_i, embed_dim)
            feat_seq = feat_proj.flatten(2).transpose(1, 2)
            
            # Generate positions
            positions = self._get_positions(H_i, W_i, device)  # (H_i*W_i, 2)
            positions_batch = positions.unsqueeze(0).expand(B, -1, -1)  # (B, H_i*W_i, 2)
            
            # Generate anchor positions for this scale
            anchor_pos = generate_anchor_grid(
                self.n_anchors, (H_i, W_i), device=device
            )  # (n_anchors, 2)
            
            # Apply GCHA blocks
            for gcha_block in self.gcha_blocks:
                feat_seq = gcha_block(
                    feat_seq,
                    pos=positions_batch,
                    anchor_pos=anchor_pos
                )
            
            # Reshape back to spatial: (B, H_i*W_i, embed_dim) -> (B, embed_dim, H_i, W_i)
            feat_spatial = feat_seq.transpose(1, 2).view(B, self.embed_dim, H_i, W_i)
            
            # Apply fusion
            feat_fused = self.fusion[i](feat_spatial)
            
            processed_features.append(feat_fused)
        
        # Upsample all features to the largest resolution (first feature map size)
        target_size = (H, W)
        upsampled_features = []
        
        for feat in processed_features:
            if feat.shape[2:] != target_size:
                feat_up = nn.functional.interpolate(
                    feat, size=target_size, mode='bilinear', align_corners=False
                )
            else:
                feat_up = feat
            upsampled_features.append(feat_up)
        
        # Concatenate multi-scale features
        # Note: This concatenation can use significant memory with many stages
        # For memory-constrained scenarios, consider:
        # 1. Progressive fusion (fusing features incrementally)
        # 2. Learned fusion with 1x1 convs to reduce channels first
        # 3. Attention-based weighted fusion instead of concatenation
        fused = torch.cat(upsampled_features, dim=1)  # (B, embed_dim*num_stages, H, W)
        
        # Decode to segmentation map
        output = self.decode_head(fused)  # (B, num_classes, H, W)
        
        return output
    
    def get_anchor_positions(self, feature_size: Tuple[int, int]) -> torch.Tensor:
        """
        Get anchor positions for a given feature size.
        
        Args:
            feature_size: (H, W) of the feature map
            
        Returns:
            torch.Tensor: Anchor positions (n_anchors, 2)
        """
        device = next(self.parameters()).device
        return generate_anchor_grid(self.n_anchors, feature_size, device=device)


def build_gcha_net(config: dict) -> GCHANet:
    """
    Build GCHA-Net from configuration dictionary.
    
    Args:
        config: Configuration dictionary with model parameters
        
    Returns:
        GCHANet: Initialized model
    """
    model = GCHANet(
        in_channels=config.get('in_channels', 3),
        num_classes=config.get('num_classes', 19),
        embed_dim=config.get('embed_dim', 256),
        num_heads=config.get('num_heads', 8),
        n_anchors=config.get('n_total', 64),  # N_total parameter
        epsilon=config.get('epsilon', 1e-6),
        num_layers=config.get('num_layers', 4),
        mlp_ratio=config.get('mlp_ratio', 4.0),
        dropout=config.get('dropout', 0.1),
        base_channels=config.get('base_channels', 64),
        num_stages=config.get('num_stages', 4)
    )
    return model
