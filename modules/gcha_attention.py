"""
GCHA (Grid-based Channel-wise Hierarchical Attention) implementation.
This module contains the GCHA attention layer and mask generation logic.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class GCHAAttention(nn.Module):
    """
    Grid-based Channel-wise Hierarchical Attention layer.
    
    This attention mechanism combines spatial and channel-wise attention
    with a hierarchical grid-based approach for efficient feature refinement.
    """
    
    def __init__(
        self,
        in_channels: int,
        num_heads: int = 8,
        grid_size: Tuple[int, int] = (7, 7),
        N_total: int = 256,
        epsilon: float = 1e-6,
        dropout: float = 0.1
    ):
        """
        Initialize GCHA attention layer.
        
        Args:
            in_channels: Number of input channels
            num_heads: Number of attention heads
            grid_size: Spatial grid size for hierarchical attention
            N_total: Total number of attention parameters
            epsilon: Small value for numerical stability
            dropout: Dropout probability
        """
        super(GCHAAttention, self).__init__()
        
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.grid_size = grid_size
        self.N_total = N_total
        self.epsilon = epsilon
        self.head_dim = in_channels // num_heads
        
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"
        
        # Query, Key, Value projections
        self.query_proj = nn.Linear(in_channels, in_channels)
        self.key_proj = nn.Linear(in_channels, in_channels)
        self.value_proj = nn.Linear(in_channels, in_channels)
        
        # Grid-based parameter learning
        self.grid_params = nn.Parameter(torch.randn(N_total, in_channels) * 0.01)
        
        # Spatial attention parameters
        self.spatial_conv = nn.Conv2d(in_channels, num_heads, kernel_size=1)
        
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 16, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Output projection
        self.out_proj = nn.Linear(in_channels, in_channels)
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)
        
    def generate_attention_mask(
        self,
        features: torch.Tensor,
        threshold: float = 0.5
    ) -> torch.Tensor:
        """
        Generate hierarchical attention mask based on feature importance.
        
        Args:
            features: Input features of shape (B, C, H, W)
            threshold: Threshold for mask binarization
            
        Returns:
            mask: Attention mask of shape (B, num_heads, H, W)
        """
        B, C, H, W = features.shape
        
        # Generate spatial attention scores
        spatial_scores = self.spatial_conv(features)  # (B, num_heads, H, W)
        
        # Apply sigmoid and threshold
        spatial_mask = torch.sigmoid(spatial_scores)
        
        # Grid-based hierarchical masking
        grid_h, grid_w = self.grid_size
        mask_grid = F.adaptive_avg_pool2d(spatial_mask, (grid_h, grid_w))
        mask_grid = (mask_grid > threshold).float()
        
        # Upsample back to original size
        hierarchical_mask = F.interpolate(
            mask_grid, 
            size=(H, W), 
            mode='nearest'
        )
        
        return hierarchical_mask
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of GCHA attention.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            mask: Optional attention mask of shape (B, num_heads, H, W)
            
        Returns:
            output: Attention output of shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Generate attention mask if not provided
        if mask is None:
            mask = self.generate_attention_mask(x)
        
        # Apply channel attention
        channel_weights = self.channel_attention(x)
        x_channel = x * channel_weights
        
        # Reshape for multi-head attention
        # (B, C, H, W) -> (B, H*W, C)
        x_flat = x_channel.flatten(2).transpose(1, 2)
        
        # Compute Q, K, V
        Q = self.query_proj(x_flat)  # (B, H*W, C)
        K = self.key_proj(x_flat)    # (B, H*W, C)
        V = self.value_proj(x_flat)  # (B, H*W, C)
        
        # Reshape for multi-head attention
        # (B, H*W, C) -> (B, H*W, num_heads, head_dim) -> (B, num_heads, H*W, head_dim)
        Q = Q.view(B, H*W, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, H*W, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, H*W, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply spatial mask
        mask_flat = mask.flatten(2)  # (B, num_heads, H*W)
        mask_flat = mask_flat.unsqueeze(-1) * mask_flat.unsqueeze(-2)  # (B, num_heads, H*W, H*W)
        attention_scores = attention_scores + (1.0 - mask_flat) * (-1e9)
        
        # Apply softmax
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, V)  # (B, num_heads, H*W, head_dim)
        
        # Reshape back
        # (B, num_heads, H*W, head_dim) -> (B, H*W, num_heads, head_dim) -> (B, H*W, C)
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(B, H*W, C)
        
        # Output projection
        attention_output = self.out_proj(attention_output)
        attention_output = self.dropout(attention_output)
        
        # Residual connection and layer norm
        x_flat = self.norm1(x_flat + attention_output)
        
        # Reshape back to spatial format
        output = x_flat.transpose(1, 2).view(B, C, H, W)
        
        return output


class GCHABlock(nn.Module):
    """
    Complete GCHA block with attention and feed-forward network.
    """
    
    def __init__(
        self,
        in_channels: int,
        num_heads: int = 8,
        grid_size: Tuple[int, int] = (7, 7),
        N_total: int = 256,
        epsilon: float = 1e-6,
        dropout: float = 0.1,
        expansion: int = 4
    ):
        """
        Initialize GCHA block.
        
        Args:
            in_channels: Number of input channels
            num_heads: Number of attention heads
            grid_size: Spatial grid size
            N_total: Total number of attention parameters
            epsilon: Numerical stability parameter
            dropout: Dropout probability
            expansion: Expansion factor for FFN
        """
        super(GCHABlock, self).__init__()
        
        self.attention = GCHAAttention(
            in_channels=in_channels,
            num_heads=num_heads,
            grid_size=grid_size,
            N_total=N_total,
            epsilon=epsilon,
            dropout=dropout
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * expansion, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(in_channels * expansion, in_channels, kernel_size=1),
            nn.Dropout(dropout)
        )
        
        # Ensure num_groups divides in_channels evenly
        num_groups = 32
        while in_channels % num_groups != 0 and num_groups > 1:
            num_groups //= 2
        self.norm = nn.GroupNorm(num_groups, in_channels)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through GCHA block.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            mask: Optional attention mask
            
        Returns:
            output: Block output of shape (B, C, H, W)
        """
        # Attention with residual
        x = x + self.attention(x, mask)
        
        # FFN with residual
        x = x + self.ffn(self.norm(x))
        
        return x
