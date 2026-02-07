"""
Old Version: Used for secondary prediction offset
Anchor generation utilities for GCHA-Net.

This module provides functions for generating the parameter grid A (anchors)
used in the GCHA attention mechanism.
"""

import torch
import numpy as np
from typing import Tuple, List, Optional


def generate_anchor_grid(
    n_total: int,
    feature_size: Tuple[int, int],
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """
    Generate anchor parameter grid A for GCHA attention.
    
    Args:
        n_total: Total number of anchors to generate
        feature_size: Size of the feature map (H, W)
        device: Device to place the anchors on
        
    Returns:
        torch.Tensor: Anchor grid of shape (n_total, 2) containing (y, x) coordinates
    """
    H, W = feature_size
    
    # Generate uniform grid of anchors across the feature map
    n_h = int(np.sqrt(n_total * H / W))
    n_w = int(n_total / n_h)
    
    # Adjust to get exactly n_total anchors
    while n_h * n_w < n_total:
        if n_h < n_w:
            n_h += 1
        else:
            n_w += 1
    
    # Generate grid coordinates
    y_coords = torch.linspace(0, H - 1, n_h, device=device)
    x_coords = torch.linspace(0, W - 1, n_w, device=device)
    
    # Create meshgrid
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
    
    # Flatten and combine
    anchors = torch.stack([yy.flatten(), xx.flatten()], dim=1)
    
    # Take exactly n_total anchors
    anchors = anchors[:n_total]
    
    return anchors


def generate_hierarchical_anchors(
    n_total: int,
    feature_sizes: List[Tuple[int, int]],
    device: torch.device = torch.device('cpu')
) -> List[torch.Tensor]:
    """
    Generate hierarchical anchor grids for multi-scale features.
    
    Args:
        n_total: Total number of anchors per level
        feature_sizes: List of feature map sizes [(H1, W1), (H2, W2), ...]
        device: Device to place the anchors on
        
    Returns:
        List[torch.Tensor]: List of anchor grids, one per feature level
    """
    anchor_grids = []
    
    for feature_size in feature_sizes:
        anchors = generate_anchor_grid(n_total, feature_size, device)
        anchor_grids.append(anchors)
    
    return anchor_grids


def compute_anchor_distances(
    query_positions: torch.Tensor,
    anchor_positions: torch.Tensor,
    epsilon: float = 1e-6
) -> torch.Tensor:
    """
    Compute distances between query positions and anchor positions.
    
    Args:
        query_positions: Query positions of shape (N, 2) or (B, N, 2)
        anchor_positions: Anchor positions of shape (M, 2) or (B, M, 2)
        epsilon: Small value for numerical stability
        
    Returns:
        torch.Tensor: Distance matrix of shape (N, M) or (B, N, M)
    """
    if query_positions.dim() == 2:
        # (N, 2) -> (N, 1, 2)
        query_positions = query_positions.unsqueeze(1)
        # (M, 2) -> (1, M, 2)
        anchor_positions = anchor_positions.unsqueeze(0)
    elif query_positions.dim() == 3:
        # (B, N, 2) -> (B, N, 1, 2)
        query_positions = query_positions.unsqueeze(2)
        # (B, M, 2) -> (B, 1, M, 2)
        anchor_positions = anchor_positions.unsqueeze(1)
    
    # Compute Euclidean distance
    distances = torch.norm(query_positions - anchor_positions, dim=-1, p=2)
    distances = distances + epsilon  # Add epsilon for stability
    
    return distances


def get_position_encoding(
    positions: torch.Tensor,
    d_model: int,
    temperature: float = 10000.0
) -> torch.Tensor:
    """
    Generate sinusoidal position encodings for 2D positions.
    
    Args:
        positions: Position coordinates of shape (..., 2)
        d_model: Dimension of the encoding (should be even)
        temperature: Temperature for the sinusoidal encoding
        
    Returns:
        torch.Tensor: Position encodings of shape (..., d_model)
    """
    assert d_model % 2 == 0, "d_model must be even for 2D position encoding"
    
    d_half = d_model // 2
    div_term = torch.exp(
        torch.arange(0, d_half, 2, dtype=torch.float32, device=positions.device)
        * -(torch.log(torch.tensor(temperature, device=positions.device)) / d_half)
    )
    
    # Split into y and x encodings
    y_pos = positions[..., 0:1]  # (..., 1)
    x_pos = positions[..., 1:2]  # (..., 1)
    
    # Generate encodings
    y_enc_sin = torch.sin(y_pos * div_term)
    y_enc_cos = torch.cos(y_pos * div_term)
    x_enc_sin = torch.sin(x_pos * div_term)
    x_enc_cos = torch.cos(x_pos * div_term)
    
    # Interleave y and x encodings
    pos_enc = torch.cat([
        y_enc_sin, y_enc_cos, x_enc_sin, x_enc_cos
    ], dim=-1)
    
    return pos_enc
