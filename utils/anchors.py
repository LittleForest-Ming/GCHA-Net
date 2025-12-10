"""
Anchor generation utilities for GCHA-Net.
This module provides functions for generating the parameter grid A (anchors).
"""

import torch
import numpy as np
from typing import Tuple, List


def generate_anchor_grid(
    feature_size: Tuple[int, int],
    num_anchors: int = 9,
    scales: List[float] = None,
    aspect_ratios: List[float] = None,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Generate anchor grid for spatial attention.
    
    Args:
        feature_size: (H, W) size of the feature map
        num_anchors: Number of anchors per location
        scales: List of anchor scales (default: [0.5, 1.0, 2.0])
        aspect_ratios: List of aspect ratios (default: [0.5, 1.0, 2.0])
        device: Device to place anchors on
        
    Returns:
        anchors: Tensor of shape (H*W*num_anchors, 4) with (cx, cy, w, h) format
    """
    if scales is None:
        scales = [0.5, 1.0, 2.0]
    if aspect_ratios is None:
        aspect_ratios = [0.5, 1.0, 2.0]
    
    H, W = feature_size
    
    # Generate base anchors
    base_anchors = []
    for scale in scales:
        for ratio in aspect_ratios:
            w = scale * np.sqrt(ratio)
            h = scale / np.sqrt(ratio)
            base_anchors.append([0, 0, w, h])
    
    base_anchors = torch.tensor(base_anchors, dtype=torch.float32, device=device)
    
    # Generate grid centers
    shift_x = torch.arange(0, W, dtype=torch.float32, device=device) / W
    shift_y = torch.arange(0, H, dtype=torch.float32, device=device) / H
    shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij')
    shifts = torch.stack([shift_x.flatten(), shift_y.flatten()], dim=1)
    
    # Combine shifts with base anchors
    anchors = []
    for shift in shifts:
        for base_anchor in base_anchors:
            cx, cy = shift[0], shift[1]
            w, h = base_anchor[2], base_anchor[3]
            anchors.append([cx, cy, w, h])
    
    anchors = torch.stack([torch.tensor(a, device=device) for a in anchors])
    return anchors


def generate_parameter_grid(
    N_total: int,
    feature_dim: int,
    grid_size: Tuple[int, int] = (7, 7),
    epsilon: float = 1e-6,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Generate parameter grid A for GCHA attention.
    
    Args:
        N_total: Total number of attention parameters
        feature_dim: Dimension of input features
        grid_size: Spatial grid size for parameter distribution
        epsilon: Small value for numerical stability
        device: Device to place parameters on
        
    Returns:
        param_grid: Parameter tensor of shape (N_total, feature_dim)
    """
    # Initialize parameter grid with Xavier uniform initialization
    param_grid = torch.empty(N_total, feature_dim, device=device)
    torch.nn.init.xavier_uniform_(param_grid)
    
    # Add small epsilon for numerical stability
    param_grid = param_grid + epsilon * torch.randn_like(param_grid)
    
    return param_grid


def compute_anchor_iou(anchors1: torch.Tensor, anchors2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between two sets of anchors.
    
    Args:
        anchors1: Tensor of shape (N, 4) with (cx, cy, w, h)
        anchors2: Tensor of shape (M, 4) with (cx, cy, w, h)
        
    Returns:
        iou: Tensor of shape (N, M) with IoU values
    """
    # Convert from (cx, cy, w, h) to (x1, y1, x2, y2)
    def convert_to_corners(anchors):
        cx, cy, w, h = anchors[:, 0], anchors[:, 1], anchors[:, 2], anchors[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return torch.stack([x1, y1, x2, y2], dim=1)
    
    boxes1 = convert_to_corners(anchors1)
    boxes2 = convert_to_corners(anchors2)
    
    # Compute intersection
    x1_max = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    y1_max = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    x2_min = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    y2_min = torch.min(boxes1[:, None, 3], boxes2[:, 3])
    
    intersection = torch.clamp(x2_min - x1_max, min=0) * torch.clamp(y2_min - y1_max, min=0)
    
    # Compute union
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2 - intersection
    
    # Compute IoU
    iou = intersection / (union + 1e-6)
    return iou
