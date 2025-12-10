"""Geometric utilities for GCHA-Net.

This module implements geometric constraint generation for polynomial lane detection.
"""

import torch
import numpy as np


def generate_anchors(k_range=(-0.5, 0.5), m_range=(-1.0, 1.0), b_range=(0.0, 1.0),
                     k_steps=5, m_steps=9, b_steps=9):
    """Generate a 3D grid of polynomial parameters (k, m, b).
    
    The polynomial is defined as: x = k*y^2 + m*y + b
    where x, y are normalized coordinates in [0, 1]^2.
    
    Args:
        k_range: Tuple (min, max) for quadratic coefficient k
        m_range: Tuple (min, max) for linear coefficient m
        b_range: Tuple (min, max) for constant coefficient b
        k_steps: Number of discrete values for k
        m_steps: Number of discrete values for m
        b_steps: Number of discrete values for b
        
    Returns:
        anchors: Tensor of shape (k_steps * m_steps * b_steps, 3)
                 Each row is [k, m, b]
    """
    k_values = np.linspace(k_range[0], k_range[1], k_steps)
    m_values = np.linspace(m_range[0], m_range[1], m_steps)
    b_values = np.linspace(b_range[0], b_range[1], b_steps)
    
    # Create 3D grid
    k_grid, m_grid, b_grid = np.meshgrid(k_values, m_values, b_values, indexing='ij')
    
    # Flatten and stack
    anchors = np.stack([k_grid.flatten(), m_grid.flatten(), b_grid.flatten()], axis=1)
    
    return torch.from_numpy(anchors).float()


def generate_geometric_mask(H, W, anchors, epsilon=0.05):
    """Generate geometric attention masks for polynomial trajectories.
    
    The mask prevents attention to pixels outside a polynomial trajectory.
    Coordinates are normalized to [0, 1]^2 with inverted Y-axis for 
    navigation perspective.
    
    Args:
        H: Image height in pixels
        W: Image width in pixels
        anchors: Tensor of shape (N, 3) containing [k, m, b] parameters
        epsilon: Distance threshold for masking (in normalized coordinates)
        
    Returns:
        masks: Tensor of shape (N, H, W) where:
               - 0 if pixel is within epsilon of the polynomial
               - -inf if pixel is outside the trajectory
    """
    N = anchors.shape[0]
    device = anchors.device
    
    # Ensure anchors are on the correct device
    anchors = anchors.to(device)
    
    # Create pixel coordinate grids
    # v ranges from 0 to H-1, u ranges from 0 to W-1
    v_coords = torch.arange(H, dtype=torch.float32, device=device)  # [H]
    u_coords = torch.arange(W, dtype=torch.float32, device=device)  # [W]
    
    # Create meshgrid
    v_grid, u_grid = torch.meshgrid(v_coords, u_coords, indexing='ij')  # [H, W]
    
    # Normalize coordinates to [0, 1]
    # x_tilde corresponds to horizontal axis (u/W)
    # y_tilde corresponds to vertical axis with inverted Y (1 - v/H)
    x_tilde = u_grid / W  # [H, W]
    y_tilde = 1.0 - v_grid / H  # [H, W] - inverted Y-axis
    
    # Expand dimensions for broadcasting
    x_tilde = x_tilde.unsqueeze(0)  # [1, H, W]
    y_tilde = y_tilde.unsqueeze(0)  # [1, H, W]
    
    # Extract anchor parameters
    k = anchors[:, 0].view(N, 1, 1)  # [N, 1, 1]
    m = anchors[:, 1].view(N, 1, 1)  # [N, 1, 1]
    b = anchors[:, 2].view(N, 1, 1)  # [N, 1, 1]
    
    # Compute polynomial: x_poly = k*y^2 + m*y + b
    x_poly = k * y_tilde.pow(2) + m * y_tilde + b  # [N, H, W]
    
    # Compute distance from polynomial
    distance = torch.abs(x_tilde - x_poly)  # [N, H, W]
    
    # Create mask: 0 if within epsilon, -inf otherwise
    masks = torch.where(
        distance < epsilon,
        torch.zeros_like(distance),
        torch.full_like(distance, float('-inf'))
    )
    
    return masks


def polynomial_fit_from_points(points, image_height, image_width):
    """Fit a polynomial (k, m, b) to a set of lane points.
    
    Converts pixel coordinates to normalized coordinates with inverted Y-axis,
    then fits x = k*y^2 + m*y + b using least squares.
    
    Args:
        points: Array of shape (N, 2) containing [u, v] pixel coordinates
        image_height: Height of the image
        image_width: Width of the image
        
    Returns:
        params: Tensor of shape (3,) containing [k, m, b]
                Returns [0, 0, 0.5] if fitting fails
    """
    if len(points) < 3:
        # Not enough points for quadratic fit, return default
        return torch.tensor([0.0, 0.0, 0.5], dtype=torch.float32)
    
    points = np.array(points)
    
    # Normalize coordinates
    x_tilde = points[:, 0] / image_width
    y_tilde = 1.0 - points[:, 1] / image_height  # Inverted Y-axis
    
    # Build design matrix for least squares: x = k*y^2 + m*y + b
    # A = [y^2, y, 1]
    A = np.stack([y_tilde**2, y_tilde, np.ones_like(y_tilde)], axis=1)
    
    try:
        # Solve least squares: A @ [k, m, b]^T = x
        params, residuals, rank, s = np.linalg.lstsq(A, x_tilde, rcond=None)
        return torch.from_numpy(params).float()
    except np.linalg.LinAlgError:
        # If fitting fails, return default parameters
        return torch.tensor([0.0, 0.0, 0.5], dtype=torch.float32)
