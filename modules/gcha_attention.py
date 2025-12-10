"""
Geometry-Constrained Hough Attention (GCHA) Layer

This module implements the core GCHA layer which replaces standard cross-attention
with a geometry-constrained variant using a pre-computed geometric mask.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings


class PerformanceWarning(UserWarning):
    """Warning for performance-related issues"""
    pass


class GCHALayer(nn.Module):
    """
    Geometry-Constrained Hough Attention Layer
    
    This layer implements a masked attention mechanism where the geometric mask
    M_geo enforces spatial constraints based on the Hough transform geometry.
    
    The attention is computed as:
        Attention = Softmax((Q @ K^T) / sqrt(d) + M_geo)
    
    Where M_geo[i, j] = 0 if |x̃_j - x̂_i,j| < ε, else -∞
    
    Args:
        d_model (int): The dimension of the model (embedding dimension)
        n_total (int): Total number of query positions (N_total)
        hw (int): Height × Width of the feature map
        epsilon (float): Threshold for the geometric constraint (ε)
        num_heads (int): Number of attention heads (default: 8)
        dropout (float): Dropout probability (default: 0.1)
    """
    
    def __init__(self, d_model, n_total, hw, epsilon=0.1, num_heads=8, dropout=0.1):
        super(GCHALayer, self).__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.n_total = n_total
        self.hw = hw
        self.epsilon = epsilon
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear layers for Q, K, V projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Pre-computed static geometric mask M_geo of shape (N_total, HW)
        # This will be set by calling set_geometric_mask or will be computed
        # during forward pass if positions are provided
        self.register_buffer('M_geo', None)
        
    def compute_geometric_mask(self, x_tilde, x_hat):
        """
        Compute the static geometric mask M_geo.
        
        Args:
            x_tilde: Spatial positions of feature map
                     - Shape (HW,): 1D positions (automatically expanded to (HW, 1))
                     - Shape (HW, d): d-dimensional positions (e.g., 2D or 3D coordinates)
            x_hat: Predicted positions for each query
                   - Shape (N_total, HW): 1D positions (automatically expanded to (N_total, HW, 1))
                   - Shape (N_total, HW, d): d-dimensional positions
            
        Returns:
            M_geo: Geometric mask of shape (N_total, HW)
                   M_geo[i, j] = 0 if |x̃_j - x̂_i,j| < ε
                   M_geo[i, j] = -∞ otherwise
                   
        Note:
            For 1D inputs, they are automatically expanded to 2D with dimension 1.
            Ensure that x_tilde and x_hat have matching spatial dimensions (d).
        """
        # Handle different input shapes with validation
        if x_tilde.dim() == 1:
            # 1D positions, expand to (HW, 1)
            x_tilde = x_tilde.unsqueeze(-1)
        elif x_tilde.dim() != 2:
            raise ValueError(f"x_tilde must be 1D or 2D, got {x_tilde.dim()}D")
        
        if x_hat.dim() == 2:
            # 1D positions, expand to (N_total, HW, 1)
            x_hat = x_hat.unsqueeze(-1)
        elif x_hat.dim() != 3:
            raise ValueError(f"x_hat must be 2D or 3D, got {x_hat.dim()}D")
        
        # Validate matching dimensions
        if x_tilde.size(-1) != x_hat.size(-1):
            raise ValueError(
                f"Spatial dimensions must match: x_tilde has {x_tilde.size(-1)} "
                f"dimensions but x_hat has {x_hat.size(-1)} dimensions"
            )
        
        # Validate shape compatibility
        if x_tilde.size(0) != x_hat.size(1):
            raise ValueError(
                f"HW dimension mismatch: x_tilde has {x_tilde.size(0)} positions "
                f"but x_hat expects {x_hat.size(1)} positions"
            )
        
        # Compute the L2 distance |x̃_j - x̂_i,j|
        # x_tilde: (HW, dim) -> expand to (1, HW, dim)
        # x_hat: (N_total, HW, dim)
        x_tilde_expanded = x_tilde.unsqueeze(0)  # (1, HW, dim)
        
        # Compute L2 distance efficiently
        distance = torch.norm(x_tilde_expanded - x_hat, p=2, dim=-1)  # (N_total, HW)
        
        # Create mask: 0 if distance < epsilon, -inf otherwise
        M_geo = torch.where(
            distance < self.epsilon,
            torch.zeros_like(distance),
            torch.full_like(distance, float('-inf'))
        )
        
        return M_geo
    
    def set_geometric_mask(self, x_tilde, x_hat):
        """
        Pre-compute and set the static geometric mask.
        
        Args:
            x_tilde: Spatial positions of feature map
            x_hat: Predicted positions for each query
        """
        self.M_geo = self.compute_geometric_mask(x_tilde, x_hat)
    
    def forward(self, query, key, value, x_tilde=None, x_hat=None, mask=None):
        """
        Forward pass of GCHA layer.
        
        Args:
            query: Query tensor of shape (batch_size, n_total, d_model)
            key: Key tensor of shape (batch_size, hw, d_model)
            value: Value tensor of shape (batch_size, hw, d_model)
            x_tilde: Optional spatial positions of feature map (for mask computation)
            x_hat: Optional predicted positions (for mask computation)
            mask: Optional additional attention mask
            
        Returns:
            output: Attention output of shape (batch_size, n_total, d_model)
            attention_weights: Attention weights of shape (batch_size, num_heads, n_total, hw)
            
        Note:
            Computing the geometric mask during forward pass (by providing x_tilde and x_hat)
            can be expensive and should be avoided in production. For better performance,
            pre-compute the mask using set_geometric_mask() before the forward pass.
        """
        batch_size = query.size(0)
        n_queries = query.size(1)
        n_keys = key.size(1)
        
        # If geometric positions are provided, compute the mask
        # Warning: This can be expensive in the forward pass
        if x_tilde is not None and x_hat is not None:
            warnings.warn(
                "Computing geometric mask during forward pass. For better performance, "
                "pre-compute the mask using set_geometric_mask() before forward pass.",
                PerformanceWarning,
                stacklevel=2
            )
            self.M_geo = self.compute_geometric_mask(x_tilde, x_hat)
        
        # Linear projections and reshape for multi-head attention
        # Q: (batch_size, n_total, d_model) -> (batch_size, num_heads, n_total, d_k)
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # K: (batch_size, hw, d_model) -> (batch_size, num_heads, hw, d_k)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # V: (batch_size, hw, d_model) -> (batch_size, num_heads, hw, d_k)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores: Q @ K^T / sqrt(d_k)
        # (batch_size, num_heads, n_total, d_k) @ (batch_size, num_heads, d_k, hw)
        # -> (batch_size, num_heads, n_total, hw)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Add geometric mask M_geo
        # M_geo shape: (N_total, HW) -> expand to (1, 1, N_total, HW) for broadcasting
        if self.M_geo is not None:
            # Validate mask shape matches the current inputs
            if self.M_geo.shape != (n_queries, n_keys):
                raise ValueError(
                    f"Geometric mask shape {self.M_geo.shape} does not match "
                    f"expected shape ({n_queries}, {n_keys}). "
                    f"Please re-compute the mask with correct dimensions."
                )
            M_geo_expanded = self.M_geo.unsqueeze(0).unsqueeze(0)  # (1, 1, N_total, HW)
            scores = scores + M_geo_expanded
        
        # Apply additional mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        # (batch_size, num_heads, n_total, hw) @ (batch_size, num_heads, hw, d_k)
        # -> (batch_size, num_heads, n_total, d_k)
        context = torch.matmul(attention_weights, V)
        
        # Reshape and concatenate heads
        # (batch_size, num_heads, n_total, d_k) -> (batch_size, n_total, num_heads, d_k)
        # -> (batch_size, n_total, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final linear projection
        output = self.W_o(context)
        
        return output, attention_weights
