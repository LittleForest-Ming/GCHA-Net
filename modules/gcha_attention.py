"""
Geometry-Constrained Hough Attention (GCHA) Layer

This module implements the core GCHA layer which replaces standard cross-attention
with a geometry-constrained variant using a pre-computed geometric mask.
GCHA (Guided Cross-Hierarchical Attention) implementation.

This module implements the GCHA attention layer and mask generation logic.
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
from typing import Optional, Tuple
import math


class GCHAAttention(nn.Module):
    """
    Guided Cross-Hierarchical Attention Layer.
    
    This attention mechanism uses anchor-based guidance to attend across
    hierarchical feature representations.
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        n_anchors: Number of anchor points
        epsilon: Small value for numerical stability in distance computation
        dropout: Dropout probability
        bias: Whether to use bias in projections
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        n_anchors: int = 64,
        epsilon: float = 1e-6,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.n_anchors = n_anchors
        self.epsilon = epsilon
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Anchor embedding
        self.anchor_embed = nn.Parameter(torch.randn(n_anchors, embed_dim))
        
        # Position encoding projection
        self.pos_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Guided attention parameters
        self.guide_scale = nn.Parameter(torch.ones(1))
        
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.xavier_uniform_(self.pos_proj.weight)
        
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0)
            nn.init.constant_(self.k_proj.bias, 0)
            nn.init.constant_(self.v_proj.bias, 0)
            nn.init.constant_(self.out_proj.bias, 0)
            nn.init.constant_(self.pos_proj.bias, 0)
        
        # Initialize anchor embeddings
        nn.init.normal_(self.anchor_embed, std=0.02)
    
    def generate_attention_mask(
        self,
        query_pos: torch.Tensor,
        key_pos: torch.Tensor,
        anchor_pos: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate guided attention mask based on anchor positions.
        
        Args:
            query_pos: Query positions (B, N, 2)
            key_pos: Key positions (B, M, 2)
            anchor_pos: Anchor positions (n_anchors, 2)
            
        Returns:
            torch.Tensor: Attention mask of shape (B, N, M)
        """
        B, N, _ = query_pos.shape
        M = key_pos.shape[1]
        
        # Expand anchor positions for batch
        anchor_pos = anchor_pos.unsqueeze(0).expand(B, -1, -1)  # (B, n_anchors, 2)
        
        # Compute distances from queries to anchors
        query_pos_exp = query_pos.unsqueeze(2)  # (B, N, 1, 2)
        anchor_pos_exp = anchor_pos.unsqueeze(1)  # (B, 1, n_anchors, 2)
        q_anchor_dist = torch.norm(
            query_pos_exp - anchor_pos_exp, dim=-1, p=2
        )  # (B, N, n_anchors)
        
        # Compute distances from keys to anchors
        key_pos_exp = key_pos.unsqueeze(2)  # (B, M, 1, 2)
        k_anchor_dist = torch.norm(
            key_pos_exp - anchor_pos_exp, dim=-1, p=2
        )  # (B, M, n_anchors)
        
        # Find nearest anchor for each query and key
        q_nearest_anchor = torch.argmin(q_anchor_dist, dim=-1)  # (B, N)
        k_nearest_anchor = torch.argmin(k_anchor_dist, dim=-1)  # (B, M)
        
        # Create mask: allow attention between queries and keys with same nearest anchor
        q_nearest_exp = q_nearest_anchor.unsqueeze(2)  # (B, N, 1)
        k_nearest_exp = k_nearest_anchor.unsqueeze(1)  # (B, 1, M)
        mask = (q_nearest_exp == k_nearest_exp).float()  # (B, N, M)
        
        # Soft masking: add distance-based weighting
        query_key_dist = torch.norm(
            query_pos.unsqueeze(2) - key_pos.unsqueeze(1), dim=-1, p=2
        )  # (B, N, M)
        dist_weight = torch.exp(-query_key_dist / (2 * self.epsilon))
        
        # Combine hard and soft masks
        combined_mask = mask * dist_weight
        
        return combined_mask
    
    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        key_pos: Optional[torch.Tensor] = None,
        anchor_pos: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of GCHA attention.
        
        Args:
            query: Query tensor (B, N, C)
            key: Key tensor (B, M, C), defaults to query if None
            value: Value tensor (B, M, C), defaults to key if None
            query_pos: Query positions (B, N, 2) for spatial guidance
            key_pos: Key positions (B, M, 2) for spatial guidance
            anchor_pos: Anchor positions (n_anchors, 2)
            attn_mask: Additional attention mask (B, N, M) or (B, num_heads, N, M)
            need_weights: Whether to return attention weights
            
        Returns:
            Tuple of (output, attention_weights)
        """
        if key is None:
            key = query
        if value is None:
            value = key
        
        B, N, C = query.shape
        M = key.shape[1]
        
        # Project to Q, K, V
        Q = self.q_proj(query)  # (B, N, C)
        K = self.k_proj(key)    # (B, M, C)
        V = self.v_proj(value)  # (B, M, C)
        
        # Reshape for multi-head attention
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, D)
        K = K.view(B, M, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, M, D)
        V = V.view(B, M, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, M, D)
        
        # Compute attention scores
        scale = math.sqrt(self.head_dim)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (B, H, N, M)
        
        # Generate and apply guided attention mask if positions are provided
        if query_pos is not None and key_pos is not None and anchor_pos is not None:
            guided_mask = self.generate_attention_mask(
                query_pos, key_pos, anchor_pos
            )  # (B, N, M)
            
            # Expand for heads
            guided_mask = guided_mask.unsqueeze(1)  # (B, 1, N, M)
            
            # Apply guided mask with learnable scale
            attn_scores = attn_scores + self.guide_scale * torch.log(guided_mask + self.epsilon)
        
        # Apply additional attention mask if provided
        if attn_mask is not None:
            if attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)  # (B, 1, N, M)
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))
        
        # Apply softmax
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, H, N, M)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, V)  # (B, H, N, D)
        
        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(B, N, C)  # (B, N, C)
        output = self.out_proj(output)
        output = self.dropout(output)
        
        if need_weights:
            # Average attention weights across heads
            attn_weights = attn_weights.mean(dim=1)  # (B, N, M)
            return output, attn_weights
        else:
            return output, None


class GCHABlock(nn.Module):
    """
    GCHA Transformer Block with feed-forward network.
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        n_anchors: Number of anchor points
        epsilon: Small value for numerical stability
        mlp_ratio: Ratio of MLP hidden dim to embedding dim
        dropout: Dropout probability
        activation: Activation function
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        n_anchors: int = 64,
        epsilon: float = 1e-6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = GCHAAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            n_anchors=n_anchors,
            epsilon=epsilon,
            dropout=dropout
        )
        
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            self._get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def _get_activation(self, activation: str):
        """Get activation function by name."""
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'gelu':
            return nn.GELU()
        elif activation == 'silu':
            return nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(
        self,
        x: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
        anchor_pos: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of GCHA block.
        
        Args:
            x: Input tensor (B, N, C)
            pos: Position encoding (B, N, 2)
            anchor_pos: Anchor positions (n_anchors, 2)
            attn_mask: Attention mask
            
        Returns:
            torch.Tensor: Output tensor (B, N, C)
        """
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(
            x_norm, x_norm, x_norm,
            query_pos=pos,
            key_pos=pos,
            anchor_pos=anchor_pos,
            attn_mask=attn_mask
        )
        x = x + attn_out
        
        # Feed-forward with residual
        x = x + self.mlp(self.norm2(x))
        
        return x
