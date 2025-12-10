"""
GCHA (Guided Cross-Hierarchical Attention) implementation.

This module implements the GCHA attention layer and mask generation logic.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
