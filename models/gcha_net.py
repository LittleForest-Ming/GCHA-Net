"""GCHA-Net: Geometry-Constrained Hierarchical Attention Network for Lane Detection.

This module implements the GCHA-Net architecture with:
- ResNet50 + FPN backbone
- Geometry-Constrained Attention decoder
- Classification and regression heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.ops import FeaturePyramidNetwork

from utils.geometry import generate_anchors, generate_geometric_mask


class GeometryConstrainedAttention(nn.Module):
    """Geometry-Constrained Attention layer.
    
    This replaces standard cross-attention with a geometric constraint that
    prevents attention to pixels outside polynomial trajectories.
    """
    
    def __init__(self, embed_dim=256, num_heads=8, dropout=0.1):
        """Initialize the GeometryConstrainedAttention layer.
        
        Args:
            embed_dim: Dimension of input features
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, \
            "embed_dim must be divisible by num_heads"
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, query, key, value, geometric_mask=None):
        """Forward pass with geometric masking.
        
        Args:
            query: Tensor of shape (N, num_queries, embed_dim)
            key: Tensor of shape (N, H*W, embed_dim) - spatial features
            value: Tensor of shape (N, H*W, embed_dim) - spatial features
            geometric_mask: Optional tensor of shape (num_queries, H*W)
                           with 0 for valid positions, -inf for masked
                           
        Returns:
            output: Tensor of shape (N, num_queries, embed_dim)
        """
        N, num_queries, _ = query.shape
        _, seq_len, _ = key.shape
        
        # Project and reshape for multi-head attention
        q = self.q_proj(query)  # [N, num_queries, embed_dim]
        k = self.k_proj(key)    # [N, seq_len, embed_dim]
        v = self.v_proj(value)  # [N, seq_len, embed_dim]
        
        # Reshape to [N, num_heads, num_queries/seq_len, head_dim]
        q = q.view(N, num_queries, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(N, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(N, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # [N, num_heads, num_queries, seq_len]
        
        # Apply geometric mask if provided
        if geometric_mask is not None:
            # geometric_mask shape: [num_queries, seq_len]
            # Expand for batch and heads: [1, 1, num_queries, seq_len]
            mask_expanded = geometric_mask.unsqueeze(0).unsqueeze(0)
            attn = attn + mask_expanded
        
        # Apply softmax
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # [N, num_heads, num_queries, head_dim]
        
        # Reshape back
        out = out.transpose(1, 2).contiguous().view(N, num_queries, self.embed_dim)
        
        # Output projection
        out = self.out_proj(out)
        
        return out


class GCHADecoder(nn.Module):
    """GCHA Decoder with Geometry-Constrained Attention."""
    
    def __init__(self, embed_dim=256, num_heads=8, num_layers=3, dropout=0.1):
        """Initialize the GCHA Decoder.
        
        Args:
            embed_dim: Dimension of features
            num_heads: Number of attention heads
            num_layers: Number of decoder layers
            dropout: Dropout probability
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        
        # Decoder layers
        self.layers = nn.ModuleList([
            GCHADecoderLayer(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, queries, features, geometric_masks=None):
        """Forward pass through the decoder.
        
        Args:
            queries: Tensor of shape (N, num_queries, embed_dim) - anchor queries
            features: Tensor of shape (N, H*W, embed_dim) - spatial features
            geometric_masks: Optional tensor of shape (num_queries, H*W)
            
        Returns:
            output: Tensor of shape (N, num_queries, embed_dim)
        """
        output = queries
        
        for layer in self.layers:
            output = layer(output, features, geometric_masks)
        
        output = self.norm(output)
        
        return output


class GCHADecoderLayer(nn.Module):
    """Single GCHA Decoder Layer."""
    
    def __init__(self, embed_dim=256, num_heads=8, dropout=0.1, dim_feedforward=1024):
        """Initialize a decoder layer.
        
        Args:
            embed_dim: Dimension of features
            num_heads: Number of attention heads
            dropout: Dropout probability
            dim_feedforward: Dimension of feedforward network
        """
        super().__init__()
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        # Cross-attention with geometric constraint
        self.cross_attn = GeometryConstrainedAttention(embed_dim, num_heads, dropout)
        
        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, queries, features, geometric_mask=None):
        """Forward pass through decoder layer.
        
        Args:
            queries: Tensor of shape (N, num_queries, embed_dim)
            features: Tensor of shape (N, H*W, embed_dim)
            geometric_mask: Optional tensor of shape (num_queries, H*W)
            
        Returns:
            output: Tensor of shape (N, num_queries, embed_dim)
        """
        # Self-attention
        q2, _ = self.self_attn(queries, queries, queries)
        queries = queries + self.dropout(q2)
        queries = self.norm1(queries)
        
        # Cross-attention with geometric constraint
        q2 = self.cross_attn(queries, features, features, geometric_mask)
        queries = queries + self.dropout(q2)
        queries = self.norm2(queries)
        
        # Feedforward
        q2 = self.ffn(queries)
        queries = queries + q2
        queries = self.norm3(queries)
        
        return queries


class GCHANet(nn.Module):
    """GCHA-Net: Geometry-Constrained Hierarchical Attention Network.
    
    Architecture:
    - Backbone: ResNet50 + FPN
    - Decoder: GCHA Decoder with geometric constraints
    - Heads: Classification and regression MLPs
    """
    
    def __init__(self, num_anchors=405, embed_dim=256, num_decoder_layers=3,
                 num_heads=8, dropout=0.1, epsilon=0.05):
        """Initialize GCHA-Net.
        
        Args:
            num_anchors: Number of anchor polynomials (default: 5*9*9=405)
            embed_dim: Dimension of feature embeddings
            num_decoder_layers: Number of decoder layers
            num_heads: Number of attention heads
            dropout: Dropout probability
            epsilon: Distance threshold for geometric masking
        """
        super().__init__()
        
        self.num_anchors = num_anchors
        self.embed_dim = embed_dim
        self.epsilon = epsilon
        
        # Generate anchors (k, m, b parameters)
        self.register_buffer('anchors', generate_anchors())
        assert self.anchors.shape[0] == num_anchors, \
            f"Number of anchors mismatch: {self.anchors.shape[0]} vs {num_anchors}"
        
        # Backbone: ResNet50 with FPN
        self.backbone = self._initialize_backbone()
        
        # Feature projection to embed_dim
        self.feature_proj = nn.Conv2d(256, embed_dim, kernel_size=1)
        
        # Learnable anchor queries
        self.anchor_queries = nn.Embedding(num_anchors, embed_dim)
        
        # GCHA Decoder
        self.decoder = GCHADecoder(embed_dim, num_heads, num_decoder_layers, dropout)
        
        # Classification head (binary)
        self.cls_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1)  # Binary classification
        )
        
        # Regression head (delta_k, delta_m, delta_b)
        self.reg_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 3)  # 3 parameters: delta_k, delta_m, delta_b
        )
        
    def _initialize_backbone(self):
        """Initialize ResNet50 + FPN backbone layers."""
        # Load pretrained ResNet50
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        
        # Extract feature extraction layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1  # 256 channels
        self.layer2 = resnet.layer2  # 512 channels
        self.layer3 = resnet.layer3  # 1024 channels
        self.layer4 = resnet.layer4  # 2048 channels
        
        # FPN
        in_channels_list = [256, 512, 1024, 2048]
        out_channels = 256
        self.fpn = FeaturePyramidNetwork(in_channels_list, out_channels)
    
    def extract_features(self, x):
        """Extract features using ResNet50 + FPN.
        
        Args:
            x: Input image tensor of shape (N, 3, H, W)
            
        Returns:
            features: FPN feature map of shape (N, 256, H', W')
        """
        # ResNet stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # ResNet stages
        c1 = self.layer1(x)   # 1/4
        c2 = self.layer2(c1)  # 1/8
        c3 = self.layer3(c2)  # 1/16
        c4 = self.layer4(c3)  # 1/32
        
        # FPN
        features_dict = {'0': c1, '1': c2, '2': c3, '3': c4}
        fpn_features = self.fpn(features_dict)
        
        # Use the finest FPN level (typically '0')
        features = fpn_features['0']
        
        return features
    
    def forward(self, x):
        """Forward pass.
        
        Args:
            x: Input image tensor of shape (N, 3, H, W)
            
        Returns:
            cls_logits: Classification logits of shape (N, num_anchors)
            reg_deltas: Regression deltas of shape (N, num_anchors, 3)
        """
        N, _, H, W = x.shape
        
        # Extract features
        features = self.extract_features(x)  # [N, 256, H', W']
        _, _, H_feat, W_feat = features.shape
        
        # Project features
        features = self.feature_proj(features)  # [N, embed_dim, H', W']
        
        # Reshape features for attention: [N, H'*W', embed_dim]
        features_flat = features.flatten(2).transpose(1, 2)
        
        # Generate geometric masks
        geometric_masks = generate_geometric_mask(
            H_feat, W_feat, self.anchors, self.epsilon
        )  # [num_anchors, H', W']
        geometric_masks_flat = geometric_masks.flatten(1)  # [num_anchors, H'*W']
        
        # Get anchor queries
        anchor_queries = self.anchor_queries.weight.unsqueeze(0).expand(N, -1, -1)
        # [N, num_anchors, embed_dim]
        
        # Decode with GCHA decoder
        decoded_features = self.decoder(anchor_queries, features_flat, geometric_masks_flat)
        # [N, num_anchors, embed_dim]
        
        # Classification head
        cls_logits = self.cls_head(decoded_features).squeeze(-1)  # [N, num_anchors]
        
        # Regression head
        reg_deltas = self.reg_head(decoded_features)  # [N, num_anchors, 3]
        
        return cls_logits, reg_deltas
    
    def get_refined_anchors(self, reg_deltas):
        """Get refined anchor parameters by applying deltas.
        
        Args:
            reg_deltas: Regression deltas of shape (N, num_anchors, 3)
            
        Returns:
            refined_anchors: Refined parameters of shape (N, num_anchors, 3)
        """
        # anchors: [num_anchors, 3]
        # reg_deltas: [N, num_anchors, 3]
        anchors_expanded = self.anchors.unsqueeze(0)  # [1, num_anchors, 3]
        refined_anchors = anchors_expanded + reg_deltas
        
        return refined_anchors
