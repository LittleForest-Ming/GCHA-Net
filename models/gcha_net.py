"""
GCHA-Net: Geometry-Constrained Highway Attention Network

This module implements the GCHA-Net architecture with:
- ResNet50 backbone with Feature Pyramid Network (FPN)
- Geometry-Constrained Attention mechanism
- Dual-head architecture for anchor classification and parameter regression
GCHA-Net: Guided Cross-Hierarchical Attention Network.

Main model definition integrating all components including GCHA attention,
anchor generation, and hierarchical feature processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from typing import Optional, Tuple


class FeaturePyramidNetwork(nn.Module):
    """
    Feature Pyramid Network (FPN) that takes multi-scale features from ResNet
    and produces a unified feature map.
    """
    
    def __init__(self, in_channels_list=None, out_channels=256):
        """
        Args:
            in_channels_list: List of channels from ResNet stages (C2, C3, C4, C5)
            out_channels: Output channels for unified feature map
        """
        super().__init__()
        if in_channels_list is None:
            in_channels_list = [256, 512, 1024, 2048]
        self.out_channels = out_channels
        
        # Lateral connections (1x1 convolutions to reduce channels)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            for in_channels in in_channels_list
        ])
        
        # Output convolutions (3x3 convolutions after upsampling and addition)
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels_list
        ])
        
        # Final fusion layer to merge FPN outputs into unified feature map
        self.fusion_conv = nn.Conv2d(
            out_channels * len(in_channels_list), 
            out_channels, 
            kernel_size=1
        )
        
    def forward(self, features):
        """
        Args:
            features: List of feature maps from ResNet [C2, C3, C4, C5]
        
        Returns:
            Unified feature map of shape (B, out_channels, H, W)
        """
        # Build lateral connections
        laterals = [
            lateral_conv(features[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        
        # Top-down pathway with lateral connections
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], 
                size=laterals[i - 1].shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        
        # Apply output convolutions
        fpn_features = [
            output_conv(laterals[i])
            for i, output_conv in enumerate(self.output_convs)
        ]
        
        # Resize all FPN outputs to the same size (using the largest resolution)
        target_size = fpn_features[0].shape[-2:]
        resized_features = [fpn_features[0]]
        for feat in fpn_features[1:]:
            resized_features.append(
                F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            )
        
        # Concatenate and fuse
        concatenated = torch.cat(resized_features, dim=1)
        unified_features = self.fusion_conv(concatenated)
        
        return unified_features


class GeometryConstrainedAttention(nn.Module):
    """
    Geometry-Constrained Attention layer that replaces standard cross-attention.
    Uses a static boolean mask M_geo to prevent attention to pixels outside
    a polynomial trajectory region.
    """
    
    def __init__(self, embed_dim=256, num_heads=8, dropout=0.1):
        """
        Args:
            embed_dim: Embedding dimension for attention
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.scale = self.head_dim ** -0.5
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, geometry_mask: Optional[torch.Tensor] = None):
        """
        Forward pass with geometry-constrained attention.
        
        Args:
            query: Query tensor of shape (B, N_q, embed_dim)
            key: Key tensor of shape (B, N_k, embed_dim)
            value: Value tensor of shape (B, N_k, embed_dim)
            geometry_mask: Boolean mask M_geo of shape (B, N_q, N_k) or (N_q, N_k)
                          True indicates positions to KEEP, False indicates positions to MASK
        
        Returns:
            Output tensor of shape (B, N_q, embed_dim)
        """
        B, N_q, _ = query.shape
        N_k = key.shape[1]
        
        # Project Q, K, V
        Q = self.q_proj(query)  # (B, N_q, embed_dim)
        K = self.k_proj(key)     # (B, N_k, embed_dim)
        V = self.v_proj(value)   # (B, N_k, embed_dim)
        
        # Reshape for multi-head attention
        Q = Q.reshape(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, N_q, head_dim)
        K = K.reshape(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, N_k, head_dim)
        V = V.reshape(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, N_k, head_dim)
        
        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, num_heads, N_q, N_k)
        
        # Apply geometry mask if provided
        if geometry_mask is not None:
            # Ensure mask is on the same device
            geometry_mask = geometry_mask.to(attn_scores.device)
            
            # Handle 2D mask (N_q, N_k) or 3D mask (B, N_q, N_k)
            if geometry_mask.dim() == 2:
                geometry_mask = geometry_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, N_q, N_k)
            elif geometry_mask.dim() == 3:
                geometry_mask = geometry_mask.unsqueeze(1)  # (B, 1, N_q, N_k)
            
            # Apply mask: set masked positions to large negative value
            # geometry_mask: True = keep, False = mask out
            attn_scores = attn_scores.masked_fill(~geometry_mask, float('-inf'))
        
        # Softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, num_heads, N_q, N_k)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # (B, num_heads, N_q, head_dim)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).reshape(B, N_q, self.embed_dim)
        output = self.out_proj(attn_output)
        
        return output


class GCHADecoder(nn.Module):
    """
    GCHA Decoder that uses Geometry-Constrained Attention layers.
    """
    
    def __init__(self, embed_dim=256, num_heads=8, num_layers=6, dropout=0.1):
        """
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of decoder layers
            dropout: Dropout probability
        """
        super().__init__()
        self.embed_dim = embed_dim
        
        # Stack of GCHA layers
        self.layers = nn.ModuleList([
            GCHADecoderLayer(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, tgt, memory, geometry_mask: Optional[torch.Tensor] = None):
        """
        Args:
            tgt: Target embeddings (B, N_tgt, embed_dim)
            memory: Memory from encoder (B, N_mem, embed_dim)
            geometry_mask: Geometry constraint mask (B, N_tgt, N_mem) or (N_tgt, N_mem)
        
        Returns:
            Decoded features (B, N_tgt, embed_dim)
        """
        output = tgt
        
        for layer in self.layers:
            output = layer(output, memory, geometry_mask)
        
        output = self.norm(output)
        
        return output


class GCHADecoderLayer(nn.Module):
    """
    Single GCHA Decoder layer with self-attention and geometry-constrained cross-attention.
    """
    
    def __init__(self, embed_dim=256, num_heads=8, dropout=0.1, ffn_dim=2048):
        """
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            ffn_dim: Feed-forward network hidden dimension
        """
        super().__init__()
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        # Geometry-constrained cross-attention
        self.cross_attn = GeometryConstrainedAttention(embed_dim, num_heads, dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, tgt, memory, geometry_mask: Optional[torch.Tensor] = None):
        """
        Args:
            tgt: Target embeddings (B, N_tgt, embed_dim)
            memory: Memory from encoder (B, N_mem, embed_dim)
            geometry_mask: Geometry constraint mask
        
        Returns:
            Updated target embeddings (B, N_tgt, embed_dim)
        """
        # Self-attention
        tgt2, _ = self.self_attn(tgt, tgt, tgt)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm1(tgt)
        
        # Geometry-constrained cross-attention
        tgt2 = self.cross_attn(tgt, memory, memory, geometry_mask)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm2(tgt)
        
        # Feed-forward
        tgt2 = self.ffn(tgt)
        tgt = tgt + tgt2
        tgt = self.norm3(tgt)
        
        return tgt


class AnchorClassificationHead(nn.Module):
    """
    MLP head for binary anchor classification.
    """
    
    def __init__(self, embed_dim=256, hidden_dim=512, num_layers=3):
        """
        Args:
            embed_dim: Input embedding dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of MLP layers
        """
        super().__init__()
        
        layers = []
        in_dim = embed_dim
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            in_dim = hidden_dim
        
        # Final layer outputs single logit for binary classification
        layers.append(nn.Linear(in_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Args:
            x: Input features (B, N, embed_dim)
        
        Returns:
            Classification logits (B, N, 1)
        """
        return self.mlp(x)


class ParameterRegressionHead(nn.Module):
    """
    MLP head for parameter regression (offsets Δk, Δm, Δb).
    """
    
    def __init__(self, embed_dim=256, hidden_dim=512, num_layers=3, num_params=3):
        """
        Args:
            embed_dim: Input embedding dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of MLP layers
            num_params: Number of parameters to regress (default: 3 for Δk, Δm, Δb)
        """
        super().__init__()
        
        layers = []
        in_dim = embed_dim
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            in_dim = hidden_dim
        
        # Final layer outputs num_params values
        layers.append(nn.Linear(in_dim, num_params))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Args:
            x: Input features (B, N, embed_dim)
        
        Returns:
            Parameter offsets (B, N, num_params) for [Δk, Δm, Δb]
        """
        return self.mlp(x)
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
    GCHA-Net: Geometry-Constrained Highway Attention Network
    
    Architecture:
    1. Backbone: ResNet50 with Feature Pyramid Network (FPN)
    2. GCHA Decoder: Custom decoder with geometry-constrained attention
    3. Dual heads: Anchor classification and parameter regression
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
        embed_dim=256,
        num_heads=8,
        num_decoder_layers=6,
        num_queries=100,
        dropout=0.1,
        pretrained=True
    ):
        """
        Args:
            embed_dim: Embedding dimension for decoder
            num_heads: Number of attention heads
            num_decoder_layers: Number of decoder layers
            num_queries: Number of query embeddings (anchors)
            dropout: Dropout probability
            pretrained: Whether to use pretrained ResNet50 weights
        """
        super().__init__()
        
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        
        # 1. Backbone: ResNet50
        if pretrained:
            self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            self.backbone = resnet50(weights=None)
        
        # Remove fully connected layer and average pooling
        self.backbone = nn.ModuleDict({
            'conv1': self.backbone.conv1,
            'bn1': self.backbone.bn1,
            'relu': self.backbone.relu,
            'maxpool': self.backbone.maxpool,
            'layer1': self.backbone.layer1,  # C2: 256 channels
            'layer2': self.backbone.layer2,  # C3: 512 channels
            'layer3': self.backbone.layer3,  # C4: 1024 channels
            'layer4': self.backbone.layer4,  # C5: 2048 channels
        })
        
        # 2. Feature Pyramid Network
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=embed_dim
        )
        
        # 3. Query embeddings (learnable)
        self.query_embed = nn.Embedding(num_queries, embed_dim)
        
        # 4. GCHA Decoder
        self.decoder = GCHADecoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_decoder_layers,
            dropout=dropout
        )
        
        # 5. Anchor Classification Head
        self.classification_head = AnchorClassificationHead(embed_dim)
        
        # 6. Parameter Regression Head
        self.regression_head = ParameterRegressionHead(embed_dim)
        
        # Initialize only non-backbone parameters
        self._reset_parameters()
        
    def _reset_parameters(self):
        """Initialize parameters for non-backbone modules."""
        # Initialize FPN, decoder, and head parameters
        for module in [self.fpn, self.decoder, self.classification_head, self.regression_head]:
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
                elif p.dim() == 1:
                    nn.init.zeros_(p)
        
        # Initialize query embeddings
        nn.init.normal_(self.query_embed.weight, std=0.01)
    
    def forward_backbone(self, x):
        """
        Extract multi-scale features from ResNet50 backbone.
        
        Args:
            x: Input images (B, 3, H, W)
        
        Returns:
            List of feature maps [C2, C3, C4, C5]
        """
        # Initial conv layers
        x = self.backbone['conv1'](x)
        x = self.backbone['bn1'](x)
        x = self.backbone['relu'](x)
        x = self.backbone['maxpool'](x)
        
        # ResNet stages
        c2 = self.backbone['layer1'](x)   # 1/4 resolution
        c3 = self.backbone['layer2'](c2)  # 1/8 resolution
        c4 = self.backbone['layer3'](c3)  # 1/16 resolution
        c5 = self.backbone['layer4'](c4)  # 1/32 resolution
        
        return [c2, c3, c4, c5]
    
    def forward(
        self, 
        images: torch.Tensor, 
        geometry_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through GCHA-Net.
        
        Args:
            images: Input images (B, 3, H, W)
            geometry_mask: Static boolean mask M_geo (B, num_queries, H*W) or (num_queries, H*W)
                          True indicates pixels within polynomial trajectory (to keep)
                          False indicates pixels outside trajectory (to mask out)
        
        Returns:
            Tuple of:
                - anchor_logits: Binary classification logits (B, num_queries, 1)
                - param_offsets: Parameter regression outputs (B, num_queries, 3) for [Δk, Δm, Δb]
        """
        B = images.shape[0]
        
        # 1. Extract multi-scale features from backbone
        feature_maps = self.forward_backbone(images)
        
        # 2. Get unified feature map from FPN
        unified_features = self.fpn(feature_maps)  # (B, embed_dim, H_feat, W_feat)
        
        # 3. Flatten spatial dimensions for decoder
        B, C, H_feat, W_feat = unified_features.shape
        memory = unified_features.flatten(2).permute(0, 2, 1)  # (B, H_feat*W_feat, embed_dim)
        
        # 4. Get query embeddings
        query_embed = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # (B, num_queries, embed_dim)
        
        # 5. Decode with GCHA decoder
        decoder_output = self.decoder(query_embed, memory, geometry_mask)  # (B, num_queries, embed_dim)
        
        # 6. Apply prediction heads
        anchor_logits = self.classification_head(decoder_output)  # (B, num_queries, 1)
        param_offsets = self.regression_head(decoder_output)      # (B, num_queries, 3)
        
        return anchor_logits, param_offsets
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
