"""
Old Version: Does not involve cubic offset
Example usage of the GCHA Layer

This script demonstrates how to use the Geometry-Constrained Hough Attention (GCHA) layer
in different scenarios.
"""

import torch
from modules.gcha_attention import GCHALayer


def example_basic_usage():
    """Basic usage example without geometric mask"""
    print("=" * 70)
    print("Example 1: Basic Usage (without geometric mask)")
    print("=" * 70)
    
    # Configuration
    batch_size = 4
    d_model = 256
    n_total = 50  # Number of query positions
    hw = 49  # 7x7 feature map
    num_heads = 8
    
    # Create GCHA layer
    gcha_layer = GCHALayer(
        d_model=d_model,
        n_total=n_total,
        hw=hw,
        epsilon=0.5,
        num_heads=num_heads,
        dropout=0.1
    )
    
    print(f"Created GCHA layer:")
    print(f"  - Model dimension: {d_model}")
    print(f"  - Query positions: {n_total}")
    print(f"  - Feature map size: {hw} (e.g., 7×7)")
    print(f"  - Attention heads: {num_heads}")
    
    # Create dummy data
    query = torch.randn(batch_size, n_total, d_model)
    key = torch.randn(batch_size, hw, d_model)
    value = torch.randn(batch_size, hw, d_model)
    
    # Forward pass
    output, attention_weights = gcha_layer(query, key, value)
    
    print(f"\nInput shapes:")
    print(f"  - Query: {query.shape}")
    print(f"  - Key: {key.shape}")
    print(f"  - Value: {value.shape}")
    
    print(f"\nOutput shapes:")
    print(f"  - Output: {output.shape}")
    print(f"  - Attention weights: {attention_weights.shape}")
    
    print("\n✓ Basic usage completed successfully!\n")


def example_with_geometric_mask():
    """Example with pre-computed geometric mask"""
    print("=" * 70)
    print("Example 2: Usage with Pre-computed Geometric Mask")
    print("=" * 70)
    
    # Configuration
    batch_size = 2
    d_model = 128
    n_total = 30
    hw = 25  # 5x5 feature map
    epsilon = 1.0  # Threshold for geometric constraint
    
    # Create GCHA layer
    gcha_layer = GCHALayer(
        d_model=d_model,
        n_total=n_total,
        hw=hw,
        epsilon=epsilon,
        num_heads=4
    )
    
    print(f"Created GCHA layer with epsilon={epsilon}")
    
    # Create spatial positions
    # For a 5x5 feature map, create grid positions
    h, w = 5, 5
    y_coords = torch.arange(h).float().unsqueeze(1).repeat(1, w).flatten()
    x_coords = torch.arange(w).float().unsqueeze(0).repeat(h, 1).flatten()
    x_tilde = torch.stack([x_coords, y_coords], dim=-1)  # Shape: (hw, 2)
    
    # Create predicted positions for each query
    # Here we simulate that each query predicts positions across the feature map
    x_hat = torch.randn(n_total, hw, 2) * 2.0  # Shape: (n_total, hw, 2)
    
    print(f"\nSpatial positions:")
    print(f"  - x_tilde (feature map positions): {x_tilde.shape}")
    print(f"  - x_hat (predicted positions): {x_hat.shape}")
    
    # Pre-compute and set geometric mask
    gcha_layer.set_geometric_mask(x_tilde, x_hat)
    
    print(f"\nGeometric mask computed:")
    print(f"  - Shape: {gcha_layer.M_geo.shape}")
    
    # Count zeros and -inf in the mask
    num_zeros = (gcha_layer.M_geo == 0).sum().item()
    num_inf = torch.isinf(gcha_layer.M_geo).sum().item()
    total = gcha_layer.M_geo.numel()
    
    print(f"  - Allowed connections (0): {num_zeros} ({100*num_zeros/total:.1f}%)")
    print(f"  - Blocked connections (-inf): {num_inf} ({100*num_inf/total:.1f}%)")
    
    # Create inputs
    query = torch.randn(batch_size, n_total, d_model)
    key = torch.randn(batch_size, hw, d_model)
    value = torch.randn(batch_size, hw, d_model)
    
    # Forward pass with geometric mask
    output, attention_weights = gcha_layer(query, key, value)
    
    print(f"\nForward pass with geometric mask:")
    print(f"  - Output: {output.shape}")
    print(f"  - Attention weights: {attention_weights.shape}")
    
    print("\n✓ Geometric mask example completed successfully!\n")


def example_dynamic_mask():
    """Example with dynamically computed mask during forward pass"""
    print("=" * 70)
    print("Example 3: Dynamic Mask Computation During Forward Pass")
    print("=" * 70)
    
    # Configuration
    batch_size = 1
    d_model = 64
    n_total = 10
    hw = 16  # 4x4 feature map
    epsilon = 0.8
    
    # Create GCHA layer
    gcha_layer = GCHALayer(
        d_model=d_model,
        n_total=n_total,
        hw=hw,
        epsilon=epsilon,
        num_heads=2
    )
    
    print(f"Created GCHA layer (epsilon={epsilon})")
    
    # Create spatial positions (4x4 grid)
    h, w = 4, 4
    y_coords = torch.arange(h).float().unsqueeze(1).repeat(1, w).flatten()
    x_coords = torch.arange(w).float().unsqueeze(0).repeat(h, 1).flatten()
    x_tilde = torch.stack([x_coords, y_coords], dim=-1)
    
    # Create predicted positions
    x_hat = torch.randn(n_total, hw, 2) * 1.5
    
    # Create inputs
    query = torch.randn(batch_size, n_total, d_model)
    key = torch.randn(batch_size, hw, d_model)
    value = torch.randn(batch_size, hw, d_model)
    
    # Forward pass with dynamic mask computation
    # Pass x_tilde and x_hat directly to forward()
    output, attention_weights = gcha_layer(
        query, key, value,
        x_tilde=x_tilde,
        x_hat=x_hat
    )
    
    print(f"\nDynamic mask computation:")
    print(f"  - Mask was computed during forward pass")
    print(f"  - Mask shape: {gcha_layer.M_geo.shape}")
    print(f"  - Output: {output.shape}")
    
    print("\n✓ Dynamic mask example completed successfully!\n")


def example_integration_with_transformer():
    """Example showing how to integrate GCHA into a transformer-like architecture"""
    print("=" * 70)
    print("Example 4: Integration with Transformer Architecture")
    print("=" * 70)
    
    # Typical transformer decoder scenario
    batch_size = 2
    d_model = 512
    n_queries = 100  # Number of object queries (e.g., in DETR-like models)
    h, w = 8, 8  # Feature map from encoder
    hw = h * w
    
    # Create GCHA layer as replacement for cross-attention
    gcha_cross_attention = GCHALayer(
        d_model=d_model,
        n_total=n_queries,
        hw=hw,
        epsilon=1.0,
        num_heads=8,
        dropout=0.1
    )
    
    print(f"GCHA Cross-Attention Layer:")
    print(f"  - Model dimension: {d_model}")
    print(f"  - Number of queries: {n_queries}")
    print(f"  - Encoder feature map: {h}×{w} = {hw}")
    
    # Simulated encoder output (spatial features)
    encoder_output = torch.randn(batch_size, hw, d_model)
    
    # Simulated decoder queries
    decoder_queries = torch.randn(batch_size, n_queries, d_model)
    
    # In a real scenario, you would compute x_tilde and x_hat based on:
    # - x_tilde: spatial grid positions of encoder features
    # - x_hat: predicted positions from Hough voting or similar
    
    # Create grid positions for encoder features
    y_coords = torch.arange(h).float().unsqueeze(1).repeat(1, w).flatten()
    x_coords = torch.arange(w).float().unsqueeze(0).repeat(h, 1).flatten()
    x_tilde = torch.stack([x_coords, y_coords], dim=-1)
    
    # Simulate predicted positions (in practice, these come from Hough transform)
    x_hat = torch.randn(n_queries, hw, 2) * 3.0
    
    # Set geometric mask
    gcha_cross_attention.set_geometric_mask(x_tilde, x_hat)
    
    # Cross-attention: queries attend to encoder features
    cross_attention_output, attn_weights = gcha_cross_attention(
        query=decoder_queries,
        key=encoder_output,
        value=encoder_output
    )
    
    print(f"\nCross-Attention Output:")
    print(f"  - Shape: {cross_attention_output.shape}")
    print(f"  - Expected: (batch={batch_size}, queries={n_queries}, d_model={d_model})")
    
    print("\n✓ Transformer integration example completed successfully!\n")


def main():
    """Run all examples"""
    print("\n" + "=" * 70)
    print("GCHA Layer Examples")
    print("=" * 70 + "\n")
    
    example_basic_usage()
    example_with_geometric_mask()
    example_dynamic_mask()
    example_integration_with_transformer()
    
    print("=" * 70)
    print("All examples completed successfully!")
Example usage of GCHA-Net for highway lane detection.

This script demonstrates:
1. Creating a GCHA-Net model
2. Using it with and without geometry masks
3. Interpreting the outputs
"""

import torch
import torch.nn as nn
from models import GCHANet


def create_polynomial_trajectory_mask(batch_size, num_queries, spatial_h, spatial_w, 
                                      trajectory_width=0.3):
    """
    Create a simple polynomial trajectory mask for demonstration.
    
    In practice, this would be computed based on actual trajectory polynomials.
    
    Args:
        batch_size: Batch size
        num_queries: Number of query anchors
        spatial_h: Height of feature map
        spatial_w: Width of feature map
        trajectory_width: Relative width of trajectory region (0-1)
    
    Returns:
        Boolean mask of shape (batch_size, num_queries, spatial_h * spatial_w)
    """
    mask = torch.zeros(batch_size, num_queries, spatial_h * spatial_w, dtype=torch.bool)
    
    # Create masks for different trajectory regions for each query
    # This is a simplified example - in practice, you'd compute this from polynomial parameters
    for b in range(batch_size):
        for q in range(num_queries):
            # Create a vertical trajectory region for this query
            # Each query might correspond to different lane positions
            center_x = int((q / num_queries) * spatial_w)
            width = int(trajectory_width * spatial_w)
            
            for h in range(spatial_h):
                for w in range(spatial_w):
                    # Simple vertical band trajectory
                    if abs(w - center_x) < width:
                        idx = h * spatial_w + w
                        mask[b, q, idx] = True
    
    return mask


def main():
    print("=" * 70)
    print("GCHA-Net Example Usage")
    print("=" * 70)
    print()
    
    # Configuration
    batch_size = 4
    img_height = 224
    img_width = 224
    num_queries = 50
    
    # 1. Create GCHA-Net model
    print("1. Creating GCHA-Net model...")
    model = GCHANet(
        embed_dim=256,
        num_heads=8,
        num_decoder_layers=6,
        num_queries=num_queries,
        pretrained=False  # Set to True to use pretrained ResNet50
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print()
    
    # 2. Create sample input images
    print("2. Creating sample input...")
    images = torch.randn(batch_size, 3, img_height, img_width)
    print(f"   Input shape: {images.shape}")
    print()
    
    # 3. Forward pass without geometry mask
    print("3. Forward pass WITHOUT geometry mask...")
    model.eval()
    with torch.no_grad():
        anchor_logits, param_offsets = model(images)
    
    print(f"   Output shapes:")
    print(f"   - Anchor logits: {anchor_logits.shape}")
    print(f"   - Parameter offsets: {param_offsets.shape}")
    print()
    
    # Interpret outputs
    print("   Interpreting outputs:")
    anchor_probs = torch.sigmoid(anchor_logits)  # Convert logits to probabilities
    print(f"   - Anchor probabilities (first sample, first 5 queries):")
    print(f"     {anchor_probs[0, :5, 0].numpy()}")
    print(f"   - Parameter offsets [Δk, Δm, Δb] (first sample, first query):")
    print(f"     {param_offsets[0, 0, :].numpy()}")
    print()
    
    # 4. Forward pass WITH geometry mask
    print("4. Forward pass WITH geometry mask...")
    
    # First, get the feature map spatial dimensions
    with torch.no_grad():
        feature_maps = model.forward_backbone(images)
        unified_features = model.fpn(feature_maps)
        _, _, H_feat, W_feat = unified_features.shape
    
    print(f"   Feature map size: {H_feat}x{W_feat}")
    
    # Create geometry mask
    geometry_mask = create_polynomial_trajectory_mask(
        batch_size, num_queries, H_feat, W_feat, trajectory_width=0.2
    )
    print(f"   Geometry mask shape: {geometry_mask.shape}")
    print(f"   Mask density (% True): {geometry_mask.float().mean().item() * 100:.1f}%")
    print()
    
    # Forward pass with mask
    with torch.no_grad():
        anchor_logits_masked, param_offsets_masked = model(images, geometry_mask=geometry_mask)
    
    print(f"   Output shapes:")
    print(f"   - Anchor logits: {anchor_logits_masked.shape}")
    print(f"   - Parameter offsets: {param_offsets_masked.shape}")
    print()
    
    # 5. Compare outputs with and without mask
    print("5. Comparing masked vs unmasked outputs...")
    diff_logits = torch.abs(anchor_logits - anchor_logits_masked).mean().item()
    diff_params = torch.abs(param_offsets - param_offsets_masked).mean().item()
    print(f"   Mean absolute difference in anchor logits: {diff_logits:.6f}")
    print(f"   Mean absolute difference in param offsets: {diff_params:.6f}")
    print()
    
    # 6. Training example
    print("6. Example training step...")
    model.train()
    
    # Dummy targets
    target_anchors = torch.randint(0, 2, (batch_size, num_queries, 1)).float()
    target_params = torch.randn(batch_size, num_queries, 3)
    
    # Forward pass
    pred_logits, pred_params = model(images, geometry_mask=geometry_mask)
    
    # Compute losses
    cls_loss_fn = nn.BCEWithLogitsLoss()
    reg_loss_fn = nn.SmoothL1Loss()
    
    cls_loss = cls_loss_fn(pred_logits, target_anchors)
    reg_loss = reg_loss_fn(pred_params, target_params)
    total_loss = cls_loss + reg_loss
    
    print(f"   Classification loss: {cls_loss.item():.4f}")
    print(f"   Regression loss: {reg_loss.item():.4f}")
    print(f"   Total loss: {total_loss.item():.4f}")
    print()
    
    print("=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
