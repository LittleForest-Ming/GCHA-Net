"""
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
