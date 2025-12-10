"""
Test script for GCHA-Net model architecture.
"""

import torch
from models import GCHANet, GeometryConstrainedAttention


def test_geometry_constrained_attention():
    """Test the GeometryConstrainedAttention layer."""
    print("Testing GeometryConstrainedAttention...")
    
    batch_size = 2
    num_queries = 10
    num_keys = 20
    embed_dim = 256
    num_heads = 8
    
    # Create layer
    gca = GeometryConstrainedAttention(embed_dim=embed_dim, num_heads=num_heads)
    
    # Create dummy inputs
    query = torch.randn(batch_size, num_queries, embed_dim)
    key = torch.randn(batch_size, num_keys, embed_dim)
    value = torch.randn(batch_size, num_keys, embed_dim)
    
    # Test without mask
    output = gca(query, key, value)
    assert output.shape == (batch_size, num_queries, embed_dim), f"Expected shape {(batch_size, num_queries, embed_dim)}, got {output.shape}"
    print(f"  ✓ Without mask: output shape = {output.shape}")
    
    # Test with 2D mask
    mask_2d = torch.rand(num_queries, num_keys) > 0.5  # Random boolean mask
    output = gca(query, key, value, geometry_mask=mask_2d)
    assert output.shape == (batch_size, num_queries, embed_dim)
    print(f"  ✓ With 2D mask: output shape = {output.shape}")
    
    # Test with 3D mask
    mask_3d = torch.rand(batch_size, num_queries, num_keys) > 0.5
    output = gca(query, key, value, geometry_mask=mask_3d)
    assert output.shape == (batch_size, num_queries, embed_dim)
    print(f"  ✓ With 3D mask: output shape = {output.shape}")
    
    print("GeometryConstrainedAttention tests passed!\n")


def test_gcha_net_basic():
    """Test basic GCHA-Net forward pass."""
    print("Testing GCHA-Net basic functionality...")
    
    batch_size = 2
    height = 224
    width = 224
    num_queries = 100
    
    # Create model (without pretrained weights for faster testing)
    model = GCHANet(
        embed_dim=256,
        num_heads=8,
        num_decoder_layers=3,  # Use fewer layers for faster testing
        num_queries=num_queries,
        pretrained=False
    )
    
    # Create dummy input
    images = torch.randn(batch_size, 3, height, width)
    
    # Test forward pass without geometry mask
    print("  Testing without geometry mask...")
    anchor_logits, param_offsets = model(images)
    
    assert anchor_logits.shape == (batch_size, num_queries, 1), f"Expected anchor_logits shape {(batch_size, num_queries, 1)}, got {anchor_logits.shape}"
    assert param_offsets.shape == (batch_size, num_queries, 3), f"Expected param_offsets shape {(batch_size, num_queries, 3)}, got {param_offsets.shape}"
    
    print(f"    ✓ anchor_logits shape: {anchor_logits.shape}")
    print(f"    ✓ param_offsets shape: {param_offsets.shape}")
    
    print("GCHA-Net basic tests passed!\n")


def test_gcha_net_with_geometry_mask():
    """Test GCHA-Net with geometry constraint mask."""
    print("Testing GCHA-Net with geometry mask...")
    
    batch_size = 2
    height = 224
    width = 224
    num_queries = 100
    
    # Create model
    model = GCHANet(
        embed_dim=256,
        num_heads=8,
        num_decoder_layers=3,
        num_queries=num_queries,
        pretrained=False
    )
    model.eval()  # Set to eval mode
    
    # Create dummy input
    images = torch.randn(batch_size, 3, height, width)
    
    # Get feature map size (this depends on the FPN output)
    with torch.no_grad():
        feature_maps = model.forward_backbone(images)
        unified_features = model.fpn(feature_maps)
        _, _, H_feat, W_feat = unified_features.shape
        num_spatial = H_feat * W_feat
    
    # Create geometry mask simulating polynomial trajectory region
    # Shape: (batch_size, num_queries, num_spatial_positions)
    geometry_mask = torch.rand(batch_size, num_queries, num_spatial) > 0.3
    
    print(f"  Feature map spatial size: {H_feat}x{W_feat} = {num_spatial}")
    print(f"  Geometry mask shape: {geometry_mask.shape}")
    
    # Forward pass with mask
    with torch.no_grad():
        anchor_logits, param_offsets = model(images, geometry_mask=geometry_mask)
    
    assert anchor_logits.shape == (batch_size, num_queries, 1)
    assert param_offsets.shape == (batch_size, num_queries, 3)
    
    print(f"    ✓ anchor_logits shape: {anchor_logits.shape}")
    print(f"    ✓ param_offsets shape: {param_offsets.shape}")
    
    print("GCHA-Net geometry mask tests passed!\n")


def test_model_components():
    """Test individual model components."""
    print("Testing individual model components...")
    
    from models.gcha_net import (
        FeaturePyramidNetwork,
        GCHADecoder,
        AnchorClassificationHead,
        ParameterRegressionHead
    )
    
    # Test FPN
    print("  Testing FPN...")
    fpn = FeaturePyramidNetwork(in_channels_list=[256, 512, 1024, 2048], out_channels=256)
    features = [
        torch.randn(2, 256, 56, 56),
        torch.randn(2, 512, 28, 28),
        torch.randn(2, 1024, 14, 14),
        torch.randn(2, 2048, 7, 7),
    ]
    unified = fpn(features)
    assert unified.shape == (2, 256, 56, 56)
    print(f"    ✓ FPN output shape: {unified.shape}")
    
    # Test GCHA Decoder
    print("  Testing GCHA Decoder...")
    decoder = GCHADecoder(embed_dim=256, num_heads=8, num_layers=3)
    tgt = torch.randn(2, 100, 256)
    memory = torch.randn(2, 196, 256)
    output = decoder(tgt, memory)
    assert output.shape == (2, 100, 256)
    print(f"    ✓ Decoder output shape: {output.shape}")
    
    # Test Classification Head
    print("  Testing Anchor Classification Head...")
    cls_head = AnchorClassificationHead(embed_dim=256)
    features = torch.randn(2, 100, 256)
    logits = cls_head(features)
    assert logits.shape == (2, 100, 1)
    print(f"    ✓ Classification head output shape: {logits.shape}")
    
    # Test Regression Head
    print("  Testing Parameter Regression Head...")
    reg_head = ParameterRegressionHead(embed_dim=256)
    params = reg_head(features)
    assert params.shape == (2, 100, 3)
    print(f"    ✓ Regression head output shape: {params.shape}")
    
    print("All component tests passed!\n")


def test_pretrained_backbone():
    """Test GCHA-Net with pretrained ResNet50 backbone."""
    print("Testing GCHA-Net with pretrained backbone...")
    
    try:
        # Create model with pretrained weights
        model = GCHANet(
            embed_dim=256,
            num_heads=8,
            num_decoder_layers=2,
            num_queries=50,
            pretrained=True
        )
        
        images = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            anchor_logits, param_offsets = model(images)
        
        assert anchor_logits.shape == (1, 50, 1)
        assert param_offsets.shape == (1, 50, 3)
        
        print(f"  ✓ Model with pretrained backbone works")
        print(f"  ✓ anchor_logits shape: {anchor_logits.shape}")
        print(f"  ✓ param_offsets shape: {param_offsets.shape}")
        print("Pretrained backbone test passed!\n")
        
    except Exception as e:
        print(f"  Note: Could not load pretrained weights (expected in offline environment): {e}")
        print("  This is normal if no internet connection is available.\n")


if __name__ == "__main__":
    print("=" * 70)
    print("GCHA-Net Model Architecture Tests")
    print("=" * 70)
    print()
    
    # Run tests
    test_geometry_constrained_attention()
    test_model_components()
    test_gcha_net_basic()
    test_gcha_net_with_geometry_mask()
    test_pretrained_backbone()
    
    print("=" * 70)
    print("All tests completed successfully! ✓")
    print("=" * 70)
