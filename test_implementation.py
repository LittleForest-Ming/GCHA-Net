"""Simple validation script for GCHA-Net implementation.

This script tests basic functionality of the implemented components.
"""

import torch
import sys

print("Testing GCHA-Net Implementation...")
print("=" * 60)

# Test 1: Import modules
print("\n1. Testing imports...")
try:
    from models.gcha_net import GCHANet, GeometryConstrainedAttention, GCHADecoder
    from utils.geometry import generate_anchors, generate_geometric_mask, polynomial_fit_from_points
    from datasets.agroscapes import DummyDataset
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Generate anchors
print("\n2. Testing anchor generation...")
try:
    anchors = generate_anchors()
    assert anchors.shape == (405, 3), f"Expected shape (405, 3), got {anchors.shape}"
    assert anchors.dtype == torch.float32
    print(f"✓ Anchors generated: shape={anchors.shape}, dtype={anchors.dtype}")
    print(f"  Sample anchor: k={anchors[0, 0]:.3f}, m={anchors[0, 1]:.3f}, b={anchors[0, 2]:.3f}")
except Exception as e:
    print(f"✗ Anchor generation failed: {e}")
    sys.exit(1)

# Test 3: Generate geometric mask
print("\n3. Testing geometric mask generation...")
try:
    H, W = 36, 100  # Feature map size
    test_anchors = anchors[:5]  # Test with 5 anchors
    masks = generate_geometric_mask(H, W, test_anchors, epsilon=0.05)
    assert masks.shape == (5, H, W), f"Expected shape (5, {H}, {W}), got {masks.shape}"
    
    # Check mask values
    num_valid = (masks == 0).sum()
    num_invalid = torch.isinf(masks).sum()
    print(f"✓ Geometric masks generated: shape={masks.shape}")
    print(f"  Valid pixels: {num_valid}, Invalid pixels: {num_invalid}")
except Exception as e:
    print(f"✗ Geometric mask generation failed: {e}")
    sys.exit(1)

# Test 4: Polynomial fitting
print("\n4. Testing polynomial fitting...")
try:
    # Create sample points
    points = [[100, 200], [110, 180], [120, 160], [130, 140]]
    params = polynomial_fit_from_points(points, 288, 800)
    assert params.shape == (3,), f"Expected shape (3,), got {params.shape}"
    print(f"✓ Polynomial fit successful: k={params[0]:.3f}, m={params[1]:.3f}, b={params[2]:.3f}")
except Exception as e:
    print(f"✗ Polynomial fitting failed: {e}")
    sys.exit(1)

# Test 5: GeometryConstrainedAttention layer
print("\n5. Testing GeometryConstrainedAttention layer...")
try:
    batch_size = 2
    num_queries = 10
    seq_len = 100
    embed_dim = 256
    
    gca_layer = GeometryConstrainedAttention(embed_dim=embed_dim, num_heads=8)
    query = torch.randn(batch_size, num_queries, embed_dim)
    key = torch.randn(batch_size, seq_len, embed_dim)
    value = torch.randn(batch_size, seq_len, embed_dim)
    geo_mask = torch.zeros(num_queries, seq_len)
    
    output = gca_layer(query, key, value, geo_mask)
    assert output.shape == (batch_size, num_queries, embed_dim)
    print(f"✓ GeometryConstrainedAttention forward pass successful: {output.shape}")
except Exception as e:
    print(f"✗ GeometryConstrainedAttention failed: {e}")
    sys.exit(1)

# Test 6: GCHA Decoder
print("\n6. Testing GCHA Decoder...")
try:
    decoder = GCHADecoder(embed_dim=256, num_heads=8, num_layers=3)
    queries = torch.randn(batch_size, num_queries, 256)
    features = torch.randn(batch_size, seq_len, 256)
    geo_masks = torch.zeros(num_queries, seq_len)
    
    output = decoder(queries, features, geo_masks)
    assert output.shape == (batch_size, num_queries, 256)
    print(f"✓ GCHA Decoder forward pass successful: {output.shape}")
except Exception as e:
    print(f"✗ GCHA Decoder failed: {e}")
    sys.exit(1)

# Test 7: Full GCHA-Net model
print("\n7. Testing full GCHA-Net model...")
try:
    model = GCHANet(
        num_anchors=405,
        embed_dim=256,
        num_decoder_layers=3,
        num_heads=8,
        dropout=0.1,
        epsilon=0.05
    )
    
    # Test forward pass
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 288, 800)
    
    print("  Running forward pass (this may take a moment)...")
    cls_logits, reg_deltas = model(input_tensor)
    
    assert cls_logits.shape == (batch_size, 405), f"Expected cls shape (2, 405), got {cls_logits.shape}"
    assert reg_deltas.shape == (batch_size, 405, 3), f"Expected reg shape (2, 405, 3), got {reg_deltas.shape}"
    
    print(f"✓ GCHA-Net forward pass successful")
    print(f"  Classification logits: {cls_logits.shape}")
    print(f"  Regression deltas: {reg_deltas.shape}")
    
    # Test get_refined_anchors
    refined = model.get_refined_anchors(reg_deltas)
    assert refined.shape == (batch_size, 405, 3)
    print(f"✓ Anchor refinement successful: {refined.shape}")
    
except Exception as e:
    print(f"✗ GCHA-Net model failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: Dummy Dataset
print("\n8. Testing Dummy Dataset...")
try:
    dataset = DummyDataset(num_samples=10, image_height=288, image_width=800, max_lanes=4)
    assert len(dataset) == 10
    
    image, targets = dataset[0]
    assert image.shape == (3, 288, 800)
    assert targets['lane_params'].shape == (4, 3)
    assert targets['lane_valid'].shape == (4,)
    
    print(f"✓ Dummy dataset working")
    print(f"  Image shape: {image.shape}")
    print(f"  Lane params shape: {targets['lane_params'].shape}")
    print(f"  Valid lanes: {targets['lane_valid'].sum().item()}")
    
except Exception as e:
    print(f"✗ Dataset failed: {e}")
    sys.exit(1)

# Test 9: DataLoader
print("\n9. Testing DataLoader...")
try:
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch_images, batch_targets = next(iter(dataloader))
    
    assert batch_images.shape == (4, 3, 288, 800)
    assert batch_targets['lane_params'].shape == (4, 4, 3)
    
    print(f"✓ DataLoader working")
    print(f"  Batch images: {batch_images.shape}")
    print(f"  Batch lane params: {batch_targets['lane_params'].shape}")
    
except Exception as e:
    print(f"✗ DataLoader failed: {e}")
    sys.exit(1)

# Summary
print("\n" + "=" * 60)
print("✓ All tests passed successfully!")
print("=" * 60)
print("\nImplementation validated. The following components are working:")
print("  • Anchor generation")
print("  • Geometric mask generation")
print("  • Polynomial fitting from points")
print("  • Geometry-Constrained Attention layer")
print("  • GCHA Decoder")
print("  • Full GCHA-Net model")
print("  • Dataset and DataLoader")
print("\nYou can now run training with:")
print("  python train.py --use_dummy --batch_size 2 --max_epochs 5")
