"""
Test script to verify GCHA-Net implementation.

This script tests basic functionality of all components without requiring
a full dataset or training run.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from models.gcha_net import GCHANet, build_gcha_net
from modules.gcha_attention import GCHAAttention, GCHABlock
from utils.anchors import (
    generate_anchor_grid,
    generate_hierarchical_anchors,
    compute_anchor_distances,
    get_position_encoding
)
import yaml


def test_anchor_generation():
    """Test anchor generation utilities."""
    print("Testing anchor generation...")
    
    # Test single scale
    n_total = 64
    feature_size = (32, 32)
    anchors = generate_anchor_grid(n_total, feature_size)
    assert anchors.shape == (n_total, 2), f"Expected shape ({n_total}, 2), got {anchors.shape}"
    print(f"✓ Single scale anchors: {anchors.shape}")
    
    # Test hierarchical
    feature_sizes = [(64, 64), (32, 32), (16, 16), (8, 8)]
    hierarchical = generate_hierarchical_anchors(n_total, feature_sizes)
    assert len(hierarchical) == 4, f"Expected 4 levels, got {len(hierarchical)}"
    print(f"✓ Hierarchical anchors: {len(hierarchical)} levels")
    
    # Test distance computation
    query_pos = torch.randn(10, 2)
    anchor_pos = anchors
    distances = compute_anchor_distances(query_pos, anchor_pos)
    assert distances.shape == (10, n_total), f"Expected shape (10, {n_total}), got {distances.shape}"
    print(f"✓ Distance computation: {distances.shape}")
    
    # Test position encoding
    positions = torch.randn(100, 2)
    pos_enc = get_position_encoding(positions, d_model=256)
    assert pos_enc.shape == (100, 256), f"Expected shape (100, 256), got {pos_enc.shape}"
    print(f"✓ Position encoding: {pos_enc.shape}")
    
    print("✅ Anchor generation tests passed!\n")


def test_gcha_attention():
    """Test GCHA attention module."""
    print("Testing GCHA attention...")
    
    batch_size = 2
    seq_len = 256
    embed_dim = 128
    num_heads = 8
    n_anchors = 32
    
    # Create attention layer
    attn = GCHAAttention(embed_dim, num_heads, n_anchors)
    
    # Test forward pass
    x = torch.randn(batch_size, seq_len, embed_dim)
    pos = torch.randn(batch_size, seq_len, 2)
    anchor_pos = torch.randn(n_anchors, 2)
    
    output, weights = attn(x, query_pos=pos, key_pos=pos, anchor_pos=anchor_pos, need_weights=True)
    
    assert output.shape == (batch_size, seq_len, embed_dim), f"Expected shape {(batch_size, seq_len, embed_dim)}, got {output.shape}"
    assert weights.shape == (batch_size, seq_len, seq_len), f"Expected weights shape {(batch_size, seq_len, seq_len)}, got {weights.shape}"
    
    print(f"✓ Attention output: {output.shape}")
    print(f"✓ Attention weights: {weights.shape}")
    print("✅ GCHA attention tests passed!\n")


def test_gcha_block():
    """Test GCHA block."""
    print("Testing GCHA block...")
    
    batch_size = 2
    seq_len = 256
    embed_dim = 128
    n_anchors = 32
    
    # Create block
    block = GCHABlock(embed_dim, num_heads=8, n_anchors=n_anchors)
    
    # Test forward pass
    x = torch.randn(batch_size, seq_len, embed_dim)
    pos = torch.randn(batch_size, seq_len, 2)
    anchor_pos = torch.randn(n_anchors, 2)
    
    output = block(x, pos=pos, anchor_pos=anchor_pos)
    
    assert output.shape == (batch_size, seq_len, embed_dim), f"Expected shape {(batch_size, seq_len, embed_dim)}, got {output.shape}"
    
    print(f"✓ Block output: {output.shape}")
    print("✅ GCHA block tests passed!\n")


def test_gcha_net():
    """Test full GCHA-Net model."""
    print("Testing GCHA-Net model...")
    
    batch_size = 1
    in_channels = 3
    height, width = 64, 64  # Reduced from 256x256 to avoid memory issues
    num_classes = 19
    
    # Create model
    model = GCHANet(
        in_channels=in_channels,
        num_classes=num_classes,
        embed_dim=64,  # Reduced from 128
        num_heads=4,
        n_anchors=16,  # Reduced from 32
        num_layers=1,  # Reduced from 2
        base_channels=16,  # Reduced from 32
        num_stages=2  # Reduced from 3
    )
    
    # Test forward pass
    x = torch.randn(batch_size, in_channels, height, width)
    output = model(x)
    
    assert output.shape == (batch_size, num_classes, height, width), \
        f"Expected shape {(batch_size, num_classes, height, width)}, got {output.shape}"
    
    print(f"✓ Model output: {output.shape}")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Total parameters: {num_params:,}")
    
    print("✅ GCHA-Net model tests passed!\n")


def test_build_from_config():
    """Test building model from config."""
    print("Testing model building from config...")
    
    # Load config
    config_path = "config/default.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Modify config for smaller test model
    config['model']['embed_dim'] = 64
    config['model']['num_layers'] = 1
    config['model']['base_channels'] = 16
    config['model']['num_stages'] = 2
    config['model']['n_total'] = 16
    config['data']['image_size'] = [64, 64]
    
    # Build model
    model = build_gcha_net(config['model'])
    
    # Test forward pass
    batch_size = 1
    x = torch.randn(
        batch_size,
        config['model']['in_channels'],
        config['data']['image_size'][0],
        config['data']['image_size'][1]
    )
    
    output = model(x)
    
    expected_shape = (
        batch_size,
        config['model']['num_classes'],
        config['data']['image_size'][0],
        config['data']['image_size'][1]
    )
    
    assert output.shape == expected_shape, \
        f"Expected shape {expected_shape}, got {output.shape}"
    
    print(f"✓ Model from config output: {output.shape}")
    print("✅ Config-based building tests passed!\n")


def test_mask_generation():
    """Test attention mask generation."""
    print("Testing attention mask generation...")
    
    batch_size = 2
    n_queries = 100
    n_keys = 100
    n_anchors = 16
    
    attn = GCHAAttention(embed_dim=128, num_heads=4, n_anchors=n_anchors)
    
    query_pos = torch.randn(batch_size, n_queries, 2)
    key_pos = torch.randn(batch_size, n_keys, 2)
    anchor_pos = torch.randn(n_anchors, 2)
    
    mask = attn.generate_attention_mask(query_pos, key_pos, anchor_pos)
    
    assert mask.shape == (batch_size, n_queries, n_keys), \
        f"Expected shape {(batch_size, n_queries, n_keys)}, got {mask.shape}"
    
    # Check that mask values are reasonable
    assert (mask >= 0).all(), "Mask should have non-negative values"
    
    print(f"✓ Mask shape: {mask.shape}")
    print(f"✓ Mask value range: [{mask.min().item():.4f}, {mask.max().item():.4f}]")
    print("✅ Mask generation tests passed!\n")


def main():
    """Run all tests."""
    print("=" * 60)
    print("GCHA-Net Implementation Tests")
    print("=" * 60)
    print()
    
    try:
        test_anchor_generation()
        test_gcha_attention()
        test_gcha_block()
        test_mask_generation()
        test_gcha_net()
        test_build_from_config()
        
        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nThe GCHA-Net implementation is working correctly.")
        print("You can now proceed to train the model with:")
        print("  python train.py --config config/default.yaml")
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ TESTS FAILED!")
        print("=" * 60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
