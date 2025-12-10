"""
Test script to verify GCHA-Net implementation.
This script tests all components independently and together.
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.gcha_net import GCHANet, build_gcha_net
from modules.gcha_attention import GCHAAttention, GCHABlock
from utils.anchors import generate_anchor_grid, generate_parameter_grid, compute_anchor_iou


def test_anchors():
    """Test anchor generation utilities."""
    print("Testing anchor generation...")
    
    # Test anchor grid generation
    anchors = generate_anchor_grid(
        feature_size=(32, 32),
        num_anchors=9,
        device='cpu'
    )
    print(f"✓ Generated anchor grid: {anchors.shape}")
    assert anchors.shape[0] == 32 * 32 * 9, "Incorrect anchor grid size"
    assert anchors.shape[1] == 4, "Incorrect anchor format"
    
    # Test parameter grid generation
    param_grid = generate_parameter_grid(
        N_total=256,
        feature_dim=128,
        device='cpu'
    )
    print(f"✓ Generated parameter grid: {param_grid.shape}")
    assert param_grid.shape == (256, 128), "Incorrect parameter grid size"
    
    # Test IoU computation
    anchors1 = torch.rand(10, 4)
    anchors2 = torch.rand(15, 4)
    iou = compute_anchor_iou(anchors1, anchors2)
    print(f"✓ Computed IoU: {iou.shape}")
    assert iou.shape == (10, 15), "Incorrect IoU shape"
    
    print("✓ All anchor tests passed!\n")


def test_gcha_attention():
    """Test GCHA attention module."""
    print("Testing GCHA attention...")
    
    # Test GCHA attention layer
    attention = GCHAAttention(
        in_channels=64,
        num_heads=4,
        grid_size=(7, 7),
        N_total=256
    )
    
    x = torch.randn(2, 64, 32, 32)
    output = attention(x)
    print(f"✓ GCHA attention output: {output.shape}")
    assert output.shape == x.shape, "Output shape mismatch"
    
    # Test mask generation
    mask = attention.generate_attention_mask(x)
    print(f"✓ Generated attention mask: {mask.shape}")
    assert mask.shape == (2, 4, 32, 32), "Incorrect mask shape"
    
    # Test GCHA block
    block = GCHABlock(
        in_channels=64,
        num_heads=4,
        grid_size=(7, 7)
    )
    
    output = block(x)
    print(f"✓ GCHA block output: {output.shape}")
    assert output.shape == x.shape, "Block output shape mismatch"
    
    print("✓ All GCHA attention tests passed!\n")


def test_gcha_net():
    """Test complete GCHA-Net model."""
    print("Testing GCHA-Net model...")
    
    # Test segmentation model (using smaller model for memory efficiency)
    model = GCHANet(
        in_channels=3,
        num_classes=19,
        base_channels=32,
        num_blocks=[1, 1, 1, 1],
        num_heads=[2, 4, 8, 16],
        task='segmentation'
    )
    
    x = torch.randn(1, 3, 128, 128)
    output = model(x)
    print(f"✓ Segmentation output: {output.shape}")
    assert output.shape == (1, 19, 128, 128), "Incorrect segmentation output shape"
    
    # Test classification model
    model_cls = GCHANet(
        in_channels=3,
        num_classes=100,
        base_channels=32,
        num_blocks=[1, 1, 1, 1],
        num_heads=[2, 4, 8, 16],
        task='classification'
    )
    
    output_cls = model_cls(x)
    print(f"✓ Classification output: {output_cls.shape}")
    assert output_cls.shape == (1, 100), "Incorrect classification output shape"
    
    # Test parameter grid
    param_grid = model.get_parameter_grid(feature_dim=128, device='cpu')
    print(f"✓ Model parameter grid: {param_grid.shape}")
    
    print("✓ All GCHA-Net tests passed!\n")


def test_model_config():
    """Test model building from configuration."""
    print("Testing model configuration...")
    
    config = {
        'in_channels': 3,
        'num_classes': 19,
        'base_channels': 32,
        'num_blocks': [1, 1, 1, 1],
        'num_heads': [2, 4, 8, 16],
        'grid_size': [7, 7],
        'N_total': 256,
        'epsilon': 1e-6,
        'dropout': 0.1,
        'task': 'segmentation'
    }
    
    model = build_gcha_net(config)
    x = torch.randn(1, 3, 128, 128)
    output = model(x)
    print(f"✓ Config-based model output: {output.shape}")
    assert output.shape == (1, 19, 128, 128), "Incorrect output from config model"
    
    print("✓ Configuration tests passed!\n")


def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model_stats():
    """Display model statistics."""
    print("Model Statistics:")
    print("-" * 50)
    
    model = GCHANet(
        in_channels=3,
        num_classes=19,
        base_channels=32,
        num_blocks=[1, 1, 1, 1],
        num_heads=[2, 4, 8, 16],
        task='segmentation'
    )
    
    num_params = count_parameters(model)
    print(f"Total trainable parameters: {num_params:,}")
    
    # Test forward pass time
    import time
    x = torch.randn(1, 3, 128, 128)
    
    # Warmup
    with torch.no_grad():
        _ = model(x)
    
    # Measure
    start = time.time()
    with torch.no_grad():
        output = model(x)
    end = time.time()
    
    print(f"Forward pass time (CPU): {(end - start) * 1000:.2f} ms")
    print(f"Output shape: {output.shape}")
    print("-" * 50)
    print()


def main():
    """Run all tests."""
    print("=" * 50)
    print("GCHA-Net Implementation Tests")
    print("=" * 50)
    print()
    
    try:
        test_anchors()
        test_gcha_attention()
        test_gcha_net()
        test_model_config()
        test_model_stats()
        
        print("=" * 50)
        print("✓ ALL TESTS PASSED!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
