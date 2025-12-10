"""
Example usage of GCHA-Net for semantic segmentation.
"""

import torch
from models.gcha_net import GCHANet
import yaml


def example_segmentation():
    """Example: Semantic segmentation with GCHA-Net."""
    print("=" * 60)
    print("Example: Semantic Segmentation with GCHA-Net")
    print("=" * 60)
    
    # Create model
    model = GCHANet(
        in_channels=3,
        num_classes=19,  # e.g., Cityscapes has 19 classes
        base_channels=64,
        num_blocks=[2, 2, 6, 2],
        num_heads=[2, 4, 8, 16],
        grid_size=(7, 7),
        N_total=256,
        epsilon=1e-6,
        dropout=0.1,
        task='segmentation'
    )
    
    print(f"\nModel created:")
    print(f"  - Input channels: 3 (RGB)")
    print(f"  - Output classes: 19")
    print(f"  - Architecture: 4 stages with [2, 2, 6, 2] GCHA blocks")
    print(f"  - Attention heads: [2, 4, 8, 16] per stage")
    
    # Set model to evaluation mode
    model.eval()
    
    # Create dummy input (batch_size=1, channels=3, height=512, width=512)
    x = torch.randn(1, 3, 512, 512)
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    print(f"  - Batch size: {output.shape[0]}")
    print(f"  - Number of classes: {output.shape[1]}")
    print(f"  - Height: {output.shape[2]}")
    print(f"  - Width: {output.shape[3]}")
    
    # Get predicted class for each pixel
    predictions = torch.argmax(output, dim=1)
    print(f"\nPrediction map shape: {predictions.shape}")
    print(f"Unique classes in prediction: {torch.unique(predictions).tolist()}")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters: {num_params:,}")
    
    print("\n" + "=" * 60)


def example_classification():
    """Example: Image classification with GCHA-Net."""
    print("\n" + "=" * 60)
    print("Example: Image Classification with GCHA-Net")
    print("=" * 60)
    
    # Create model
    model = GCHANet(
        in_channels=3,
        num_classes=1000,  # e.g., ImageNet has 1000 classes
        base_channels=64,
        num_blocks=[2, 2, 6, 2],
        num_heads=[2, 4, 8, 16],
        task='classification'
    )
    
    print(f"\nModel created:")
    print(f"  - Input channels: 3 (RGB)")
    print(f"  - Output classes: 1000 (ImageNet)")
    print(f"  - Task: Classification")
    
    # Set model to evaluation mode
    model.eval()
    
    # Create dummy input (batch_size=2, channels=3, height=224, width=224)
    x = torch.randn(2, 3, 224, 224)
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    print(f"  - Batch size: {output.shape[0]}")
    print(f"  - Number of classes: {output.shape[1]}")
    
    # Get top-5 predictions for each image
    probs = torch.softmax(output, dim=1)
    top5_probs, top5_indices = torch.topk(probs, k=5, dim=1)
    
    print(f"\nTop-5 predictions for first image:")
    for i in range(5):
        print(f"  Class {top5_indices[0, i].item()}: {top5_probs[0, i].item():.4f}")
    
    print("\n" + "=" * 60)


def example_with_config():
    """Example: Load model from configuration file."""
    print("\n" + "=" * 60)
    print("Example: Load Model from Configuration")
    print("=" * 60)
    
    # Load config
    with open('config/default.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"\nLoaded configuration from 'config/default.yaml'")
    print(f"Key parameters:")
    print(f"  - N_total: {config['model']['N_total']}")
    print(f"  - epsilon: {config['model']['epsilon']}")
    print(f"  - grid_size: {config['model']['grid_size']}")
    print(f"  - num_blocks: {config['model']['num_blocks']}")
    print(f"  - num_heads: {config['model']['num_heads']}")
    
    from models.gcha_net import build_gcha_net
    model = build_gcha_net(config['model'])
    
    print(f"\nModel built from configuration")
    print(f"Task: {config['model']['task']}")
    
    print("\n" + "=" * 60)


def example_attention_mechanism():
    """Example: Using GCHA attention layer directly."""
    print("\n" + "=" * 60)
    print("Example: GCHA Attention Mechanism")
    print("=" * 60)
    
    from modules.gcha_attention import GCHAAttention
    
    # Create attention layer
    attention = GCHAAttention(
        in_channels=256,
        num_heads=8,
        grid_size=(7, 7),
        N_total=256,
        epsilon=1e-6
    )
    
    print(f"\nGCHA Attention layer created:")
    print(f"  - Input channels: 256")
    print(f"  - Number of attention heads: 8")
    print(f"  - Grid size: (7, 7)")
    print(f"  - Parameter count: 256")
    
    # Create input
    x = torch.randn(2, 256, 32, 32)
    print(f"\nInput shape: {x.shape}")
    
    # Generate attention mask
    attention.eval()
    with torch.no_grad():
        mask = attention.generate_attention_mask(x, threshold=0.5)
        print(f"Generated attention mask shape: {mask.shape}")
        
        # Apply attention
        output = attention(x)
        print(f"Attention output shape: {output.shape}")
        
        # Check how many positions are attended to
        attended_ratio = mask.mean().item()
        print(f"\nAttention statistics:")
        print(f"  - Percentage of attended positions: {attended_ratio * 100:.2f}%")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    # Run all examples
    example_segmentation()
    example_classification()
    example_with_config()
    example_attention_mechanism()
    
    print("\n✓ All examples completed successfully!")
