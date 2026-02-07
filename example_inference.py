"""
Old Version: Does not involve cubic offset
Example usage of GCHA-Net for inference.

This script demonstrates how to:
1. Load the model
2. Prepare input data
3. Run inference
4. Visualize results
"""

import torch
import yaml
import numpy as np
from pathlib import Path

from models.gcha_net import build_gcha_net


def load_model(config_path: str, checkpoint_path: str = None) -> torch.nn.Module:
    """
    Load GCHA-Net model from configuration.
    
    Args:
        config_path: Path to config YAML file
        checkpoint_path: Optional path to model checkpoint
        
    Returns:
        Loaded model in eval mode
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Build model
    model = build_gcha_net(config['model'])
    
    # Load checkpoint if provided
    if checkpoint_path and Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    model.eval()
    return model


def preprocess_image(image: np.ndarray, image_size: tuple, 
                     mean: list, std: list) -> torch.Tensor:
    """
    Preprocess input image.
    
    Args:
        image: Input image as numpy array (H, W, C)
        image_size: Target size (H, W)
        mean: Normalization mean
        std: Normalization std
        
    Returns:
        Preprocessed tensor (1, C, H, W)
    """
    # Resize (simplified - use proper interpolation in production)
    # image = cv2.resize(image, (image_size[1], image_size[0]))
    
    # Convert to tensor and normalize
    image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
    
    # Normalize
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    image = (image - mean) / std
    
    # Add batch dimension
    image = image.unsqueeze(0)
    
    return image


def run_inference(model: torch.nn.Module, image_tensor: torch.Tensor,
                 device: str = 'cpu') -> np.ndarray:
    """
    Run inference on input image.
    
    Args:
        model: GCHA-Net model
        image_tensor: Input tensor (1, C, H, W)
        device: Device to run on
        
    Returns:
        Segmentation mask as numpy array (H, W)
    """
    model = model.to(device)
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        output = model(image_tensor)  # (1, num_classes, H, W)
        prediction = torch.argmax(output, dim=1)  # (1, H, W)
    
    # Convert to numpy
    mask = prediction.cpu().numpy()[0]
    
    return mask


def visualize_prediction(image: np.ndarray, mask: np.ndarray, 
                        num_classes: int = 19):
    """
    Visualize segmentation prediction.
    
    Args:
        image: Original image (H, W, C)
        mask: Predicted mask (H, W)
        num_classes: Number of classes
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        
        # Create color map
        colors = plt.cm.get_cmap('tab20', num_classes)
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Segmentation mask
        axes[1].imshow(mask, cmap=colors, vmin=0, vmax=num_classes-1)
        axes[1].set_title('Segmentation Mask')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(image)
        axes[2].imshow(mask, alpha=0.5, cmap=colors, vmin=0, vmax=num_classes-1)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig('prediction_result.png', dpi=150, bbox_inches='tight')
        print("Visualization saved to prediction_result.png")
        
    except ImportError:
        print("Matplotlib not available. Skipping visualization.")


def main():
    """Main example function."""
    print("=" * 60)
    print("GCHA-Net Inference Example")
    print("=" * 60)
    
    # Configuration
    config_path = "config/default.yaml"
    checkpoint_path = None  # Set to checkpoint path if available
    
    # Load model
    print("\n1. Loading model...")
    model = load_model(config_path, checkpoint_path)
    print(f"   Model loaded successfully")
    
    # Create dummy input (replace with actual image loading)
    print("\n2. Preparing input...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Use smaller image size for demo (avoid memory issues)
    demo_image_size = [64, 64]  # Small size for CPU demo
    print(f"   NOTE: Using very small image size {demo_image_size} for CPU demo")
    print(f"         For production use, consider GPU or memory-efficient attention")
    
    # Create dummy RGB image
    height, width = demo_image_size
    dummy_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    print(f"   Input image shape: {dummy_image.shape}")
    
    # Preprocess
    image_tensor = preprocess_image(
        dummy_image,
        demo_image_size,
        config['data']['mean'],
        config['data']['std']
    )
    print(f"   Preprocessed tensor shape: {image_tensor.shape}")
    
    # Run inference
    print("\n3. Running inference...")
    mask = run_inference(model, image_tensor, device='cpu')
    print(f"   Prediction mask shape: {mask.shape}")
    print(f"   Unique classes in prediction: {np.unique(mask)}")
    
    # Visualize
    print("\n4. Visualizing results...")
    visualize_prediction(dummy_image, mask, config['model']['num_classes'])
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)
    print("\nTo use with real images:")
    print("1. Load your image with PIL or OpenCV")
    print("2. Convert to numpy array (H, W, C)")
    print("3. Pass to preprocess_image()")
    print("4. Run inference with run_inference()")
    print("\nNOTE: For large images (>128x128), consider using a GPU")
    print("      or processing in smaller patches.")


if __name__ == '__main__':
    main()
