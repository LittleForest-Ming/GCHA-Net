"""
Example usage of the Prediction Heads module

This example demonstrates how to use the ClassificationHead, RegressionHead,
and PredictionHeads modules for processing decoder output.
"""

import torch
from gcha_net.models.prediction_heads import (
    ClassificationHead,
    RegressionHead,
    PredictionHeads
)


def example_classification_head():
    """Example: Using ClassificationHead independently"""
    print("=" * 60)
    print("Example 1: Classification Head")
    print("=" * 60)
    
    # Initialize classification head
    cls_head = ClassificationHead(
        in_channels=256,
        num_anchors=9,
        num_classes=1,
        hidden_dim=256
    )
    
    # Create dummy decoder output (batch_size=2, channels=256, H=32, W=32)
    decoder_output = torch.randn(2, 256, 32, 32)
    
    # Forward pass
    cls_scores = cls_head(decoder_output)
    
    print(f"Input shape: {decoder_output.shape}")
    print(f"Output shape: {cls_scores.shape}")
    print(f"Output (first 5 values): {cls_scores[0, 0, 0, :5]}")
    print()


def example_regression_head():
    """Example: Using RegressionHead independently"""
    print("=" * 60)
    print("Example 2: Regression Head")
    print("=" * 60)
    
    # Initialize regression head for predicting (Δk, Δm, Δb)
    reg_head = RegressionHead(
        in_channels=256,
        num_anchors=9,
        num_params=3,  # For Δk, Δm, Δb
        hidden_dim=256
    )
    
    # Create dummy decoder output
    decoder_output = torch.randn(2, 256, 32, 32)
    
    # Forward pass
    bbox_preds = reg_head(decoder_output)
    
    print(f"Input shape: {decoder_output.shape}")
    print(f"Output shape: {bbox_preds.shape}")
    print(f"Number of parameters per anchor: 3 (Δk, Δm, Δb)")
    print(f"Output (first 3 values): {bbox_preds[0, 0, 0, :3]}")
    print()


def example_combined_heads():
    """Example: Using both heads in parallel with PredictionHeads"""
    print("=" * 60)
    print("Example 3: Combined Prediction Heads (Parallel Processing)")
    print("=" * 60)
    
    # Initialize combined prediction heads
    pred_heads = PredictionHeads(
        in_channels=256,
        num_anchors=9,
        num_classes=1,
        num_params=3,  # For Δk, Δm, Δb
        hidden_dim=256
    )
    
    # Create dummy decoder output
    decoder_output = torch.randn(2, 256, 32, 32)
    
    # Forward pass through both heads in parallel
    cls_scores, bbox_preds = pred_heads(decoder_output)
    
    print(f"Input shape: {decoder_output.shape}")
    print(f"Classification scores shape: {cls_scores.shape}")
    print(f"Regression predictions shape: {bbox_preds.shape}")
    print()
    print("Classification output (confidence scores):")
    print(f"  Shape: {cls_scores.shape}")
    print(f"  Sample values: {cls_scores[0, 0, 0, :5]}")
    print()
    print("Regression output (Δk, Δm, Δb offsets):")
    print(f"  Shape: {bbox_preds.shape}")
    print(f"  Sample values: {bbox_preds[0, :3, 0, 0]}")
    print()


def example_multiclass():
    """Example: Multi-class classification"""
    print("=" * 60)
    print("Example 4: Multi-class Classification")
    print("=" * 60)
    
    # Initialize with multiple classes
    pred_heads = PredictionHeads(
        in_channels=256,
        num_anchors=9,
        num_classes=10,  # 10 classes
        num_params=3,
        hidden_dim=256
    )
    
    # Create dummy decoder output
    decoder_output = torch.randn(2, 256, 32, 32)
    
    # Forward pass
    cls_scores, bbox_preds = pred_heads(decoder_output)
    
    print(f"Input shape: {decoder_output.shape}")
    print(f"Number of classes: 10")
    print(f"Number of anchors: 9")
    print(f"Classification scores shape: {cls_scores.shape}")
    print(f"  (batch_size, num_anchors * num_classes, height, width)")
    print(f"  = (2, 9 * 10, 32, 32)")
    print(f"Regression predictions shape: {bbox_preds.shape}")
    print()


def example_training_setup():
    """Example: Training setup with loss computation"""
    print("=" * 60)
    print("Example 5: Training Setup")
    print("=" * 60)
    
    # Initialize prediction heads
    pred_heads = PredictionHeads(
        in_channels=256,
        num_anchors=9,
        num_classes=1,
        num_params=3,
        hidden_dim=256
    )
    
    # Create dummy decoder output and targets
    decoder_output = torch.randn(2, 256, 32, 32)
    target_cls = torch.randint(0, 2, (2, 9, 32, 32)).float()
    target_reg = torch.randn(2, 27, 32, 32)  # 9 anchors * 3 params
    
    # Forward pass
    cls_scores, bbox_preds = pred_heads(decoder_output)
    
    # Example loss computation (simplified)
    import torch.nn.functional as F
    
    # Binary cross-entropy for classification
    cls_loss = F.binary_cross_entropy_with_logits(cls_scores, target_cls)
    
    # L1 loss for regression
    reg_loss = F.l1_loss(bbox_preds, target_reg)
    
    # Combined loss
    total_loss = cls_loss + reg_loss
    
    print(f"Classification Loss: {cls_loss.item():.4f}")
    print(f"Regression Loss: {reg_loss.item():.4f}")
    print(f"Total Loss: {total_loss.item():.4f}")
    print()
    print("Note: In practice, you would use more sophisticated loss functions")
    print("such as Focal Loss for classification and Smooth L1 for regression.")
    print()


def example_inference():
    """Example: Inference mode"""
    print("=" * 60)
    print("Example 6: Inference Mode")
    print("=" * 60)
    
    # Initialize prediction heads
    pred_heads = PredictionHeads(
        in_channels=256,
        num_anchors=9,
        num_classes=1,
        num_params=3,
        hidden_dim=256
    )
    
    # Set to evaluation mode
    pred_heads.eval()
    
    # Create dummy decoder output
    decoder_output = torch.randn(1, 256, 32, 32)
    
    # Inference without gradient computation
    with torch.no_grad():
        cls_scores, bbox_preds = pred_heads(decoder_output)
        
        # Apply sigmoid to get probabilities
        cls_probs = torch.sigmoid(cls_scores)
        
        print(f"Input shape: {decoder_output.shape}")
        print(f"Confidence scores (after sigmoid):")
        print(f"  Min: {cls_probs.min().item():.4f}")
        print(f"  Max: {cls_probs.max().item():.4f}")
        print(f"  Mean: {cls_probs.mean().item():.4f}")
        print()
        print(f"Regression predictions (Δk, Δm, Δb):")
        print(f"  Shape: {bbox_preds.shape}")
        print(f"  Sample anchor offsets: {bbox_preds[0, :3, 16, 16]}")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("GCHA-Net Prediction Heads Examples")
    print("=" * 60 + "\n")
    
    # Run all examples
    example_classification_head()
    example_regression_head()
    example_combined_heads()
    example_multiclass()
    example_training_setup()
    example_inference()
    
    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
