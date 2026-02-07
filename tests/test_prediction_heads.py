"""
Old Version: Used for secondary prediction offset
Tests for prediction heads module
"""

import pytest
import torch

from gcha_net.models.prediction_heads import (
    ClassificationHead,
    RegressionHead,
    PredictionHeads
)


class TestClassificationHead:
    """Test cases for ClassificationHead"""
    
    def test_initialization(self):
        """Test ClassificationHead initialization"""
        head = ClassificationHead(
            in_channels=256,
            num_anchors=9,
            hidden_dim=256,
            num_classes=1
        )
        
        assert head.in_channels == 256
        assert head.num_anchors == 9
        assert head.num_classes == 1
    
    def test_forward_shape(self):
        """Test forward pass output shape"""
        batch_size = 2
        in_channels = 256
        height, width = 32, 32
        num_anchors = 9
        num_classes = 1
        
        head = ClassificationHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=num_classes
        )
        
        # Create dummy input
        x = torch.randn(batch_size, in_channels, height, width)
        
        # Forward pass
        output = head(x)
        
        # Check output shape
        expected_shape = (batch_size, num_anchors * num_classes, height, width)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    
    def test_forward_multiclass(self):
        """Test forward pass with multiple classes"""
        batch_size = 2
        in_channels = 128
        height, width = 16, 16
        num_anchors = 5
        num_classes = 10
        
        head = ClassificationHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=num_classes
        )
        
        x = torch.randn(batch_size, in_channels, height, width)
        output = head(x)
        
        expected_shape = (batch_size, num_anchors * num_classes, height, width)
        assert output.shape == expected_shape
    
    def test_different_hidden_dim(self):
        """Test with custom hidden dimension"""
        head = ClassificationHead(
            in_channels=256,
            num_anchors=9,
            hidden_dim=512,
            num_classes=1
        )
        
        x = torch.randn(1, 256, 16, 16)
        output = head(x)
        
        assert output.shape == (1, 9, 16, 16)
    
    def test_gradient_flow(self):
        """Test that gradients flow through the network"""
        head = ClassificationHead(
            in_channels=64,
            num_anchors=3,
            num_classes=1
        )
        
        x = torch.randn(1, 64, 8, 8, requires_grad=True)
        output = head(x)
        loss = output.sum()
        loss.backward()
        
        # Check that input has gradients
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestRegressionHead:
    """Test cases for RegressionHead"""
    
    def test_initialization(self):
        """Test RegressionHead initialization"""
        head = RegressionHead(
            in_channels=256,
            num_anchors=9,
            hidden_dim=256,
            num_params=3
        )
        
        assert head.in_channels == 256
        assert head.num_anchors == 9
        assert head.num_params == 3
    
    def test_forward_shape(self):
        """Test forward pass output shape"""
        batch_size = 2
        in_channels = 256
        height, width = 32, 32
        num_anchors = 9
        num_params = 3
        
        head = RegressionHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_params=num_params
        )
        
        # Create dummy input
        x = torch.randn(batch_size, in_channels, height, width)
        
        # Forward pass
        output = head(x)
        
        # Check output shape
        expected_shape = (batch_size, num_anchors * num_params, height, width)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    
    def test_forward_different_params(self):
        """Test forward pass with different number of parameters"""
        batch_size = 2
        in_channels = 128
        height, width = 16, 16
        num_anchors = 5
        num_params = 4  # Different from default 3
        
        head = RegressionHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_params=num_params
        )
        
        x = torch.randn(batch_size, in_channels, height, width)
        output = head(x)
        
        expected_shape = (batch_size, num_anchors * num_params, height, width)
        assert output.shape == expected_shape
    
    def test_different_hidden_dim(self):
        """Test with custom hidden dimension"""
        head = RegressionHead(
            in_channels=256,
            num_anchors=9,
            hidden_dim=512,
            num_params=3
        )
        
        x = torch.randn(1, 256, 16, 16)
        output = head(x)
        
        assert output.shape == (1, 27, 16, 16)  # 9 anchors * 3 params
    
    def test_gradient_flow(self):
        """Test that gradients flow through the network"""
        head = RegressionHead(
            in_channels=64,
            num_anchors=3,
            num_params=3
        )
        
        x = torch.randn(1, 64, 8, 8, requires_grad=True)
        output = head(x)
        loss = output.sum()
        loss.backward()
        
        # Check that input has gradients
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestPredictionHeads:
    """Test cases for combined PredictionHeads"""
    
    def test_initialization(self):
        """Test PredictionHeads initialization"""
        heads = PredictionHeads(
            in_channels=256,
            num_anchors=9,
            hidden_dim=256,
            num_classes=1,
            num_params=3
        )
        
        assert heads.in_channels == 256
        assert heads.num_anchors == 9
        assert heads.num_classes == 1
        assert heads.num_params == 3
        assert isinstance(heads.classification_head, ClassificationHead)
        assert isinstance(heads.regression_head, RegressionHead)
    
    def test_forward_shape(self):
        """Test forward pass output shapes"""
        batch_size = 2
        in_channels = 256
        height, width = 32, 32
        num_anchors = 9
        num_classes = 1
        num_params = 3
        
        heads = PredictionHeads(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=num_classes,
            num_params=num_params
        )
        
        # Create dummy input
        x = torch.randn(batch_size, in_channels, height, width)
        
        # Forward pass
        cls_scores, bbox_preds = heads(x)
        
        # Check output shapes
        expected_cls_shape = (batch_size, num_anchors * num_classes, height, width)
        expected_bbox_shape = (batch_size, num_anchors * num_params, height, width)
        
        assert cls_scores.shape == expected_cls_shape, \
            f"Classification scores: expected {expected_cls_shape}, got {cls_scores.shape}"
        assert bbox_preds.shape == expected_bbox_shape, \
            f"Bbox predictions: expected {expected_bbox_shape}, got {bbox_preds.shape}"
    
    def test_parallel_processing(self):
        """Test that both heads process input in parallel"""
        heads = PredictionHeads(
            in_channels=128,
            num_anchors=5,
            num_classes=2,
            num_params=3
        )
        
        x = torch.randn(1, 128, 16, 16)
        cls_scores, bbox_preds = heads(x)
        
        # Both outputs should be tensors
        assert isinstance(cls_scores, torch.Tensor)
        assert isinstance(bbox_preds, torch.Tensor)
        
        # Both should have the same spatial dimensions
        assert cls_scores.shape[2:] == bbox_preds.shape[2:]
    
    def test_gradient_flow_both_heads(self):
        """Test that gradients flow through both heads"""
        heads = PredictionHeads(
            in_channels=64,
            num_anchors=3,
            num_classes=1,
            num_params=3
        )
        
        x = torch.randn(1, 64, 8, 8, requires_grad=True)
        cls_scores, bbox_preds = heads(x)
        
        # Compute loss from both outputs
        loss = cls_scores.sum() + bbox_preds.sum()
        loss.backward()
        
        # Check that input has gradients
        assert x.grad is not None
        assert x.grad.shape == x.shape
    
    def test_multiclass_multiparams(self):
        """Test with multiple classes and parameters"""
        heads = PredictionHeads(
            in_channels=256,
            num_anchors=9,
            num_classes=10,
            num_params=4
        )
        
        x = torch.randn(2, 256, 32, 32)
        cls_scores, bbox_preds = heads(x)
        
        assert cls_scores.shape == (2, 90, 32, 32)  # 9 * 10
        assert bbox_preds.shape == (2, 36, 32, 32)  # 9 * 4
    
    def test_batch_processing(self):
        """Test processing of different batch sizes"""
        heads = PredictionHeads(
            in_channels=128,
            num_anchors=5,
            num_classes=1,
            num_params=3
        )
        
        # Test different batch sizes
        for batch_size in [1, 2, 4, 8]:
            x = torch.randn(batch_size, 128, 16, 16)
            cls_scores, bbox_preds = heads(x)
            
            assert cls_scores.shape[0] == batch_size
            assert bbox_preds.shape[0] == batch_size
