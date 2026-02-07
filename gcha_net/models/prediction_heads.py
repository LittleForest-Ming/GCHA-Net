"""
(Old Version: Used for secondary prediction offset)
Prediction Heads for GCHA-Net

This module implements two parallel prediction heads that operate on decoder output:
1. ClassificationHead: Predicts confidence scores for each anchor
2. RegressionHead: Predicts continuous residual offsets (Δk, Δm, Δb) relative to anchor parameters
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class ClassificationHead(nn.Module):
    """
    Classification Head for predicting confidence scores for each anchor.
    
    This head takes decoder output features and predicts a confidence score
    for each anchor, indicating the likelihood that an anchor corresponds to
    a valid detection.
    
    Args:
        in_channels (int): Number of input channels from decoder output
        num_anchors (int): Number of anchors per spatial location
        hidden_dim (int, optional): Hidden dimension for intermediate layers. 
                                   If None, uses in_channels
        num_classes (int, optional): Number of classes for classification. 
                                    Default is 1 (binary classification)
    """
    
    def __init__(
        self,
        in_channels: int,
        num_anchors: int,
        hidden_dim: Optional[int] = None,
        num_classes: int = 1
    ):
        super(ClassificationHead, self).__init__()
        
        self.in_channels = in_channels
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        hidden_dim = hidden_dim or in_channels
        
        # Feature transformation layers
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        
        # Classification output layer
        # Output: num_anchors * num_classes confidence scores per location
        self.cls_score = nn.Conv2d(
            hidden_dim, 
            num_anchors * num_classes, 
            kernel_size=1
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize layer weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through classification head.
        
        Args:
            x (torch.Tensor): Decoder output features of shape 
                            (batch_size, in_channels, height, width)
        
        Returns:
            torch.Tensor: Classification scores of shape 
                         (batch_size, num_anchors * num_classes, height, width)
        """
        # Apply feature transformation
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        # Predict classification scores
        cls_score = self.cls_score(out)
        
        return cls_score


class RegressionHead(nn.Module):
    """
    Regression Head for predicting continuous residual offsets relative to anchor parameters.
    
    This head takes decoder output features and predicts residual offsets (Δk, Δm, Δb)
    for each anchor, which are used to refine the anchor parameters.
    
    Args:
        in_channels (int): Number of input channels from decoder output
        num_anchors (int): Number of anchors per spatial location
        hidden_dim (int, optional): Hidden dimension for intermediate layers.
                                   If None, uses in_channels
        num_params (int, optional): Number of parameters to regress (default: 3 for Δk, Δm, Δb)
    """
    
    def __init__(
        self,
        in_channels: int,
        num_anchors: int,
        hidden_dim: Optional[int] = None,
        num_params: int = 3
    ):
        super(RegressionHead, self).__init__()
        
        self.in_channels = in_channels
        self.num_anchors = num_anchors
        self.num_params = num_params
        hidden_dim = hidden_dim or in_channels
        
        # Feature transformation layers
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        
        # Regression output layer
        # Output: num_anchors * num_params residual offsets per location
        self.bbox_pred = nn.Conv2d(
            hidden_dim,
            num_anchors * num_params,
            kernel_size=1
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize layer weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through regression head.
        
        Args:
            x (torch.Tensor): Decoder output features of shape 
                            (batch_size, in_channels, height, width)
        
        Returns:
            torch.Tensor: Regression offsets of shape 
                         (batch_size, num_anchors * num_params, height, width)
                         where num_params typically represents (Δk, Δm, Δb)
        """
        # Apply feature transformation
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        # Predict regression offsets
        bbox_pred = self.bbox_pred(out)
        
        return bbox_pred


class PredictionHeads(nn.Module):
    """
    Combined Prediction Heads module with parallel Classification and Regression heads.
    
    This module combines both the ClassificationHead and RegressionHead to process
    decoder output in parallel, producing both confidence scores and residual offsets.
    
    Args:
        in_channels (int): Number of input channels from decoder output
        num_anchors (int): Number of anchors per spatial location
        hidden_dim (int, optional): Hidden dimension for intermediate layers.
                                   If None, uses in_channels
        num_classes (int, optional): Number of classes for classification (default: 1)
        num_params (int, optional): Number of regression parameters (default: 3 for Δk, Δm, Δb)
    """
    
    def __init__(
        self,
        in_channels: int,
        num_anchors: int,
        hidden_dim: Optional[int] = None,
        num_classes: int = 1,
        num_params: int = 3
    ):
        super(PredictionHeads, self).__init__()
        
        self.in_channels = in_channels
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.num_params = num_params
        
        # Initialize both heads
        self.classification_head = ClassificationHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            hidden_dim=hidden_dim,
            num_classes=num_classes
        )
        
        self.regression_head = RegressionHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            hidden_dim=hidden_dim,
            num_params=num_params
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both prediction heads in parallel.
        
        Args:
            x (torch.Tensor): Decoder output features of shape 
                            (batch_size, in_channels, height, width)
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - cls_scores: Classification scores of shape 
                             (batch_size, num_anchors * num_classes, height, width)
                - bbox_preds: Regression offsets of shape 
                             (batch_size, num_anchors * num_params, height, width)
        """
        # Process through both heads in parallel
        cls_scores = self.classification_head(x)
        bbox_preds = self.regression_head(x)
        
        return cls_scores, bbox_preds
