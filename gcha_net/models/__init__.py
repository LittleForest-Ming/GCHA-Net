"""
Models module for GCHA-Net
"""

from .prediction_heads import ClassificationHead, RegressionHead, PredictionHeads

__all__ = [
    "ClassificationHead",
    "RegressionHead",
    "PredictionHeads",
]
