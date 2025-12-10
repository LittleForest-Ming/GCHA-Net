# GCHA-Net

Graph-based Cross-modal Hierarchical Attention Network with Parallel Prediction Heads

## Overview

GCHA-Net implements a neural network architecture with two parallel prediction heads that operate on decoder output:

1. **Classification Head**: Predicts confidence scores for each anchor
2. **Regression Head**: Predicts continuous residual offsets (Δk, Δm, Δb) relative to anchor parameters

## Installation

```bash
pip install -r requirements.txt
```

Or install in development mode:

```bash
pip install -e .
```

## Usage

### Basic Example

```python
import torch
from gcha_net.models.prediction_heads import PredictionHeads

# Initialize prediction heads
pred_heads = PredictionHeads(
    in_channels=256,      # Input channels from decoder
    num_anchors=9,        # Number of anchors per location
    num_classes=1,        # Number of classes (1 for binary)
    num_params=3,         # Number of regression parameters (Δk, Δm, Δb)
    hidden_dim=256        # Hidden dimension
)

# Create decoder output (batch_size=2, channels=256, H=32, W=32)
decoder_output = torch.randn(2, 256, 32, 32)

# Forward pass
cls_scores, bbox_preds = pred_heads(decoder_output)

print(f"Classification scores shape: {cls_scores.shape}")  # (2, 9, 32, 32)
print(f"Regression predictions shape: {bbox_preds.shape}") # (2, 27, 32, 32)
```

### Using Individual Heads

```python
from gcha_net.models.prediction_heads import ClassificationHead, RegressionHead

# Classification head only
cls_head = ClassificationHead(
    in_channels=256,
    num_anchors=9,
    num_classes=1
)
cls_scores = cls_head(decoder_output)

# Regression head only
reg_head = RegressionHead(
    in_channels=256,
    num_anchors=9,
    num_params=3
)
bbox_preds = reg_head(decoder_output)
```

## Architecture

### Classification Head
- Processes decoder features through convolutional layers
- Outputs confidence scores for each anchor
- Shape: `(batch_size, num_anchors * num_classes, height, width)`

### Regression Head
- Processes decoder features through convolutional layers
- Outputs residual offsets (Δk, Δm, Δb) for each anchor
- Shape: `(batch_size, num_anchors * num_params, height, width)`

### Parallel Processing
Both heads process the same decoder output independently and in parallel, allowing for efficient computation of both classification and regression tasks.

## Examples

See `examples/prediction_heads_example.py` for comprehensive usage examples including:
- Basic usage
- Multi-class classification
- Training setup
- Inference mode

Run the examples:
```bash
python examples/prediction_heads_example.py
```

## Testing

Run the test suite:
```bash
pytest tests/test_prediction_heads.py -v
```

## License

Apache License 2.0 - See LICENSE file for details