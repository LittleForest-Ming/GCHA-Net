# GCHA-Net: Geometry-Constrained Highway Attention Network

GCHA-Net is a deep learning architecture for highway lane detection with geometry-constrained attention mechanisms.

## Architecture Overview

The model consists of three main components:

### 1. Backbone: ResNet50 + Feature Pyramid Network (FPN)
- **ResNet50**: Pre-trained on ImageNet for robust feature extraction
- **FPN**: Fuses multi-scale features from ResNet stages (C2, C3, C4, C5) into a unified feature map
- Output: Unified feature map with 256 channels

### 2. GCHA Decoder: Geometry-Constrained Attention
- Replaces standard cross-attention with geometry-aware attention
- Accepts a static boolean mask `M_geo` to constrain attention to valid regions
- The mask prevents attention to pixels outside polynomial trajectory boundaries
- Multi-layer decoder with self-attention and geometry-constrained cross-attention

### 3. Dual-Head Architecture
- **Anchor Classification Head**: Binary MLP classifier for anchor validation
- **Parameter Regression Head**: MLP regressor for trajectory parameter offsets (Δk, Δm, Δb)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
import torch
from models import GCHANet

# Create model
model = GCHANet(
    embed_dim=256,
    num_heads=8,
    num_decoder_layers=6,
    num_queries=100,
    pretrained=True
)

# Input: batch of images
images = torch.randn(2, 3, 224, 224)

# Forward pass
anchor_logits, param_offsets = model(images)

# anchor_logits: (B, num_queries, 1) - binary classification scores
# param_offsets: (B, num_queries, 3) - [Δk, Δm, Δb] parameter offsets
```

### Using Geometry Mask

```python
# Create geometry mask
# Shape: (batch_size, num_queries, spatial_positions)
# True = keep attention, False = mask out
geometry_mask = create_polynomial_trajectory_mask(...)

# Forward pass with geometry constraint
anchor_logits, param_offsets = model(images, geometry_mask=geometry_mask)
```

## Model Parameters

- `embed_dim`: Embedding dimension for decoder (default: 256)
- `num_heads`: Number of attention heads (default: 8)
- `num_decoder_layers`: Number of decoder layers (default: 6)
- `num_queries`: Number of query embeddings/anchors (default: 100)
- `dropout`: Dropout probability (default: 0.1)
- `pretrained`: Use pretrained ResNet50 weights (default: True)

## Testing

Run the test suite to verify the implementation:

```bash
python test_model.py
```

## Key Features

1. **Geometry-Constrained Attention**: Custom attention mechanism that respects spatial constraints
2. **Multi-Scale Feature Fusion**: FPN combines features from multiple ResNet stages
3. **Dual-Head Design**: Separate branches for classification and regression
4. **Flexible Masking**: Support for 2D or 3D geometry masks

## Model Components

### GeometryConstrainedAttention
Custom attention layer with geometry constraints:
- Query, Key, Value projections
- Multi-head attention with configurable heads
- Boolean mask support for spatial constraints

### FeaturePyramidNetwork
Fuses multi-scale features from ResNet:
- Lateral connections with 1x1 convolutions
- Top-down pathway with upsampling
- Final fusion layer for unified output

### GCHADecoder
Transformer-style decoder with:
- Self-attention for query refinement
- Geometry-constrained cross-attention
- Feed-forward networks with residual connections

### Prediction Heads
- **AnchorClassificationHead**: 3-layer MLP for binary classification
- **ParameterRegressionHead**: 3-layer MLP for 3-parameter regression

## License

See LICENSE file for details.