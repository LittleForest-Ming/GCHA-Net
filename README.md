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
Geometry-Constrained Hough Attention Network

## Overview

GCHA-Net implements the Geometry-Constrained Hough Attention (GCHA) layer, a novel attention mechanism that replaces standard cross-attention with a geometry-aware variant. The GCHA layer uses a pre-computed geometric mask to enforce spatial constraints based on Hough transform geometry.

## Core Features

### GCHA Attention Layer

The GCHA layer implements masked attention with geometric constraints:

**Attention Formula:**
```
Attention = Softmax((Q·K^T)/√d + M_geo)
```

Where:
- `Q`, `K`, `V` are query, key, and value matrices
- `d` is the dimension per attention head
- `M_geo` is the geometric mask of shape `(N_total, HW)`

### Geometric Mask

The geometric mask `M_geo` enforces spatial constraints:

```
M_geo[i, j] = 0    if |x̃_j - x̂_i,j| < ε
M_geo[i, j] = -∞   otherwise
```

Where:
- `x̃_j` are the spatial positions of the feature map (shape: `HW`)
- `x̂_i,j` are the predicted positions for each query (shape: `N_total × HW`)
- `ε` is the distance threshold (epsilon)
# GCHA-Net: Geometry-Constrained Highway Attention Network

GCHA-Net is a deep learning architecture for highway lane detection with geometry-constrained attention mechanisms.

## Architecture Overview

The model consists of three main components:

### 1. Backbone: ResNet(18/34/50) + Feature Pyramid Network (FPN)
- **ResNet**: Pre-trained on ImageNet for robust feature extraction
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
# GCHA-Net

**Guided Cross-Hierarchical Attention Network** for semantic segmentation.

## Overview

GCHA-Net is a neural network architecture that leverages guided cross-hierarchical attention mechanisms for semantic segmentation tasks. The model uses anchor-based attention to efficiently process multi-scale features with spatial guidance.

### Key Features

- **GCHA Attention Mechanism**: Novel attention layer with anchor-based spatial guidance
- **Multi-scale Feature Processing**: Hierarchical feature extraction and fusion
- **Parameter Grid (A)**: Configurable anchor points for attention guidance
- **PyTorch Lightning Integration**: Modern training setup with automatic logging and checkpointing

## Architecture Components

### 1. `models/gcha_net.py`
Main model definition integrating all components:
- `GCHANet`: Main network architecture
- `FeatureExtractor`: Multi-scale CNN backbone
- `build_gcha_net()`: Model factory function

### 2. `modules/gcha_attention.py`
Implementation of the GCHA layer and mask generation logic:
- `GCHAAttention`: Core attention mechanism with anchor guidance
- `GCHABlock`: Transformer block with GCHA attention and feed-forward network
- Spatial mask generation based on anchor positions

### 3. `utils/anchors.py`
Functions for generating the parameter grid A:
- `generate_anchor_grid()`: Create uniform anchor distribution
- `generate_hierarchical_anchors()`: Multi-scale anchor generation
- `compute_anchor_distances()`: Distance computation utilities
- `get_position_encoding()`: Sinusoidal position encodings

### 4. `config/default.yaml`
Hyperparameters configuration:
- `N_total`: Number of anchor points (default: 64)
- `epsilon`: Numerical stability parameter (default: 1e-6)
- Learning rates, batch sizes, optimizer settings
- Dataset paths and preprocessing parameters

### 5. `train.py`
Minimal training loop setup using PyTorch Lightning:
- Command-line interface for training
- Automatic logging with TensorBoard
- Model checkpointing
- Configuration management

## Installation

```bash
# Clone the repository
git clone https://github.com/LittleForest-Ming/GCHA-Net.git
cd GCHA-Net

# Install dependencies
pip install torch
pip install -r requirements.txt
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
### Basic Usage

```python
import torch
from modules.gcha_attention import GCHALayer

# Create GCHA layer
gcha = GCHALayer(
    d_model=256,      # Model dimension
    n_total=100,      # Number of query positions
    hw=64,            # Height × Width of feature map (e.g., 8×8)
    epsilon=0.5,      # Geometric constraint threshold
    num_heads=8,      # Number of attention heads
    dropout=0.1       # Dropout rate
)

# Create inputs
batch_size = 4
query = torch.randn(batch_size, 100, 256)  # (batch, n_total, d_model)
key = torch.randn(batch_size, 64, 256)     # (batch, hw, d_model)
value = torch.randn(batch_size, 64, 256)   # (batch, hw, d_model)

# Forward pass
output, attention_weights = gcha(query, key, value)
```

### With Geometric Mask

```python
import torch
from modules.gcha_attention import GCHALayer

# Create GCHA layer
gcha = GCHALayer(d_model=256, n_total=100, hw=64, epsilon=1.0)

# Define spatial positions
# x_tilde: feature map grid positions (hw, 2)
x_tilde = torch.randn(64, 2)

# x_hat: predicted positions for each query (n_total, hw, 2)
x_hat = torch.randn(100, 64, 2)

# Pre-compute geometric mask
gcha.set_geometric_mask(x_tilde, x_hat)

# Forward pass uses the pre-computed mask
output, attn_weights = gcha(query, key, value)
```

### Dynamic Mask Computation

```python
# Compute mask during forward pass
output, attn_weights = gcha(
    query, key, value,
    x_tilde=x_tilde,  # Feature map positions
    x_hat=x_hat       # Predicted positions
)
```

## Architecture

### Classification Head
- Processes decoder features through convolutional layers
- Outputs confidence scores for each anchor
- Shape: `(batch_size, num_anchors * num_classes, height, width)`

### Regression Head
- Processes decoder features through convolutional layers
- Outputs residual offsets (Δp, Δk, Δm, Δb) for each anchor
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
```
modules/
├── __init__.py           # Module exports
└── gcha_attention.py     # GCHA layer implementation
```

### GCHALayer Class

**Parameters:**
- `d_model` (int): Model dimension (embedding dimension)
- `n_total` (int): Total number of query positions
- `hw` (int): Height × Width of the feature map
- `epsilon` (float): Threshold for geometric constraint
- `num_heads` (int): Number of attention heads (default: 8)
- `dropout` (float): Dropout probability (default: 0.1)

**Methods:**
- `forward(query, key, value, x_tilde=None, x_hat=None, mask=None)`: Forward pass
- `compute_geometric_mask(x_tilde, x_hat)`: Compute geometric mask
- `set_geometric_mask(x_tilde, x_hat)`: Pre-compute and set the mask

**Returns:**
- `output`: Attention output of shape `(batch_size, n_total, d_model)`
- `attention_weights`: Attention weights of shape `(batch_size, num_heads, n_total, hw)`

## Examples

Run the example script to see various usage scenarios:

```bash
python example_usage.py
```

This includes:
1. Basic usage without geometric mask
2. Usage with pre-computed geometric mask
3. Dynamic mask computation
4. Integration with transformer architecture

## Testing

Run the test suite:

```bash
python -m pytest tests/
```

Or run the standalone test script:

```bash
# Note: Create and run your own tests as needed
python test_gcha.py
```

## Implementation Details

### Multi-Head Attention

The GCHA layer implements multi-head attention with geometric constraints:

1. Linear projections split input into multiple heads
2. Attention scores computed as `Q·K^T/√d_k`
3. Geometric mask `M_geo` is added to scores
4. Softmax normalization applied
5. Attention applied to values
6. Heads concatenated and projected to output

### Geometric Mask Construction

The mask enforces spatial constraints:

1. Compute distance: `|x̃_j - x̂_i,j|` (L2 norm)
2. Set `M_geo[i,j] = 0` if distance < epsilon
3. Set `M_geo[i,j] = -∞` otherwise

This ensures queries only attend to spatially-constrained key positions.

## Applications

The GCHA layer is particularly useful for:

- Object detection with Hough voting
- Pose estimation with geometric constraints
- Any task requiring geometry-aware attention
- Transformer decoders with spatial priors

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
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
# param_offsets: (B, num_queries, 4) - [Δp, Δk, Δm, Δb] parameter offsets
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
- **ParameterRegressionHead**: 3-layer MLP for 4-parameter regression

## License

See LICENSE file for details.
# Clone the repository
git clone https://github.com/LittleForest-Ming/GCHA-Net.git
cd GCHA-Net

# Install dependencies
pip install -r requirements.txt
```

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- PyTorch Lightning >= 2.0.0
- See `requirements.txt` for full list

## Usage

### Training

```bash
# Train with default configuration
python train.py --config config/default.yaml

# Train with custom settings
python train.py --config config/default.yaml \
    --gpus 1 \
    --epochs 100 \
    --batch_size 8 \
    --lr 1e-4
```

### Configuration

Edit `config/default.yaml` to customize:

```yaml
model:
  n_total: 64        # Number of anchors
  epsilon: 1.0e-6    # Numerical stability
  embed_dim: 256     # Embedding dimension
  num_heads: 8       # Attention heads
  num_layers: 2      # GCHA blocks

training:
  learning_rate: 1.0e-4
  batch_size: 8
  num_epochs: 100

data:
  dataset: "dataset"
  data_root: "./data/dataset"
  image_size: [640, 360]
```

### Custom Dataset

To use your own dataset, implement a PyTorch Dataset class and replace `DummySegmentationDataset` in `train.py`:

```python
class CustomDataset(Dataset):
    def __init__(self, data_root, split='train'):
        # Load your data
        pass
    
    def __getitem__(self, idx):
        # Return (image, mask) pair
        image = ...  # torch.Tensor of shape (3, H, W)
        mask = ...   # torch.Tensor of shape (H, W)
        return image, mask
```

## Model Architecture

### GCHA Attention Flow

1. **Feature Extraction**: Multi-scale CNN extracts hierarchical features
2. **Anchor Generation**: Uniform grid of N_total anchors created for each scale
3. **Guided Attention**: 
   - Queries and keys assigned to nearest anchors
   - Attention computed within anchor-guided regions
   - Distance-based soft masking applied
4. **Cross-Hierarchical Fusion**: Features from all scales combined
5. **Segmentation Head**: Decodes fused features to class predictions

### Key Parameters

- **N_total**: Controls the number of anchor points in the parameter grid A
  - Higher values: More fine-grained spatial guidance
  - Lower values: More global attention patterns
  
- **epsilon (ε)**: Numerical stability in distance computations
  - Prevents division by zero
  - Affects soft masking sensitivity

## Project Structure

```
(Previous Versions)
GCHA-Net/
├── models/
│   ├── __init__.py
│   └── gcha_net.py           # Main model definition
├── modules/
│   ├── __init__.py
│   └── gcha_attention.py     # GCHA attention layer
├── utils/
│   ├── __init__.py
│   └── anchors.py            # Anchor generation utilities
├── config/
│   └── default.yaml          # Configuration file
├── train.py                  # Training script
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── LICENSE

```
```
(New Version)
GCHA-Net/
├── model/
│   ├── __init__.py
│   ├── anchors.py
│   ├── backbone.py
│   ├── detector.py
│   ├── matcher.py
│   ├── Encode-decode.py
│   └── gcha_net.py           # Main model definition
├── modules/
│   ├── __init__.py
│   └── gcha_attention.py     # GCHA attention layer
├── utils/
│   ├── __init__.py
│   └── dataset.py            # Anchor generation utilities
├── configs/
│   ├── __init__.py  
│   └── config.py           # Configuration file
├── train_main.py                  # Training script
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── LICENSE

```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{gcha_net,
  title={GCHA-Net: Geometry-Constrained Hough Attention Network},
  author={LittleForest-Ming},
  year={2026},
@misc{gcha-net,
  title={GCHA-Net: Guided Cross-Hierarchical Attention Network},
  author={LittleForest-Ming},
  year={2026},
  url={https://github.com/LittleForest-Ming/GCHA-Net}
}
```

## Contact

For questions or issues, please open an issue on GitHub.
## License

This project is licensed under the terms specified in the LICENSE file.

## Acknowledgments

- PyTorch and PyTorch Lightning communities
- Vision transformer and attention mechanism research
