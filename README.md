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
  num_layers: 4      # GCHA blocks

training:
  learning_rate: 1.0e-4
  batch_size: 8
  num_epochs: 100

data:
  dataset: "agroscapes"
  data_root: "./data/agroscapes"
  image_size: [512, 512]
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

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{gcha-net,
  title={GCHA-Net: Guided Cross-Hierarchical Attention Network},
  author={Your Name},
  year={2025},
  url={https://github.com/LittleForest-Ming/GCHA-Net}
}
```

## License

This project is licensed under the terms specified in the LICENSE file.

## Acknowledgments

- PyTorch and PyTorch Lightning communities
- Agroscapes dataset providers
- Vision transformer and attention mechanism research
