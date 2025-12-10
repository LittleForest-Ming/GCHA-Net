# GCHA-Net

Grid-based Channel-wise Hierarchical Attention Network for Semantic Segmentation

## Overview

GCHA-Net is a deep learning architecture that combines spatial and channel-wise attention mechanisms with a hierarchical grid-based approach for efficient feature refinement. The model is designed for semantic segmentation tasks, particularly suited for agricultural scene understanding (e.g., Agroscapes dataset).

## Architecture

The model consists of three main components:

1. **GCHA Attention Module** (`modules/gcha_attention.py`): Implements the Grid-based Channel-wise Hierarchical Attention layer with mask generation logic
2. **Anchor Utilities** (`utils/anchors.py`): Functions for generating the parameter grid $\mathcal{A}$
3. **Main Model** (`models/gcha_net.py`): Complete network architecture integrating encoder-decoder with GCHA blocks

## Key Features

- Multi-head attention with spatial and channel-wise mechanisms
- Hierarchical grid-based attention masking
- Parameter grid generation for efficient attention computation
- Support for both segmentation and classification tasks
- PyTorch Lightning integration for easy training

## Installation

```bash
# Clone the repository
git clone https://github.com/LittleForest-Ming/GCHA-Net.git
cd GCHA-Net

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Edit `config/default.yaml` to customize:

- **Model parameters**: `N_total` (attention parameters), `epsilon` (stability), grid size
- **Training settings**: Learning rate, batch size, optimizer
- **Data paths**: Dataset root directory (e.g., Agroscapes)
- **Hyperparameters**: Number of blocks, attention heads, dropout

### Key Hyperparameters

- `N_total`: Total number of attention parameters (default: 256)
- `epsilon`: Numerical stability parameter (default: 1e-6)
- `grid_size`: Spatial grid size for hierarchical attention (default: [7, 7])
- `learning_rate`: Initial learning rate (default: 0.0001)

## Usage

### Training

```bash
# Train with default configuration
python train.py

# Training will use the configuration from config/default.yaml
```

### Model Architecture

```python
from models.gcha_net import GCHANet

# Create model
model = GCHANet(
    in_channels=3,
    num_classes=19,
    base_channels=64,
    num_blocks=[2, 2, 6, 2],
    num_heads=[2, 4, 8, 16],
    grid_size=(7, 7),
    N_total=256,
    epsilon=1e-6,
    task='segmentation'
)

# Forward pass
import torch
x = torch.randn(1, 3, 512, 512)
output = model(x)  # Shape: (1, 19, 512, 512)
```

### Using GCHA Attention

```python
from modules.gcha_attention import GCHAAttention

# Create GCHA attention layer
attention = GCHAAttention(
    in_channels=256,
    num_heads=8,
    grid_size=(7, 7),
    N_total=256,
    epsilon=1e-6
)

# Apply attention
x = torch.randn(1, 256, 32, 32)
output = attention(x)  # Shape: (1, 256, 32, 32)
```

## Project Structure

```
GCHA-Net/
├── models/
│   ├── __init__.py
│   └── gcha_net.py          # Main model definition
├── modules/
│   ├── __init__.py
│   └── gcha_attention.py    # GCHA layer implementation
├── utils/
│   ├── __init__.py
│   └── anchors.py           # Parameter grid generation
├── config/
│   └── default.yaml         # Hyperparameters
├── train.py                 # Training script
├── requirements.txt         # Dependencies
└── README.md
```

## Model Components

### 1. GCHA Encoder
- Multi-stage feature extraction with GCHA blocks
- Hierarchical feature maps at different resolutions
- Channel-wise and spatial attention at each stage

### 2. GCHA Decoder
- Upsampling with skip connections
- Feature refinement at multiple scales
- Final segmentation head

### 3. GCHA Attention
- Multi-head attention mechanism
- Grid-based hierarchical masking
- Channel attention module
- Parameter grid $\mathcal{A}$ for efficient computation

## Training Details

The training script (`train.py`) uses PyTorch Lightning and includes:

- Automatic mixed precision training (AMP)
- Gradient clipping
- Cosine annealing learning rate schedule
- Model checkpointing
- TensorBoard logging
- Validation metrics (mIoU, accuracy)

## Dataset Support

Currently configured for:
- Agroscapes (agricultural scenes)
- Cityscapes (urban scenes)
- Custom datasets (implement custom dataset loader)

To use your own dataset, replace the `DummySegmentationDataset` in `train.py` with your dataset loader.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{gcha-net,
  author = {LittleForest-Ming},
  title = {GCHA-Net: Grid-based Channel-wise Hierarchical Attention Network},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/LittleForest-Ming/GCHA-Net}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PyTorch and PyTorch Lightning teams
- Vision Transformer and attention mechanism research community