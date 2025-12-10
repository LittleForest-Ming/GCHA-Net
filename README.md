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