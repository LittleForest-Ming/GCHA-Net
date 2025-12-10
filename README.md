# GCHA-Net

Geometry-Constrained Hierarchical Attention Network for Lane Detection

## Overview

GCHA-Net is a novel lane detection architecture that combines:
- **ResNet50 + FPN Backbone**: Feature extraction with multi-scale features
- **Geometry-Constrained Attention**: Custom attention mechanism that restricts attention to pixels near polynomial trajectories
- **Anchor-based Detection**: Detects lanes using a set of polynomial anchors with refinement

## Architecture

### Model Components

1. **Backbone (ResNet50 + FPN)**: Extracts unified feature maps from input images
2. **GCHA Decoder**: Implements geometry-constrained attention with learnable anchor queries
3. **Classification Head**: Binary classification for anchor validity (MLP)
4. **Regression Head**: Predicts parameter offsets (Δk, Δm, Δb) for anchor refinement (MLP)

### Geometric Constraint

The geometric logic uses polynomial trajectories defined as:
```
x = k*y² + m*y + b
```

Where:
- (x, y) are normalized coordinates in [0, 1]²
- y-axis is inverted for navigation perspective: ỹ = 1 - v/H
- Attention mask prevents focusing on pixels outside ε-distance from the polynomial

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

Train with dummy data (for testing):
```bash
python train.py --use_dummy --batch_size 4 --max_epochs 10
```

Train with real data:
```bash
python train.py --data_root /path/to/dataset --batch_size 8 --max_epochs 100
```

### Dataset Format

The dataset should follow CULane-style annotations:
```
data/
├── train/
│   ├── images/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── annotations/
│       ├── image1.txt
│       └── image2.txt
└── val/
    ├── images/
    └── annotations/
```

Annotation format (per line represents one lane):
```
x1 y1 x2 y2 x3 y3 ...
```

### Configuration

Key hyperparameters:
- `--num_anchors`: Number of polynomial anchors (default: 405 = 5×9×9)
- `--embed_dim`: Feature embedding dimension (default: 256)
- `--epsilon`: Geometric mask threshold (default: 0.05)
- `--focal_alpha`: Focal loss alpha (default: 0.25)
- `--focal_gamma`: Focal loss gamma (default: 2.0)

## Project Structure

```
GCHA-Net/
├── models/
│   ├── __init__.py
│   └── gcha_net.py          # Main model architecture
├── utils/
│   ├── __init__.py
│   └── geometry.py          # Geometric utilities
├── datasets/
│   ├── __init__.py
│   └── agroscapes.py        # Dataset loader
├── train.py                  # Training script
├── requirements.txt
└── README.md
```

## Implementation Details

### Loss Functions

1. **Focal Loss** for classification:
   - Addresses class imbalance between positive and negative anchors
   - FL(p_t) = -α_t(1-p_t)^γ log(p_t)

2. **Smooth L1 Loss** for regression:
   - Applied only to positive anchors
   - Predicts parameter offsets: Δk, Δm, Δb

### Anchor Matching

Anchors are matched to ground truth lanes using:
- L2 distance in parameter space (k, m, b)
- Threshold-based assignment (default: 0.3)
- Each anchor assigned to closest lane if within threshold

## Citation

If you use this code, please cite:

```bibtex
@article{gcha-net,
  title={GCHA-Net: Geometry-Constrained Hierarchical Attention Network for Lane Detection},
  author={Your Name},
  year={2024}
}
```

## License

Apache License 2.0