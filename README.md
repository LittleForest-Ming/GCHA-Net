# GCHA-Net

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

## Installation

```bash
# Clone the repository
git clone https://github.com/LittleForest-Ming/GCHA-Net.git
cd GCHA-Net

# Install dependencies
pip install torch
```

## Usage

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

## Citation

If you use this code in your research, please cite:

```bibtex
@software{gcha_net,
  title={GCHA-Net: Geometry-Constrained Hough Attention Network},
  author={LittleForest-Ming},
  year={2024},
  url={https://github.com/LittleForest-Ming/GCHA-Net}
}
```

## Contact

For questions or issues, please open an issue on GitHub.