# GCHA-Net Architecture Documentation --- Old Version: Does not involve cubic offset

## Overview

GCHA-Net (Geometry-Constrained Hierarchical Attention Network) is a novel deep learning architecture for lane detection that incorporates geometric constraints directly into the attention mechanism.

## Model Statistics

- **Total Parameters**: 30,380,612 (~30.4M)
- **Input Size**: (Batch, 3, 288, 800) - RGB images
- **Output**: 
  - Classification logits: (Batch, 405) - Binary classification for each anchor
  - Regression deltas: (Batch, 405, 3) - Parameter offsets (Δk, Δm, Δb)

## Architecture Components

### 1. Backbone: ResNet18/34/50 + FPN

The backbone extracts multi-scale features from input images:

```
Input (3, 288, 800)
  ↓
ResNet50 Conv1 + BN + ReLU + MaxPool
  ↓
ResNet50 Layer1 (C1: 256 channels, 1/4 resolution)
  ↓
ResNet50 Layer2 (C2: 512 channels, 1/8 resolution)
  ↓
ResNet50 Layer3 (C3: 1024 channels, 1/16 resolution)
  ↓
ResNet50 Layer4 (C4: 2048 channels, 1/32 resolution)
  ↓
Feature Pyramid Network (FPN)
  ↓
Unified Features (256 channels, 1/4 resolution)
```

**Key Features**:
- Pretrained on ImageNet for better feature extraction
- FPN combines multi-scale information
- Output feature map at 1/4 original resolution (72x200 for 288x800 input)

### 2. Feature Projection

```
FPN Output (256, H', W')
  ↓
1x1 Convolution
  ↓
Projected Features (embed_dim=256, H', W')
  ↓
Reshape to (H'×W', embed_dim) for attention
```

### 3. Anchor System

**Anchor Generation**:
- 3D grid of polynomial parameters (k, m, b)
- Default: 5 × 9 × 9 = 405 anchors
- Parameter ranges:
  - k ∈ [-0.5, 0.5] - quadratic coefficient
  - m ∈ [-1.0, 1.0] - linear coefficient  
  - b ∈ [0.0, 1.0] - constant coefficient

**Polynomial Definition**:
```
x = k·y² + m·y + b
```
where (x, y) are normalized coordinates in [0, 1]²

### 4. Geometry-Constrained Attention

**Key Innovation**: Replaces standard cross-attention with geometric constraints.

#### Coordinate Transform
```
Pixel coordinates (u, v) → Normalized coordinates (x̃, ỹ)

x̃ = u / W
ỹ = 1 - v / H    # Inverted Y-axis for navigation perspective
```

#### Geometric Mask Generation
For each anchor (k, m, b):
```
1. Compute expected x-coordinate: x_poly = k·ỹ² + m·ỹ + b
2. Compute distance: d = |x̃ - x_poly|
3. Set mask:
   - 0 if d < ε (within trajectory)
   - -∞ if d ≥ ε (outside trajectory)
```

#### Attention Mechanism
```
Query: Learnable anchor embeddings (405, embed_dim)
Key: Spatial features (H'×W', embed_dim)
Value: Spatial features (H'×W', embed_dim)

Attention = Softmax((Q·K^T) / √d_k + GeometricMask)
Output = Attention · V
```

**Multi-head Attention**:
- 8 heads by default
- Head dimension: embed_dim / num_heads = 32
- Scale factor: 1/√32 ≈ 0.177

### 5. GCHA Decoder

**Architecture**: 3 decoder layers (default)

Each layer contains:
```
1. Self-Attention Block:
   Input → MultiheadAttention → Add & Norm

2. Cross-Attention Block (with Geometric Constraint):
   Input → GeometryConstrainedAttention → Add & Norm

3. Feedforward Block:
   Input → Linear(embed_dim, 1024) → ReLU → Linear(1024, embed_dim) → Add & Norm
```

**Information Flow**:
```
Anchor Queries (405, embed_dim)
  ↓
Layer 1: Self-Attn → Cross-Attn (Geometric) → FFN
  ↓
Layer 2: Self-Attn → Cross-Attn (Geometric) → FFN
  ↓
Layer 3: Self-Attn → Cross-Attn (Geometric) → FFN
  ↓
Final LayerNorm
  ↓
Decoded Features (405, embed_dim)
```

### 6. Prediction Heads

#### Classification Head (Binary)
```
Input (405, embed_dim)
  ↓
Linear(embed_dim, embed_dim) → ReLU → Dropout(0.1)
  ↓
Linear(embed_dim, embed_dim/2) → ReLU → Dropout(0.1)
  ↓
Linear(embed_dim/2, 1)
  ↓
Logits (405,) - one per anchor
```

**Purpose**: Predicts which anchors correspond to actual lanes

#### Regression Head (3 Parameters)
```
Input (405, embed_dim)
  ↓
Linear(embed_dim, embed_dim) → ReLU → Dropout(0.1)
  ↓
Linear(embed_dim, embed_dim/2) → ReLU → Dropout(0.1)
  ↓
Linear(embed_dim/2, 3)
  ↓
Deltas (405, 3) - (Δk, Δm, Δb) per anchor
```

**Purpose**: Refines anchor parameters to fit actual lanes

**Anchor Refinement**:
```python
refined_params = anchor_params + regression_deltas
```

## Training Strategy

### Loss Functions

#### 1. Focal Loss (Classification)
```
FL(p_t) = -α·(1 - p_t)^γ · log(p_t)

where:
- α = 0.25 (class balance weight)
- γ = 2.0 (focusing parameter)
- p_t = predicted probability for true class
```

**Purpose**: Addresses class imbalance (most anchors are negative)

#### 2. Smooth L1 Loss (Regression)
```
L_reg = Σ_positive_anchors SmoothL1(predicted_delta, target_delta)
```

**Applied only to positive anchors** (matched to ground truth lanes)

#### 3. Combined Loss
```
L_total = λ_cls · L_cls + λ_reg · L_reg

where:
- λ_cls = 1.0 (classification weight)
- λ_reg = 1.0 (regression weight)
```

### Anchor Matching

**Strategy**: Closest lane assignment with threshold

```
For each anchor:
1. Compute L2 distance to all ground truth lanes in parameter space
2. Find closest lane
3. If distance < threshold (0.3):
   - Mark as positive
   - Assign regression target as (gt_params - anchor_params)
4. Else:
   - Mark as negative
```

### Optimization

- **Optimizer**: AdamW
- **Learning Rate**: 1e-4
- **Weight Decay**: 1e-4
- **Scheduler**: CosineAnnealingLR (to 1e-6)
- **Gradient Clipping**: 1.0

## Inference Pipeline

```
1. Input Image (3, 288, 800)
   ↓
2. Backbone: Extract features
   ↓
3. Generate geometric masks for all anchors
   ↓
4. GCHA Decoder: Refine anchor features with geometric constraints
   ↓
5. Classification Head: Predict anchor scores
   ↓
6. Regression Head: Predict parameter deltas
   ↓
7. Post-processing:
   - Apply NMS on anchor scores
   - Refine anchor parameters
   - Convert to lane coordinates
```

## Key Design Choices

### Why Polynomial Representation?
- **Compact**: 3 parameters vs. many points
- **Smooth**: Natural representation for lanes
- **Geometric**: Enables constraint-based attention

### Why Inverted Y-axis?
- **Navigation perspective**: y=1 at bottom, y=0 at top
- **Intuitive**: Matches how lanes appear in driving scenarios
- **Consistent**: Standard in lane detection literature

### Why Geometry-Constrained Attention?
- **Focus**: Attention restricted to relevant pixels near trajectory
- **Efficiency**: Reduces computational cost of attention
- **Interpretability**: Attention pattern follows lane geometry

## Extensibility

The architecture can be extended by:

1. **Different backbones**: Replace ResNet50 with EfficientNet, Swin, etc.
2. **More anchors**: Increase grid density for complex scenarios
3. **Additional features**: Add temporal information for video
4. **Multi-task**: Add auxiliary tasks like drivable area segmentation
5. **Ensemble**: Combine multiple anchor grids with different ranges

## Performance Considerations

- **Memory**: ~30M parameters + activations
- **FLOPs**: Dominated by backbone and attention
- **Bottlenecks**:
  - Geometric mask generation (computed every forward pass)
  - Cross-attention (405 × H'W' interactions)
- **Optimizations**:
  - Cache geometric masks for fixed input size
  - Use flash attention for memory efficiency
  - Quantization for deployment

## Citation

```bibtex
@article{gcha-net,
  title={GCHA-Net: Geometry-Constrained Hierarchical Attention Network for Lane Detection},
  author={Your Name},
  year={2024}
}
```
