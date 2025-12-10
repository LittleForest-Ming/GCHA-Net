# Quick Start Guide - GCHA-Net

This guide helps you get started with GCHA-Net quickly.

## Installation

```bash
# Clone the repository
git clone https://github.com/LittleForest-Ming/GCHA-Net.git
cd GCHA-Net

# Install dependencies
pip install -r requirements.txt
```

## Quick Test

Run the test script to verify the installation:

```bash
python test_model.py
```

Expected output:
```
==================================================
GCHA-Net Implementation Tests
==================================================
...
✓ ALL TESTS PASSED!
==================================================
```

## Quick Examples

Run the examples script to see how to use different components:

```bash
python examples.py
```

This will demonstrate:
- Semantic segmentation
- Image classification
- Loading models from configuration
- Using GCHA attention layers directly

## Training

### Quick Training Run

To start training with default settings:

```bash
python train.py
```

This will:
- Use configuration from `config/default.yaml`
- Train on dummy data (for testing)
- Save checkpoints to `./checkpoints/`
- Log metrics to TensorBoard (in `logs/`)

### Customize Training

1. Edit `config/default.yaml` to modify hyperparameters
2. Replace `DummySegmentationDataset` in `train.py` with your actual dataset
3. Run training: `python train.py`

### Monitor Training

```bash
# In a separate terminal
tensorboard --logdir logs/
```

Then open http://localhost:6006 in your browser.

## Using Your Own Dataset

1. Create a dataset class similar to `DummySegmentationDataset` in `train.py`
2. Implement `__len__` and `__getitem__` methods
3. Update the dataset creation in `train.py`:

```python
train_dataset = YourDataset(
    root='path/to/data',
    split='train',
    transform=your_transforms
)
```

## Model Configuration

Key parameters in `config/default.yaml`:

### Model Architecture
- `num_blocks`: [2, 2, 6, 2] - Number of GCHA blocks per stage
- `num_heads`: [2, 4, 8, 16] - Attention heads per stage
- `base_channels`: 64 - Base channel dimension

### GCHA-Specific
- `N_total`: 256 - Total attention parameters
- `epsilon`: 1e-6 - Numerical stability
- `grid_size`: [7, 7] - Hierarchical grid size

### Training
- `learning_rate`: 0.0001 - Initial learning rate
- `batch_size`: 8 - Training batch size
- `num_epochs`: 100 - Total training epochs

## Simple Usage Example

```python
import torch
from models.gcha_net import GCHANet

# Create model
model = GCHANet(
    in_channels=3,
    num_classes=19,
    task='segmentation'
)

# Prepare input
image = torch.randn(1, 3, 512, 512)

# Inference
model.eval()
with torch.no_grad():
    output = model(image)  # Shape: (1, 19, 512, 512)

# Get predictions
predictions = torch.argmax(output, dim=1)
```

## Common Issues

### Out of Memory (OOM)
- Reduce `batch_size` in config
- Reduce image size in augmentation settings
- Use smaller model: reduce `base_channels` or `num_blocks`

### Slow Training
- Enable mixed precision: set `use_amp: true` in config
- Increase `num_workers` for data loading
- Use GPU: ensure `device: 'cuda'` in config

### Import Errors
- Make sure you're in the repository root directory
- Verify all dependencies are installed: `pip install -r requirements.txt`

## Next Steps

1. Read the full [README.md](README.md) for detailed documentation
2. Explore the code in `models/`, `modules/`, and `utils/`
3. Customize the architecture for your specific task
4. Replace dummy dataset with your actual data
5. Tune hyperparameters in `config/default.yaml`

## Getting Help

- Check the examples in `examples.py`
- Review the test cases in `test_model.py`
- Read the inline documentation in the source code

Happy training! 🚀
