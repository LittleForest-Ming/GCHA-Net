"""Training script for GCHA-Net using PyTorch Lightning.

This script implements the training loop with:
- Focal Loss for anchor classification
- Smooth L1 Loss for parameter regression
- PyTorch Lightning for training management
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from models.gcha_net import GCHANet
from datasets.agroscapes import AgriScapesDataset, DummyDataset
from utils.geometry import generate_anchors


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor in range (0, 1) for class balance
            gamma: Exponent of modulating factor (1 - p_t)
            reduction: 'none', 'mean', or 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """Compute focal loss.
        
        Args:
            inputs: Predictions of shape (N, *)
            targets: Ground truth of shape (N, *)
            
        Returns:
            loss: Focal loss value
        """
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Get probabilities
        pt = torch.exp(-bce_loss)
        
        # Compute focal term: (1 - pt)^gamma
        focal_term = (1 - pt) ** self.gamma
        
        # Compute focal loss
        focal_loss = self.alpha * focal_term * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class GCHANetLightning(pl.LightningModule):
    """PyTorch Lightning module for GCHA-Net."""
    
    def __init__(self, config):
        """Initialize the Lightning module.
        
        Args:
            config: Configuration dictionary with hyperparameters
        """
        super().__init__()
        self.save_hyperparameters(config)
        
        # Model
        self.model = GCHANet(
            num_anchors=config.get('num_anchors', 405),
            embed_dim=config.get('embed_dim', 256),
            num_decoder_layers=config.get('num_decoder_layers', 3),
            num_heads=config.get('num_heads', 8),
            dropout=config.get('dropout', 0.1),
            epsilon=config.get('epsilon', 0.05)
        )
        
        # Loss functions
        self.focal_loss = FocalLoss(
            alpha=config.get('focal_alpha', 0.25),
            gamma=config.get('focal_gamma', 2.0)
        )
        self.smooth_l1_loss = nn.SmoothL1Loss()
        
        # Loss weights
        self.cls_weight = config.get('cls_weight', 1.0)
        self.reg_weight = config.get('reg_weight', 1.0)
        
        # Anchors for matching
        self.anchors = generate_anchors()
        
    def forward(self, x):
        """Forward pass."""
        return self.model(x)
    
    def compute_loss(self, cls_logits, reg_deltas, targets):
        """Compute combined loss.
        
        Args:
            cls_logits: Classification logits of shape (N, num_anchors)
            reg_deltas: Regression deltas of shape (N, num_anchors, 3)
            targets: Dictionary with 'lane_params' and 'lane_valid'
            
        Returns:
            loss: Total loss
            loss_dict: Dictionary with individual losses
        """
        batch_size = cls_logits.shape[0]
        num_anchors = cls_logits.shape[1]
        device = cls_logits.device
        
        lane_params = targets['lane_params']  # [N, max_lanes, 3]
        lane_valid = targets['lane_valid']    # [N, max_lanes]
        
        # Match anchors to ground truth lanes
        cls_targets, reg_targets, reg_mask = self.match_anchors_to_lanes(
            lane_params, lane_valid
        )
        
        cls_targets = cls_targets.to(device)
        reg_targets = reg_targets.to(device)
        reg_mask = reg_mask.to(device)
        
        # Classification loss (Focal Loss)
        cls_loss = self.focal_loss(cls_logits, cls_targets)
        
        # Regression loss (Smooth L1 Loss) - only for positive anchors
        if reg_mask.sum() > 0:
            reg_loss = self.smooth_l1_loss(
                reg_deltas[reg_mask],
                reg_targets[reg_mask]
            )
        else:
            reg_loss = torch.tensor(0.0, device=device)
        
        # Total loss
        total_loss = self.cls_weight * cls_loss + self.reg_weight * reg_loss
        
        loss_dict = {
            'loss': total_loss,
            'cls_loss': cls_loss,
            'reg_loss': reg_loss
        }
        
        return total_loss, loss_dict
    
    def match_anchors_to_lanes(self, lane_params, lane_valid):
        """Match anchors to ground truth lanes.
        
        Simple matching strategy: assign each anchor to the closest lane
        if the distance is below a threshold.
        
        Args:
            lane_params: Ground truth lane parameters [N, max_lanes, 3]
            lane_valid: Validity mask [N, max_lanes]
            
        Returns:
            cls_targets: Binary classification targets [N, num_anchors]
            reg_targets: Regression targets [N, num_anchors, 3]
            reg_mask: Mask for valid regression targets [N, num_anchors]
        """
        batch_size = lane_params.shape[0]
        max_lanes = lane_params.shape[1]
        num_anchors = self.anchors.shape[0]
        
        cls_targets = torch.zeros(batch_size, num_anchors)
        reg_targets = torch.zeros(batch_size, num_anchors, 3)
        reg_mask = torch.zeros(batch_size, num_anchors, dtype=torch.bool)
        
        # Expand anchors for batch
        anchors_expanded = self.anchors.unsqueeze(0).unsqueeze(0)
        # [1, 1, num_anchors, 3]
        
        for i in range(batch_size):
            valid_lanes = lane_valid[i]
            
            if not valid_lanes.any():
                continue
            
            valid_params = lane_params[i][valid_lanes]  # [num_valid, 3]
            
            # Compute distance between each anchor and each valid lane
            # Use L2 distance in parameter space
            distances = torch.cdist(
                self.anchors,  # [num_anchors, 3]
                valid_params   # [num_valid, 3]
            )  # [num_anchors, num_valid]
            
            # Find closest lane for each anchor
            min_distances, closest_lane_idx = distances.min(dim=1)
            
            # Threshold for positive assignment
            threshold = 0.3
            positive_mask = min_distances < threshold
            
            # Set classification targets
            cls_targets[i, positive_mask] = 1.0
            
            # Set regression targets (deltas from anchor to closest lane)
            reg_targets[i, positive_mask] = (
                valid_params[closest_lane_idx[positive_mask]] - 
                self.anchors[positive_mask]
            )
            reg_mask[i, positive_mask] = True
        
        return cls_targets, reg_targets, reg_mask
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        images, targets = batch
        
        # Forward pass
        cls_logits, reg_deltas = self(images)
        
        # Compute loss
        loss, loss_dict = self.compute_loss(cls_logits, reg_deltas, targets)
        
        # Log metrics
        self.log('train_loss', loss_dict['loss'], prog_bar=True)
        self.log('train_cls_loss', loss_dict['cls_loss'])
        self.log('train_reg_loss', loss_dict['reg_loss'])
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        images, targets = batch
        
        # Forward pass
        cls_logits, reg_deltas = self(images)
        
        # Compute loss
        loss, loss_dict = self.compute_loss(cls_logits, reg_deltas, targets)
        
        # Log metrics
        self.log('val_loss', loss_dict['loss'], prog_bar=True)
        self.log('val_cls_loss', loss_dict['cls_loss'])
        self.log('val_reg_loss', loss_dict['reg_loss'])
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.get('learning_rate', 1e-4),
            weight_decay=self.hparams.get('weight_decay', 1e-4)
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.get('max_epochs', 100),
            eta_min=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train GCHA-Net')
    
    # Data arguments
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Root directory for dataset')
    parser.add_argument('--use_dummy', action='store_true',
                        help='Use dummy dataset for testing')
    parser.add_argument('--image_height', type=int, default=288,
                        help='Image height')
    parser.add_argument('--image_width', type=int, default=800,
                        help='Image width')
    parser.add_argument('--max_lanes', type=int, default=4,
                        help='Maximum number of lanes per image')
    
    # Model arguments
    parser.add_argument('--num_anchors', type=int, default=405,
                        help='Number of anchor polynomials')
    parser.add_argument('--embed_dim', type=int, default=256,
                        help='Embedding dimension')
    parser.add_argument('--num_decoder_layers', type=int, default=3,
                        help='Number of decoder layers')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout probability')
    parser.add_argument('--epsilon', type=float, default=0.05,
                        help='Geometric mask threshold')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--max_epochs', type=int, default=100,
                        help='Maximum number of epochs')
    parser.add_argument('--focal_alpha', type=float, default=0.25,
                        help='Focal loss alpha')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal loss gamma')
    parser.add_argument('--cls_weight', type=float, default=1.0,
                        help='Classification loss weight')
    parser.add_argument('--reg_weight', type=float, default=1.0,
                        help='Regression loss weight')
    
    # Other arguments
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--accelerator', type=str, default='auto',
                        help='Accelerator type (cpu, gpu, auto)')
    parser.add_argument('--devices', type=int, default=1,
                        help='Number of devices')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create datasets
    if args.use_dummy:
        print("Using dummy dataset for testing")
        train_dataset = DummyDataset(
            num_samples=100,
            image_height=args.image_height,
            image_width=args.image_width,
            max_lanes=args.max_lanes
        )
        val_dataset = DummyDataset(
            num_samples=20,
            image_height=args.image_height,
            image_width=args.image_width,
            max_lanes=args.max_lanes
        )
    else:
        train_dataset = AgriScapesDataset(
            root_dir=args.data_root,
            split='train',
            image_height=args.image_height,
            image_width=args.image_width,
            max_lanes=args.max_lanes
        )
        val_dataset = AgriScapesDataset(
            root_dir=args.data_root,
            split='val',
            image_height=args.image_height,
            image_width=args.image_width,
            max_lanes=args.max_lanes
        )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    config = vars(args)
    model = GCHANetLightning(config)
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, 'checkpoints'),
        filename='gcha-net-{epoch:02d}-{val_loss:.4f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min'
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Logger
    logger = TensorBoardLogger(
        save_dir=args.output_dir,
        name='gcha_net_logs'
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        log_every_n_steps=10,
        gradient_clip_val=1.0
    )
    
    # Train
    trainer.fit(model, train_loader, val_loader)
    
    print(f"Training completed! Checkpoints saved to {args.output_dir}")


if __name__ == '__main__':
    main()
