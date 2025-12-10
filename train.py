"""
Training script for GCHA-Net using PyTorch Lightning.

This script provides a minimal training loop setup with support for:
- Model training and validation
- Checkpointing
- Logging
- Configuration management
"""

import os
import sys
import argparse
from pathlib import Path
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.gcha_net import build_gcha_net


class DummySegmentationDataset(Dataset):
    """
    Dummy dataset for testing. Replace with actual dataset implementation.
    
    For real usage, implement a dataset that loads:
    - Agroscapes: Agricultural scene images with semantic segmentation
    - Cityscapes: Urban scene images with semantic segmentation
    """
    
    def __init__(self, size=100, image_size=(512, 512), num_classes=19):
        self.size = size
        self.image_size = image_size
        self.num_classes = num_classes
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Generate random image and mask
        image = torch.randn(3, *self.image_size)
        mask = torch.randint(0, self.num_classes, self.image_size)
        return image, mask


class GCHANetLightning(pl.LightningModule):
    """
    PyTorch Lightning module for GCHA-Net.
    
    Handles training, validation, and optimization.
    """
    
    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        
        # Build model
        self.model = build_gcha_net(config['model'])
        
        # Loss function
        ignore_index = config['training']['loss'].get('ignore_index', 255)
        label_smoothing = config['training']['loss'].get('label_smoothing', 0.0)
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing
        )
        
        # Metrics
        self.train_loss = []
        self.val_loss = []
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, masks = batch
        
        # Forward pass
        outputs = self(images)
        
        # Compute loss
        loss = self.criterion(outputs, masks)
        
        # Compute accuracy
        preds = torch.argmax(outputs, dim=1)
        accuracy = (preds == masks).float().mean()
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', accuracy, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, masks = batch
        
        # Forward pass
        outputs = self(images)
        
        # Compute loss
        loss = self.criterion(outputs, masks)
        
        # Compute accuracy
        preds = torch.argmax(outputs, dim=1)
        accuracy = (preds == masks).float().mean()
        
        # Log metrics
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', accuracy, on_epoch=True, prog_bar=True)
        
        return {'val_loss': loss, 'val_acc': accuracy}
    
    def configure_optimizers(self):
        # Optimizer
        optimizer_config = self.config['training']['optimizer']
        optimizer_type = optimizer_config['type']
        
        if optimizer_type == 'AdamW':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay'],
                betas=optimizer_config['betas']
            )
        elif optimizer_type == 'Adam':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
        # Learning rate scheduler
        scheduler_config = self.config['training']['scheduler']
        scheduler_type = scheduler_config['type']
        
        if scheduler_type == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=scheduler_config['T_max'],
                eta_min=scheduler_config['eta_min']
            )
        elif scheduler_type == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=scheduler_config.get('step_size', 30),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        else:
            scheduler = None
        
        if scheduler is not None:
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        else:
            return optimizer


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train GCHA-Net')
    parser.add_argument(
        '--config',
        type=str,
        default='config/default.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--gpus',
        type=int,
        default=None,
        help='Number of GPUs to use (overrides config)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of epochs (overrides config)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Batch size (overrides config)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Learning rate (overrides config)'
    )
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.gpus is not None:
        config['system']['gpus'] = args.gpus
    if args.epochs is not None:
        config['training']['num_epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.lr is not None:
        config['training']['learning_rate'] = args.lr
    
    # Set random seed
    pl.seed_everything(config['system']['seed'])
    
    # Create datasets
    # NOTE: Replace DummySegmentationDataset with actual dataset implementation
    train_dataset = DummySegmentationDataset(
        size=config.get('train_size', 100),
        image_size=tuple(config['data']['image_size']),
        num_classes=config['model']['num_classes']
    )
    val_dataset = DummySegmentationDataset(
        size=config.get('val_size', 20),
        image_size=tuple(config['data']['image_size']),
        num_classes=config['model']['num_classes']
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['system']['num_workers'],
        pin_memory=config['system']['pin_memory']
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['system']['num_workers'],
        pin_memory=config['system']['pin_memory']
    )
    
    # Create model
    model = GCHANetLightning(config)
    
    # Create logger
    logger = TensorBoardLogger(
        save_dir=config['logging']['log_dir'],
        name='gcha_net'
    )
    
    # Create callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=config['logging']['checkpoint_dir'],
        filename='gcha_net-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min',
        every_n_epochs=config['logging']['save_freq']
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['num_epochs'],
        accelerator='gpu' if config['system']['gpus'] > 0 else 'cpu',
        devices=config['system']['gpus'] if config['system']['gpus'] > 0 else 1,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        precision=config['system']['precision'],
        log_every_n_steps=config['logging']['log_freq'],
        check_val_every_n_epoch=config['validation']['val_freq']
    )
    
    # Train model
    print("Starting training...")
    print(f"Model: GCHA-Net")
    print(f"Dataset: {config['data']['dataset']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print(f"Epochs: {config['training']['num_epochs']}")
    print(f"N_total (anchors): {config['model']['n_total']}")
    print(f"Epsilon: {config['model']['epsilon']}")
    
    trainer.fit(model, train_loader, val_loader)
    
    print("Training completed!")
    print(f"Best model checkpoint: {checkpoint_callback.best_model_path}")


if __name__ == '__main__':
    main()
