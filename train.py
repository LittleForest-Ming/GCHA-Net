"""
Training script for GCHA-Net using PyTorch Lightning.
Minimal training loop setup for semantic segmentation or classification tasks.
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from typing import Optional, Dict, Any
import numpy as np
from pathlib import Path

from models.gcha_net import build_gcha_net, GCHANet


class DummySegmentationDataset(Dataset):
    """
    Dummy dataset for testing the training pipeline.
    Replace with actual dataset loader (e.g., Agroscapes, Cityscapes).
    """
    
    def __init__(
        self,
        num_samples: int = 100,
        image_size: tuple = (512, 512),
        num_classes: int = 19,
        transform=None
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes
        self.transform = transform
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random image and mask
        image = torch.randn(3, *self.image_size)
        mask = torch.randint(0, self.num_classes, self.image_size)
        
        if self.transform:
            image, mask = self.transform(image, mask)
        
        return image, mask


class GCHANetLightning(pl.LightningModule):
    """
    PyTorch Lightning module for GCHA-Net training.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super(GCHANetLightning, self).__init__()
        
        self.config = config
        self.save_hyperparameters(config)
        
        # Build model
        self.model = build_gcha_net(config['model'])
        
        # Loss function
        if config['training']['loss'] == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss(
                ignore_index=255,
                label_smoothing=config['training'].get('label_smoothing', 0.0)
            )
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        
        # Metrics
        self.train_acc = []
        self.val_acc = []
    
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
        acc = (preds == masks).float().mean()
        
        # Log metrics
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, masks = batch
        
        # Forward pass
        outputs = self(images)
        
        # Compute loss
        loss = self.criterion(outputs, masks)
        
        # Compute accuracy
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == masks).float().mean()
        
        # Log metrics
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return {'val_loss': loss, 'val_acc': acc}
    
    def configure_optimizers(self):
        # Optimizer
        optimizer_name = self.config['training']['optimizer'].lower()
        lr = self.config['training']['learning_rate']
        weight_decay = self.config['training']['weight_decay']
        
        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=self.config['training']['betas']
            )
        elif optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=self.config['training']['betas']
            )
        elif optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=lr,
                momentum=self.config['training']['momentum'],
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        # Learning rate scheduler
        scheduler_name = self.config['training']['scheduler'].lower()
        num_epochs = self.config['training']['num_epochs']
        warmup_epochs = self.config['training']['warmup_epochs']
        
        if scheduler_name == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=num_epochs - warmup_epochs,
                eta_min=self.config['training']['min_lr']
            )
        elif scheduler_name == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=num_epochs // 3,
                gamma=0.1
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=num_epochs,
                eta_min=self.config['training']['min_lr']
            )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main training function."""
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'default.yaml')
    config = load_config(config_path)
    
    # Set random seed
    pl.seed_everything(config['environment']['seed'])
    
    # Create datasets
    train_dataset = DummySegmentationDataset(
        num_samples=100,
        image_size=tuple(config['data']['augmentation']['random_crop']),
        num_classes=config['model']['num_classes']
    )
    
    val_dataset = DummySegmentationDataset(
        num_samples=20,
        image_size=tuple(config['data']['augmentation']['random_crop']),
        num_classes=config['model']['num_classes']
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['validation']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    # Create model
    model = GCHANetLightning(config)
    
    # Callbacks
    checkpoint_dir = config['checkpoint']['save_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='gcha-net-{epoch:02d}-{val/loss:.4f}',
        save_top_k=config['checkpoint']['keep_last'],
        monitor='val/loss',
        mode='min',
        save_last=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Logger
    logger = None
    if config['logging']['tensorboard']:
        logger = TensorBoardLogger(
            save_dir='logs',
            name=config['logging']['experiment_name']
        )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['num_epochs'],
        accelerator='gpu' if torch.cuda.is_available() and config['environment']['device'] == 'cuda' else 'cpu',
        devices=config['environment']['gpu_ids'] if torch.cuda.is_available() else 1,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        log_every_n_steps=config['logging']['log_interval'],
        precision=16 if config['training']['use_amp'] else 32,
        gradient_clip_val=config['training']['grad_clip']
    )
    
    # Train
    print("Starting training...")
    print(f"Model: GCHA-Net")
    print(f"Dataset: {config['data']['dataset']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print(f"Epochs: {config['training']['num_epochs']}")
    print("-" * 50)
    
    trainer.fit(model, train_loader, val_loader)
    
    print("Training completed!")
    print(f"Best model saved to: {checkpoint_callback.best_model_path}")


if __name__ == '__main__':
    main()
