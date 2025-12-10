"""AgriScapes dataset loader with CULane-style annotations.

This module provides a PyTorch dataset class for loading lane detection data
with CULane-style annotations and converting them to polynomial parameters.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from utils.geometry import polynomial_fit_from_points


class AgriScapesDataset(Dataset):
    """Dataset for AgriScapes with CULane-style lane annotations.
    
    CULane annotation format:
    - Each image has a corresponding .txt file with the same name
    - Each line in the .txt file represents one lane
    - Each lane is a series of x,y coordinates: x1 y1 x2 y2 x3 y3 ...
    - If a y-coordinate is missing, it's marked as -2
    """
    
    def __init__(self, root_dir, split='train', image_height=288, image_width=800,
                 transform=None, max_lanes=4):
        """Initialize the dataset.
        
        Args:
            root_dir: Root directory containing images and annotations
            split: Dataset split ('train', 'val', 'test')
            image_height: Target height for resizing images
            image_width: Target width for resizing images
            transform: Optional transform to be applied on images
            max_lanes: Maximum number of lanes to process per image
        """
        self.root_dir = root_dir
        self.split = split
        self.image_height = image_height
        self.image_width = image_width
        self.transform = transform
        self.max_lanes = max_lanes
        
        # Paths
        self.image_dir = os.path.join(root_dir, split, 'images')
        self.anno_dir = os.path.join(root_dir, split, 'annotations')
        
        # Get list of images
        if os.path.exists(self.image_dir):
            self.image_files = sorted([f for f in os.listdir(self.image_dir) 
                                      if f.endswith(('.jpg', '.png', '.jpeg'))])
        else:
            # For testing without actual data
            self.image_files = []
        
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            image: Tensor of shape (3, H, W)
            targets: Dictionary containing:
                - 'lane_params': Tensor of shape (max_lanes, 3) with [k, m, b]
                - 'lane_valid': Tensor of shape (max_lanes,) indicating valid lanes
        """
        # Load image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        original_height, original_width = image.shape[:2]
        
        # Resize image
        image = cv2.resize(image, (self.image_width, self.image_height))
        
        # Load annotations
        anno_name = os.path.splitext(img_name)[0] + '.txt'
        anno_path = os.path.join(self.anno_dir, anno_name)
        
        lane_params = []
        
        if os.path.exists(anno_path):
            with open(anno_path, 'r') as f:
                for line_idx, line in enumerate(f):
                    if line_idx >= self.max_lanes:
                        break
                    
                    # Parse CULane format: x1 y1 x2 y2 ...
                    coords = line.strip().split()
                    
                    if len(coords) < 4:  # Need at least 2 points
                        continue
                    
                    # Extract points
                    points = []
                    for i in range(0, len(coords), 2):
                        x = float(coords[i])
                        y = float(coords[i + 1]) if i + 1 < len(coords) else -2
                        
                        # Skip invalid points
                        if x == -2 or y == -2:
                            continue
                        
                        # Scale points to resized image
                        x_scaled = x * self.image_width / original_width
                        y_scaled = y * self.image_height / original_height
                        
                        points.append([x_scaled, y_scaled])
                    
                    # Fit polynomial to points
                    if len(points) >= 3:
                        params = polynomial_fit_from_points(
                            points, self.image_height, self.image_width
                        )
                        lane_params.append(params)
        
        # Pad or truncate to max_lanes
        lane_valid = torch.zeros(self.max_lanes, dtype=torch.bool)
        
        if len(lane_params) > 0:
            lane_params = torch.stack(lane_params[:self.max_lanes])
            lane_valid[:len(lane_params)] = True
            
            # Pad if necessary
            if len(lane_params) < self.max_lanes:
                padding = torch.zeros(self.max_lanes - len(lane_params), 3)
                lane_params = torch.cat([lane_params, padding], dim=0)
        else:
            # No lanes found, create dummy tensor
            lane_params = torch.zeros(self.max_lanes, 3)
        
        # Convert image to tensor and normalize
        image = torch.from_numpy(image).float() / 255.0
        image = image.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
        
        if self.transform:
            image = self.transform(image)
        
        targets = {
            'lane_params': lane_params,
            'lane_valid': lane_valid
        }
        
        return image, targets
    
    @staticmethod
    def collate_fn(batch):
        """Custom collate function for batching.
        
        Args:
            batch: List of (image, targets) tuples
            
        Returns:
            images: Tensor of shape (N, 3, H, W)
            targets: Dictionary with batched targets
        """
        images = torch.stack([item[0] for item in batch])
        
        lane_params = torch.stack([item[1]['lane_params'] for item in batch])
        lane_valid = torch.stack([item[1]['lane_valid'] for item in batch])
        
        targets = {
            'lane_params': lane_params,
            'lane_valid': lane_valid
        }
        
        return images, targets


class DummyDataset(Dataset):
    """Dummy dataset for testing without actual data."""
    
    def __init__(self, num_samples=100, image_height=288, image_width=800, max_lanes=4):
        """Initialize dummy dataset.
        
        Args:
            num_samples: Number of samples to generate
            image_height: Image height
            image_width: Image width
            max_lanes: Maximum number of lanes
        """
        self.num_samples = num_samples
        self.image_height = image_height
        self.image_width = image_width
        self.max_lanes = max_lanes
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """Generate a random sample.
        
        Args:
            idx: Index (unused)
            
        Returns:
            image: Random image tensor of shape (3, H, W)
            targets: Random targets
        """
        # Random image
        image = torch.rand(3, self.image_height, self.image_width)
        
        # Random lane parameters
        num_lanes = np.random.randint(1, self.max_lanes + 1)
        lane_params = torch.zeros(self.max_lanes, 3)
        lane_valid = torch.zeros(self.max_lanes, dtype=torch.bool)
        
        for i in range(num_lanes):
            # Random polynomial parameters
            k = torch.randn(1) * 0.1
            m = torch.randn(1) * 0.3
            b = torch.rand(1) * 0.5 + 0.25
            lane_params[i] = torch.tensor([k, m, b])
            lane_valid[i] = True
        
        targets = {
            'lane_params': lane_params,
            'lane_valid': lane_valid
        }
        
        return image, targets
