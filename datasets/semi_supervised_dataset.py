
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import random
import numpy as np

class SemiSupervisedDataset(Dataset):
    def __init__(self, img_dir, mask_dir=None, paired_ratio=1.0, transform=None):
        """
        Semi-supervised dataset for Shirorekha segmentation
        
        Args:
            img_dir: Directory containing input images
            mask_dir: Directory containing ground truth masks (optional)
            paired_ratio: Ratio of paired data (0-1)
            transform: Image transformations
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.paired_ratio = paired_ratio
        self.transform = transform

        # Collect and sort image paths
        self.img_paths = sorted([
            os.path.join(img_dir, f) for f in os.listdir(img_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        
        if mask_dir:
            self.mask_paths = sorted([
                os.path.join(mask_dir, f) for f in os.listdir(mask_dir) 
                if f.lower().endswith(('.png',))
            ])
            # Validate same length
            assert len(self.mask_paths) == len(self.img_paths), \
                f"Mismatch between number of images ({len(self.img_paths)}) and masks ({len(self.mask_paths)})"
            
            # Select indices for paired data
            self.paired_indices = set(random.sample(
                range(len(self.img_paths)), 
                int(len(self.img_paths) * paired_ratio)
            ))
        else:
            self.mask_paths = None
            self.paired_indices = set()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        
        is_paired = 1 if idx in self.paired_indices else 0

        # Default dummy mask
        mask = torch.zeros((1, 256, 256))

        if is_paired and self.mask_paths:
            mask_path = self.mask_paths[idx]
            mask = Image.open(mask_path).convert('L')
            mask = np.array(mask, dtype=np.int64)
            mask = torch.from_numpy(mask).unsqueeze(0).long()

        if self.transform:
            img = self.transform(img)
        
        return img, mask, torch.tensor(is_paired)
