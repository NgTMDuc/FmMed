import pandas as pd 
import numpy as np 
import torch 
from torch.utils.data import Dataset, DataLoader
import os
import torch.nn.functional as F
class MedicalImageTransform:
    def __init__(self, fix_depth=140):
        self.fix_depth = fix_depth

    def __call__(self, image: torch.Tensor):
        """
        Process the image:
        - Normalize pixel values
        - Resize depth to `fix_depth`
        - Ensure final shape is (D, H, W, C) with C=3
        """

        # Normalize pixel values (assuming int16 image)
        image = torch.as_tensor(image, dtype=torch.float32) / 32767.0

        # Add batch & channel dimensions (1, 1, D, H, W)
        image = image.unsqueeze(0).unsqueeze(0)

        # Resize (D → fix_depth)
        image = F.interpolate(image, size=(self.fix_depth, 480, 480), mode='trilinear', align_corners=False)

        # Remove batch dim, keep channel dim → Shape becomes (1, D, H, W)
        image = image.squeeze(0)  # Keep (C, D, H, W)

        # Expand to (D, H, W, 3) if needed
        image = image.permute(1, 2, 3, 0).expand(-1, -1, -1, 3)  # (D, H, W, 3)

        return image  # Correct shape: (D, H, W, 3)
transform = MedicalImageTransform()
class MedicalDataset(Dataset):
    def __init__(self, paths= "train.txt"):
        super().__init__()

        # self.root = root
        
        with open(paths, "r") as f:
            lines = f.readlines()
            self.samples = [line.strip() for line in lines]
        
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        print(path)
        return transform(np.load(path))


def dataloader(paths):
    dataset = MedicalDataset(paths)
    dataloader = DataLoader(dataset, batch_size = 32, shuffle = True)
    return dataloader
        
