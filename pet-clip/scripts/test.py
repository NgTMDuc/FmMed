import os
import sys
import numpy as np
import torch
from monai.data import CacheDataset
from torch.utils.data import DataLoader
from monai.transforms import MapTransform, Compose, ToTensord
from monai.config import KeysCollection
import time

# Your existing dataset and transform definitions

class LoadNpyPetImage(MapTransform):
    def __init__(self, keys: KeysCollection, dtype=np.float32, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            image = np.load(d[key])
            depth = image.shape[0]
            if depth % 4 != 0:
                padding_needed = 4 - (depth % 4)
                image = np.pad(image, ((0, padding_needed), (0, 0), (0, 0)), mode='constant', constant_values=0)

            # Normalize image to range [-1, 1]
            image = image.astype(np.float32)  # Ensure float32 for normalization
            image = image / 32767.0            # Normalize to [0, 1]
            image = 2 * image - 1              # Scale to [-1, 1]
            d[key] = image
        return d

class LoadTxtReport(MapTransform): 
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            with open(d[key], 'r', encoding='utf-8') as f:
                report_text = f.read().strip()
            d[key] = report_text
        return d

class PetReportPairDataset(CacheDataset):
    def __init__(self, 
                 root, 
                 split='train', 
                 transform= Compose([
                     LoadNpyPetImage(keys=["image"]), 
                     ToTensord('image'),
                     LoadTxtReport(keys=["label"])
                 ]),
                 cache_num=sys.maxsize,
                 cache_rate=0.2,
                 num_workers=4,):
        self.root = root
        self.split = split.lower()
        self.transform = transform
        self.month_folders = []
        for month in os.listdir(root):
            month_path = os.path.join(root, month)
            if not os.path.isdir(month_path):
                continue
            if self.split == 'train':
                if month in ['THANG 10', 'THANG 11', 'THANG 12']:
                    continue
                else:
                    self.month_folders.append(month_path)
            elif self.split == 'val':
                if month == 'THANG 10':
                    self.month_folders.append(month_path)
            elif self.split == 'test':
                if month in ['THANG 11', 'THANG 12']:
                    self.month_folders.append(month_path)
        allowed_modalities = ['abdomen_pelvis', 'chest', 'head_neck']
        
        # Build the list of (image_path, report_path) pairs.
        self.datalist = []
        for month_folder in self.month_folders:
            images_root = os.path.join(month_folder, 'images')
            reports_root = os.path.join(month_folder, 'reports')
            if not os.path.isdir(images_root) or not os.path.isdir(reports_root):
                continue
            for modality in allowed_modalities:
                modality_img_folder = os.path.join(images_root, modality)
                modality_rep_folder = os.path.join(reports_root, modality)
                if not os.path.isdir(modality_img_folder) or not os.path.isdir(modality_rep_folder):
                    continue
                # List all image files ending with .npy
                image_files = sorted([f for f in os.listdir(modality_img_folder) if f.endswith('.npy')])
                for img_file in image_files:
                    base_name = os.path.splitext(img_file)[0]
                    rep_file = base_name + '.txt'
                    img_file_path = os.path.join(modality_img_folder, img_file)
                    rep_file_path = os.path.join(modality_rep_folder, rep_file)
                    if os.path.exists(rep_file_path):
                        self.datalist.append({"image": img_file_path, "label": rep_file_path})
        
        super().__init__(
            data=self.datalist,
            transform=self.transform,
            cache_num=cache_num,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )

# Timing function
def measure_time(dataset, batch_size=1, num_batches=100000, cache=True):
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    start_time = time.time()
    
    for i, data in enumerate(data_loader):
        if i >= num_batches:
            break
    
    end_time = time.time()
    return end_time - start_time

# Test with CacheDataset
data_folder = "/home/user01/aiotlab/thaind/DAC001"  # Path to your root folder
dataset_with_cache = PetReportPairDataset(root=data_folder, split='train', cache_rate=0.1)
print("Testing with CacheDataset...")
time_with_cache = measure_time(dataset_with_cache)
print(f"Time taken with CacheDataset: {time_with_cache:.4f} seconds")

# Test without CacheDataset
class PetReportPairDatasetWithoutCache(torch.utils.data.Dataset):
    def __init__(self, root, split='train', transform=Compose([LoadNpyPetImage(keys=["image"]), LoadTxtReport(keys=["label"])])):
        self.root = root
        self.split = split.lower()
        self.transform = transform
        self.month_folders = []
        for month in os.listdir(root):
            month_path = os.path.join(root, month)
            if not os.path.isdir(month_path):
                continue
            if self.split == 'train':
                if month in ['THANG 10', 'THANG 11', 'THANG 12']:
                    continue
                else:
                    self.month_folders.append(month_path)
            elif self.split == 'val':
                if month == 'THANG 10':
                    self.month_folders.append(month_path)
            elif self.split == 'test':
                if month in ['THANG 11', 'THANG 12']:
                    self.month_folders.append(month_path)
        allowed_modalities = ['abdomen_pelvis', 'chest', 'head_neck']
        
        # Build the list of (image_path, report_path) pairs.
        self.datalist = []
        for month_folder in self.month_folders:
            images_root = os.path.join(month_folder, 'images')
            reports_root = os.path.join(month_folder, 'reports')
            if not os.path.isdir(images_root) or not os.path.isdir(reports_root):
                continue
            for modality in allowed_modalities:
                modality_img_folder = os.path.join(images_root, modality)
                modality_rep_folder = os.path.join(reports_root, modality)
                if not os.path.isdir(modality_img_folder) or not os.path.isdir(modality_rep_folder):
                    continue
                # List all image files ending with .npy
                image_files = sorted([f for f in os.listdir(modality_img_folder) if f.endswith('.npy')])
                for img_file in image_files:
                    base_name = os.path.splitext(img_file)[0]
                    rep_file = base_name + '.txt'
                    img_file_path = os.path.join(modality_img_folder, img_file)
                    rep_file_path = os.path.join(modality_rep_folder, rep_file)
                    if os.path.exists(rep_file_path):
                        self.datalist.append({"image": img_file_path, "label": rep_file_path})
    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = self.datalist[idx]
        return self.transform(data)

dataset_without_cache = PetReportPairDatasetWithoutCache(root=data_folder, split='train')
print("Testing without CacheDataset...")
time_without_cache = measure_time(dataset_without_cache)
print(f"Time taken without CacheDataset: {time_without_cache:.4f} seconds")

# Comparison
print(f"CacheDataset is {time_without_cache / time_with_cache:.2f} times faster than without CacheDataset.")