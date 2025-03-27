import os
import numpy as np 
import torch 
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F 

def process_image(image: np.ndarray,
                  fix_depth = 140
                  ):
    D, H, W = image.shape
    
    image_tensor = torch.tensor(image, dtype=torch.float32) / 32767.0
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
    image_tensor = F.interpolate(image_tensor, size=(fix_depth, 480, 480), mode='trilinear', align_corners=False)
    image_tensor = image_tensor.squeeze(0).squeeze(0)
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor

class MedicalImage(Dataset):
    def __init__(self, root, split = "train", transform = None):
        self.root = root
        self.split = split.lower()
        self.transform = transform
        
        self.month_folders = []
        for month in os.listdir(root):
            month_path = os.path.join(root, month)
            if not os.path.isdir(month_path):
                continue
            if self.split == "train":
                if month in ['THANG 10', 'THANG 11', 'THANG 12']:
                    continue
                else:
                    self.month_folders.append(month_path)
            elif self.split == 'val':
                if month == 'THANG 10':
                     self.month_folders.append(month_path)
            elif self.split == "test":
                if month in ['THANG 11', 'THANG 12']:
                    self.month_folders.append(month_path)
        
        allowed_modalities = ['abdomen_pelvis', 'chest', 'head_neck']
        self.allowed_modalities = allowed_modalities
        #Load the image and label
        self.samples = []
        for month_folder in self.month_folders:
            images_root = os.path.join(month_folder, "images")
            # reports_root = os.path.join(month_folder, "reports")
            if not os.path.isdir(images_root):
                continue
            
            for modality in allowed_modalities:
                modality_img_folder = os.path.join(images_root, modality) 
                # modality_rep_folder = os.path.join(reports_root, modality)
                if not os.path.isdir(modality_img_folder):
                    continue
                # List all image files ending with .npy
                image_files = sorted([f for f in os.listdir(modality_img_folder) if f.endswith('.npy')])
                for img_file in image_files:
                    base_name = os.path.splitext(img_file)[0]
                    # rep_file = base_name + '.txt'
                    img_file_path = os.path.join(modality_img_folder, img_file)
                    self.samples.append((img_file_path, modality))
                    # rep_file_path = os.path.join(modality_rep_folder, rep_file)
                    # if os.path.exists(rep_file_path):
                        # self.samples.append((img_file_path, rep_file_path))
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, modal = self.samples[idx]
        image = np.load(img_path)
        
        if self.transform:
            image = self.transform(image)
        else:
            image = process_image(image)
        
        label = np.array([1 if modal == m else 0 for m in self.allowed_modalities], dtype=np.float32)
        return image
        return image, label

def loadDataset(cfg, split, transform = None):
    root = cfg["root"]
    
    dataset = MedicalImage(root, split, transform)
    dataloader = DataLoader(
        dataset, 
        num_workers = 4,
        batch_size = cfg["batch_size"],
        shuffle = True,
    )
    
    return dataloader
    