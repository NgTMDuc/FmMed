from cnn_based import generate_model
from util import AverageMeter, count_params, init_log, collect_params, loadDataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
from tqdm import tqdm
import argparse
import yaml
import torch.nn.functional as F
import logging
import torch.nn as nn
from torch import optim
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class MedicalImageTransform:
    def __init__(self, fix_depth=140):
        self.fix_depth = fix_depth

    def __call__(self, image: torch.Tensor):
        """
        Process the image:
        - Normalize to [0, 1]
        - Resize depth to `fix_depth`
        - Convert (D, H, W) → (C, D, H, W)
        """

        # Normalize pixel values
        image = torch.tensor(image, dtype=torch.float32) / 32767.0

        # Add batch & channel dimensions (1, 1, D, H, W) for interpolation
        image = image.unsqueeze(0).unsqueeze(0)

        # Resize (D → fix_depth, H → 480, W → 480)
        image = F.interpolate(image, size=(self.fix_depth, 480, 480), mode='trilinear', align_corners=False)

        # Remove batch & channel dims (D, H, W)
        image = image.squeeze(0).squeeze(0)

        # Convert (D, H, W) → (1, D, H, W)  (1 channel for grayscale images)
        # image = image  # Final shape: (C, D, H, W)

        return image


transform = MedicalImageTransform(fix_depth=140)

def accuracy(model, val_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for img, label in tqdm(val_loader):
            img = img.to(device)
            label = label.to(device)
            label = label.argmax(dim = 1)
            print(label)
            output = model(img)
            
            preds = torch.argmax(output, dim = 1)
            print(output)
            all_preds.extend(preds.cpu().numpy()) 
            all_labels.extend(label.cpu().numpy())
            
    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')  # 'macro' for balanced class impact
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    
    return acc, f1, precision, recall

parser = argparse.ArgumentParser(description='Foundation Model for Vietnamese medical dataset')
parser.add_argument('--config', type=str, default = "/mnt/disk3/ducntm/FmMed/config/CtViT.yaml")
args = parser.parse_args()

cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
train_loader = loadDataset(cfg, "train", transform)
val_loader = loadDataset(cfg, "val", transform)
model = generate_model(18)
criteria = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(
        model.parameters(), lr = cfg["lr"]
    )
previous_best = 0.0
for i in range(50):
    model.train()
    total_loss = AverageMeter()
    for img, label in tqdm(train_loader):
        img = img.to(device)
        label = label.to(device)
        label = label.argmax(dim = 1)
        out = model(img)
        
        loss = criteria(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss.update(loss.item())
    print('===========> Epoch: {:}, Loss: {:.2f}'.format(i, total_loss.avg))
    acc, f1, pre, recal = accuracy(model, val_loader,  device)
    previous_best = max(acc, previous_best)
    print('===========> Epoch: {:}, Acc: {:.2f}, Best Acc: {:.2f}'.format(i, acc, previous_best))
    