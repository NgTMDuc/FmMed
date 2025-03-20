from cnn_based import generate_model
from util import AverageMeter, count_params, init_log, collect_params, loadDataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
from tqdm import tqdm
import argparse
import yaml
import logging
import torch.nn as nn
from torch import optim
from torchvision import transforms
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class NormalizeImage:
    """Normalize image pixel values to [0, 1] by dividing by 32767."""
    def __call__(self, image):
        return torch.tensor(image, dtype=torch.float32) / 32767.0

class ResizeDepth:
    """Resize the depth dimension to a fixed size using trilinear interpolation."""
    def __init__(self, fix_depth=140):
        self.fix_depth = fix_depth

    def __call__(self, image):
        # Reshape to (1, 1, D, H, W) for interpolation (N, C, D, H, W)
        image = image.unsqueeze(0).unsqueeze(0)
        image = F.interpolate(image, size=(self.fix_depth, 480, 480), mode='trilinear', align_corners=False)
        return image.squeeze(0).squeeze(0)  # Remove batch & channel dimensions → (D, H, W)

class RearrangeAxes:
    """Rearrange axes from (D, H, W) to (H, W, D)."""
    def __call__(self, image):
        return image.permute(1, 2, 0)  # (D, H, W) → (H, W, D)

medical_transform = transforms.Compose([
    NormalizeImage(),  # Normalize pixel values
    ResizeDepth(fix_depth=140),  # Resize depth to 140
    RearrangeAxes()  # Convert (D, H, W) → (H, W, D)
])

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
train_loader = loadDataset(cfg, "train", medical_transform)
val_loader = loadDataset(cfg, "val", medical_transform)
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
    