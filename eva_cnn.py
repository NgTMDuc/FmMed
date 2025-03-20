from .cnn_based import generate_model
from util import AverageMeter, count_params, init_log, collect_params, loadDataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
from tqdm import tqdm
import argparse
import yaml
import logging
import torch.nn as nn
from torch import optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
parser.add_argument('--config', type=str, required=True)
args = parser.parse_args()

cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
train_loader = loadDataset(cfg, "train")
val_loader = loadDataset(cfg, "val")
model = generate_model(18, {"n_classes" : 3})
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
    