from vision_encoder import load_task, CTViT

import torch 
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import os
from util import AverageMeter, count_params, init_log, collect_params, loadDataset
import yaml
import argparse
import logging

parser = argparse.ArgumentParser(description='Foundation Model for Vietnamese medical dataset')
parser.add_argument('--config', type=str, required=True)
# parser.add_argument('--save-path', type=str, required=True)

def accuracy(model, val_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for img, label in tqdm(val_loader):
            img = img.to(device)
            label = label.to(device)
            
            output = model(img)
            
            preds = torch.argmax(output, dim = 1)
            
            all_preds.extend(preds.cpu().numpy()) 
            all_labels.extend(label.cpu().numpy())
            
    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')  # 'macro' for balanced class impact
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    
    return acc, f1, precision, recall

def main():
    args = parser.parse_args()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    logger = init_log('global', logging.INFO)
    logger.propagate = 0
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embed_ckpt = cfg.embed_ckpt
    image_encoder = CTViT(
        dim = 512,
        codebook_size = 8192,
        image_size = 480,
        patch_size = 20,
        temporal_patch_size = 10,
        spatial_depth = 4,
        temporal_depth = 4,
        dim_head = 32,
        heads = 8
    ) 
    image_encoder.load_state_dict(torch.load(embed_ckpt))
    model = load_task(cfg, image_encoder)
    params, param_names = collect_params(model)
    model.train()
    
    EPOCHS = cfg.epochs
    save_folder = cfg.save_folder
    previous_best = 0.0
    epoch = -1
    criteria = nn.CrossEntropyLoss().to(device)
    
    optimizer = optim.Adam(
        params, lr = cfg.lr
    )
    train_loader = loadDataset(cfg, "train")
    val_loader = loadDataset(cfg, "valid")
    
    if os.path.exists(os.path.join(save_folder, "lastest.pth")):
        last_checkpoint = torch.load(os.path.join(save_folder, "lastest.pth"))
        model.load_state_dict(last_checkpoint["model"])
        optimizer.load_state_dict(last_checkpoint['optimizer'])
        epoch = last_checkpoint['epoch']
        previous_best = last_checkpoint['previous_best']
        
        logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
    
    for epoch in range(epoch + 1, EPOCHS):
        total_loss = AverageMeter()
        
        logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f}'.format(epoch, optimizer.param_groups[0]['lr'], previous_best))
        
        for img, label in tqdm(train_loader):
            img = img.to(device)
            label = label.to(device)
            out = model(img)
            
            loss = criteria(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss.update(loss.item())
        
        logger.info('===========> Epoch: {:}, Loss: {:.2f}'.format(epoch, total_loss.avg))
        acc, f1, pre, recal = accuracy(model, val_loader, cfg, device)
        is_best = acc > previous_best
        
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "previous_best": previous_best
        }
        
        torch.save(checkpoint, os.path.join(save_folder, "lastest.pth"))
        if is_best:
            torch.save(checkpoint, os.path.join(save_folder, 'best.pth'))


if __name__ == "__main__":
    main()