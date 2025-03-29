import sys
sys.path.append("./cosmos_predict1/")
import pandas as pd 
import numpy as np 
import torch 
from torch import nn
from cosmos_predict1.tokenizer import CausalContinuousVideoTokenizer, CausalContinuousFactorizedVideoTokenizerConfig
from tqdm import tqdm 
from loss import FlowLoss
from dataset_loader import dataloader
import torch.optim as optim

class Trainer(torch.nn.Module):
    def __init__(self, 
                #  args, 
                 model, 
                 train_loader, 
                 test_loader,
                #  optimizer,
                 loss_rec,
                 loss_of,
                 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                 ):
        super().__init__()
        # self.args = args
        self.model = model 
        self.train_loader = train_loader
        self.test_loader = test_loader
        # self.optimizer = optimizer
        self.optimizer = optim.Adam(self.model.parameters())
        self.loss_rec = loss_rec
        self.loss_of = loss_of
        self.device = device
        self.model = self.model.to(device)
    
    def train_stage_1(self):
        self.model.train()
        total_loss = 0  # Tổng loss
        num_batches = 0  # Số batch
        for images in tqdm(self.train_loader):
            self.optimizer.zero_grad()
            images = images.to(device)
            output = self.model(images)
            print(images.shape)
            print(output.shape)
            loss_rec = self.loss_rec(images, output)
            loss_rec.backward()
            self.optimizer.step()
            total_loss += loss_rec.item()
            num_batches += 1  
        
        avg_loss = total_loss / num_batches
        return avg_loss

    def train_stage_2(self):
        self.model.train()
        total_loss = 0  # Tổng loss
        num_batches = 0  # Số batch
        for images in tqdm(self.train_loader):
            self.optimizer.zero_grad()
            images = images.to(device)
            output = self.model(images)

            loss_of = self.loss_of(images, output)
            loss_of.backward()
            self.optimizer.step()
            total_loss += loss_of.item()
            num_batches += 1  
        
        avg_loss = total_loss / num_batches
        return avg_loss

    def train(self):
        log_file = open("training_log.txt", "w")  
        EPOCHS = 20
        for epoch in range(EPOCHS):
            loss_rec = self.train_stage_1()
            loss_of = self.train_stage_2()
            log_file.write(f"Epoch {epoch+1}/{EPOCHS} - Loss_rec: {loss_rec:.4f}, Loss_of: {loss_of:.4f}\n")
            print(f"Epoch {epoch+1}/{EPOCHS} - Loss_rec: {loss_rec:.4f}, Loss_of: {loss_of:.4f}")
        log_file.close()

if __name__ == "__main__":
    # print(CausalContinuousFactorizedVideoTokenizerConfig)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = dataloader("train.txt")
    val_loader = dataloader("val.txt")
    loss_of = FlowLoss()
    loss_rec = nn.L1Loss(reduction='mean')
    model = torch.jit.load("/home/ducntm/FmMed/ckpt/Cosmos-Tokenize1-CV8x8x8-720p/autoencoder.jit", map_location = device)
    trainer = Trainer(model, train_loader, val_loader, loss_rec, loss_of, device)
    trainer.train()