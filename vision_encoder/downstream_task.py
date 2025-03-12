from torch.nn import Module
from .ct_clip import CTCLIP
import torch 
from torch import nn
#49152
class ClassificationTask(Module):
    def __init__(self, 
                 model,
                 n_Classes,
                 dim_feature = 49152,
                 ):
        self.model = model
        self.n_classes = n_Classes
        self.dim_feature = dim_feature
        self.fc = nn.Linear(49152, n_Classes)
        for param in self.model.parameters():
            param.require_grad = False
        self.initialize_weights()
        
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.require_grad:
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_features = False):
        features = self.model(x)
        out = self.fc(features)
        if return_features:
            return out, features
        else:
            return out
    
    
    