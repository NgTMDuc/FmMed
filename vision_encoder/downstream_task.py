from torch.nn import Module
from .ct_clip import CTCLIP
import torch 
from torch import nn
#49152
import torch.nn.functional as F
class ClassificationTask(Module):
    def __init__(self, 
                 model,
                 n_Classes,
                 dim_feature = 294912,
                 ):
        super().__init__()
        self.model = model
        self.n_classes = n_Classes
        self.dim_feature = dim_feature
        self.fc = nn.Linear(dim_feature, n_Classes, bias=True)
        for param in self.model.parameters():
            param.requires_grad = False 
        
        for param in self.fc.parameters():
            param.requires_grad = True

        self.initialize_weights() 
            
    def initialize_weights(self):
        """ Initialize only the fc layer """
        nn.init.kaiming_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0)
            
    def forward(self, x, return_features = False):
        features = self.model(x, return_encoded_tokens = True)
        # print(features.shape)
        features = torch.mean(features, dim = 1)
        features = features.view(features.size(0), -1)
        # print(features.shape)
        out = self.fc(features)
        out = F.softmax(out, dim = 1)
        if return_features:
            return out, features
        else:
            return out
    
    
    