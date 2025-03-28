import sys
sys.path.append("./cosmos_predict1/")
import pandas as pd 
import numpy as np 
import torch 
from torch import nn
from cosmos_predict1.tokenizer import CausalContinuousVideoTokenizer, CausalContinuousFactorizedVideoTokenizerConfig
from tqdm import tqdm 

# def load_model(
#     model_name: str = "Cosmos-Tokenize1-CV8x8x8-720p",
#     temporal_window: int = 49
# ):
#     encoder_ckpt = f"ckpt/{model_name}/encoder.jit"
#     decoder_ckpt = f"ckpt/{model_name}/decoder.jit"
    
#     tokenizer = TokenizerModel(
#         checkpoint_enc=encoder_ckpt,
#         checkpoint_dec=decoder_ckpt,
#         device="cuda",
#         dtype="bfloat16",
#     )
    
#     return tokenizer
# def load_model():
    
def stage1_train(model, train_loader):
    model.train()
    for images in tqdm(train_loader):
        images = images.cuda()
        output_images = model(images)
    

if __name__ == "__main__":
    print(CausalContinuousFactorizedVideoTokenizerConfig)