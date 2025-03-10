
from ctvit import CTViT

import torch 

input = torch.randn(1, 1, 140, 480, 480)

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

image_encoder = image_encoder.cpu()

output = image_encoder(input, return_encoded_tokens=True)

print(output.shape) # torch.Size([1, 512])