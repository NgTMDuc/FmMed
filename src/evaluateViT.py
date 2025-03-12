from ctvit import CTViT

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
print(image_encoder.vq._codebook)
import torch
image_encoder.load_state_dict(torch.load('../ckpt/CTVit.39000.pt'))