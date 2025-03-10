
from transformers import BertTokenizer, BertModel
from ct_clip import CTCLIP, TextTransformer
from CTCLIPTrainer import CTClipTrainer



# pretrain = 'microsoft/BiomedVLP-CXR-BERT-specialized'


# text_encoder = BertModel.from_pretrained(pretrain)



pretrain = 'vinai/phobert-base'
tokenizer = BertTokenizer.from_pretrained(pretrain, do_lower_case=True)
text_encoder = BertModel.from_pretrained(pretrain)

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
#dim_image = 131072,


clip = CTCLIP(
    image_encoder = image_encoder,
    text_encoder = text_encoder,
    dim_text = 768,
    dim_image = 294912,
    dim_latent = 512,
    extra_latent_projection = False,         # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
    use_mlm=False,
    downsample_image_embeds = False,
    use_all_token_embeds = False
)

# save ckpt of CT-CLIP
import torch
# torch.save(clip.state_dict(), "/home/user01/aiotlab/htien/pet-clip/scripts/models/CT-CLIP-Related/test.pt")
# load ckpt of CT-CLIP 
clip.load_state_dict(torch.load("/home/user01/aiotlab/htien/pet-clip/scripts/ct_clip_1/CTClip.31000.pt"))

# clip.text_transformer = text_encoder 

trainer = CTClipTrainer(
    clip,
    root='/home/user01/aiotlab/thaind/DAC001',
    batch_size = 8,
    tokenizer=tokenizer,
    results_folder="ct_clip",
    num_train_steps = 100001,
    num_workers = 2,
)

trainer.steps = torch.Tensor([31000])

trainer.train()
