
from transformers import BertTokenizer, BertModel
from ct_clip import CTCLIP, TextTransformer
from CTCLIPTrainer import CTClipTrainer


pretrain = 'vinai/phobert-base'
tokenizer = BertTokenizer.from_pretrained(pretrain, do_lower_case=True)

text_encoder = BertModel.from_pretrained(pretrain)

print("---------")
print(tokenizer.pad_token_id)
print(tokenizer.mask_token_id)
print("-----------")




from vit_3d import ViT
image_encoder = ViT(
            image_size = 256,          # image size
            frames = 512,               # max number of frames
            image_patch_size = 32,     # image patch size
            frame_patch_size = 4,      # frame patch size
            dim = 768,
            depth = 12,
            heads = 8,
            mlp_dim = 2048,
            channels=1,
            dropout = 0.1,
            emb_dropout = 0.1
        )
#dim_image = 131072,


clip = CTCLIP(
    image_encoder = image_encoder,
    text_encoder = text_encoder,
    dim_text = 768,
    dim_image = 49152,
    dim_latent = 512,
    extra_latent_projection = False,         # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
    use_mlm=False,
    downsample_image_embeds = False,
    use_all_token_embeds = False

)
trainer = CTClipTrainer(
    clip,
    root='/home/user01/aiotlab/thaind/DAC001',
    batch_size = 8,
    tokenizer=tokenizer,
    results_folder="output_folder2",
    num_train_steps = 500001,
    num_workers = 4,
)

trainer.train()
