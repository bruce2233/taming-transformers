# %%
from taming.modules.vqvae.quantize import VectorQuantizer2

# %%
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import torch
import PIL
from PIL import Image

def preprocess(img, target_image_size=256):
    s = min(img.size)
    
    if s < target_image_size:
        raise ValueError(f'min dim for image {s} < {target_image_size}')
        
    r = target_image_size / s
    s = (round(r * img.size[1]), round(r * img.size[0]))
    img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
    img = TF.center_crop(img, output_size=2 * [target_image_size])
    img = torch.unsqueeze(T.ToTensor()(img), 0)
    return img

# %%
import sys
print(sys.path)

# %%
sys.path.append("./")

# %%
from taming.modules.diffusionmodules.model import Encoder

# %%
model = torch.load("/mnt/g/github/ZSE-SBIR/taming-transformers/logs/vqgan_imagenet_f16_1024/checkpoints/last.ckpt",map_location=torch.device('cuda:0'))

# %%
from omegaconf import OmegaConf
config_path = "/mnt/g/github/ZSE-SBIR/taming-transformers/logs/vqgan_imagenet_f16_1024/configs/model.yaml"
config = OmegaConf.load(config_path)
print(config.model.params.ddconfig)

# %%
enc_params = {'.'.join(key.split('.')[1:]): value for key,value in model['state_dict'].items() if key.startswith('encoder.')}
print(enc_params)

# %%
enc = Encoder(**config.model.params.ddconfig)

enc.load_state_dict(enc_params)

# %%
# import cv2
import einops

# %%
# im = cv2.imread("data/ade20k_segmentations/ADE_val_00000532.png")
# im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)

im = PIL.Image.open("/mnt/g/github/ZSE-SBIR/taming-transformers/data/coco_annotations_100/val2017/000000010092.jpg")
print(im.size)

x = preprocess(im)
# im = einops.rearrange(im.unsqueeze(0), 'b h w c -> b c h w')
print(x.shape)

# %%
z = enc(x)

# %%
quantizer = VectorQuantizer2(config.model.params.n_embed, config.model.params.embed_dim, 0.25)

# %%
# for k,v in model['state_dict'].items():
#     if k.startswith('quantize'):
#         print(k)
quantizer_params =  {'.'.join(k.split('.')[1:]):v for k,v in model['state_dict'].items() if k.startswith('quantize')}
print(quantizer_params)
quantizer.load_state_dict(quantizer_params)

# %%
z_q = quantizer(z)


# %%
print(z_q[0].shape)

# %%
all_configs = OmegaConf.load("/mnt/g/github/ZSE-SBIR/taming-transformers/logs/idea2/configs/2020-11-20T12-54-32-project.yaml")
print(all_configs)
print(all_configs.model.params.transformer_config)
print(all_configs.model.params.first_stage_config)
print(all_configs.model.params.cond_stage_config)

# %%
from taming.models.cond_transformer import Net2NetTransformer

net2net = Net2NetTransformer(all_configs.model.params.transformer_config, all_configs.model.params.first_stage_config,all_configs.model.params.cond_stage_config)

# %%
# print(transformer)
quant_z, z_indices = net2net.encode_to_c(x)
print(quant_z.shape, len(z_indices))

# %%
print(z_indices.shape)
net2net.transformer(z_indices[-2:])

# %%
im_output = net2net.decode_to_img(z_indices, quant_z.shape)
print(im_output.shape)
import torchvision
torchvision.utils.save_image(im_output,"logs/im_decode.jpg")
# %%
net2net.transformer.block_size
# %%
