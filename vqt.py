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
model = torch.load("logs/2020-09-23T17-56-33_imagenet_vqgan/checkpoints/last.ckpt",map_location=torch.device('cuda:0'))

# %%
from omegaconf import OmegaConf
config_path = "logs/2020-09-23T17-56-33_imagenet_vqgan/configs/2020-09-23T17-56-33-project.yaml"
config = OmegaConf.load(config_path)
print(config.model.params.ddconfig)

# %%
enc_params = {'.'.join(key.split('.')[1:]): value for key,value in model['state_dict'].items() if key.startswith('encoder.')}
print(enc_params)

# %%
enc = Encoder(**config.model.params.ddconfig).to("cuda:0")

enc.load_state_dict(enc_params)

# %%
# import cv2
import einops

# %%
# im = cv2.imread("data/ade20k_segmentations/ADE_val_00000532.png")
# im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)

im = PIL.Image.open("data/coco_annotations_100/val2017/000000010092.jpg")
print(im.size)

x = preprocess(im).to("cuda:0")
# im = einops.rearrange(im.unsqueeze(0), 'b h w c -> b c h w')
print(x.shape,x.device)

# %%
z = enc(x)
print(z.shape,z.device)
# %%
quantizer = VectorQuantizer2(config.model.params.n_embed, config.model.params.embed_dim, 0.25).to("cuda:0")

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
print(z_q[0].shape, z_q.device)

# %%
all_configs = OmegaConf.load("logs/idea2/configs/2020-11-20T12-54-32-project.yaml")
print(all_configs)
print(all_configs.model.params.transformer_config)
print(all_configs.model.params.first_stage_config)
print(all_configs.model.params.cond_stage_config)

# %%
from taming.models.cond_transformer import Net2NetTransformer

net2net = Net2NetTransformer(all_configs.model.params.transformer_config, all_configs.model.params.first_stage_config,all_configs.model.params.cond_stage_config).to("cuda:0")

# %%
# print(transformer)
quant_z, z_indices = net2net.encode_to_c(x)
print(quant_z.shape, quant_z.device, len(z_indices))
z_indices = torch.unsqueeze(z_indices,0)
print(z_indices.shape)


# %%
print(z_indices.shape)
logits,loss = net2net.transformer(z_indices)
print(logits.shape, loss)

# %%
# logits = logits.reshape(1024,16,16)
# print(logits.shape)

# %%
logits = net2net.top_k_logits(logits, 3)
# %%
print(logits.shape)

# %%
probs = torch.nn.functional.softmax(logits, dim=-1)
print(probs.shape,probs)

# %%
ix = torch.multinomial(probs.squeeze(0), num_samples=1)
print(ix.shape,ix)

# %%
im_output = net2net.decode_to_img(ix, quant_z.shape)
print(im_output.shape)
import torchvision
torchvision.utils.save_image(im_output,"logs/im_decode_2.jpg")
# %%
net2net.transformer.block_size
# %%
