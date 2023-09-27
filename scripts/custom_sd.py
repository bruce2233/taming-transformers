# 要求被替换的model的key在替换的submodel里都有, 且满足sd[k] = cond_stage_model_sd[".".join(k.split(".")[1:])]的形式
# %%
import torch

model = torch.load("../logs/idea2/checkpoints/last_origin.ckpt")
sd = model['state_dict']
cond_stage_model_sd = torch.load("../logs/2020-09-23T17-56-33_imagenet_vqgan/checkpoints/last.ckpt")['state_dict']
print(sd["cond_stage_model.encoder.conv_in.weight"].shape)
print(model["state_dict"]["cond_stage_model.encoder.conv_in.weight"].shape)

# %%
for k,v in sd.items():
     if k.startswith("cond_stage_model"):
        sd[k] = cond_stage_model_sd[".".join(k.split(".")[1:])]
        
print(sd["cond_stage_model.encoder.conv_in.weight"].shape)
print(model["state_dict"]["cond_stage_model.encoder.conv_in.weight"].shape)

# %%
torch.save(model, "../logs/idea2/checkpoints/last_output.ckpt")
# %%
print(torch.load("../logs/idea2/checkpoints/last_output.ckpt")['state_dict']["cond_stage_model.encoder.conv_in.weight"].shape)
# %%
