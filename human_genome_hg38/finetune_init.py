import os
import torch
import torch.nn as nn
from load_model import load_model
from omegaconf import OmegaConf
import model as model_lib

print("Training sedd-small ...")
pretrained_model, pretrained_graph, pretrained_noise = load_model("louaaron/sedd-small")
pretrained_model.eval()

pretrained_state = pretrained_model.state_dict()

print(f"Total num of pramaters in pre-training: {sum(p.numel() for p in pretrained_model.parameters()):,}")

from huggingface_hub import hf_hub_download
import json

config_path = hf_hub_download(repo_id="louaaron/sedd-small", filename="config.json")
with open(config_path) as f:
    cfg_dict = json.load(f)

print(f"\nInitial pre-training vocab_size: {cfg_dict.get('vocab_size', 'Unfounded')}")
print(f"Initial pre-training config keys: {list(cfg_dict.keys())}")

HG38_VOCAB_SIZE = 5

cfg_dict["vocab_size"] = HG38_VOCAB_SIZE

cfg = OmegaConf.create(cfg_dict)
hg38_model = model_lib.SEDD(cfg)
hg38_model.train()

hg38_state = hg38_model.state_dict()

transferred, skipped = [], []

for name in hg38_state:
    if name in pretrained_state:
        pre_shape = pretrained_state[name].shape
        new_shape = hg38_state[name].shape
        if pre_shape == new_shape:
            hg38_state[name] = pretrained_state[name].clone()
            transferred.append(name)
        else:
            skipped.append((name, pre_shape, new_shape))
    else:
        skipped.append((name, "NOT exist", hg38_state[name].shape))

hg38_model.load_state_dict(hg38_state)

print(f"\n transformed: {len(transferred)} layer")
print(f" skip(unmatch): {len(skipped)} layer")
print("\nskipped layers：")
for item in skipped:
    print(f"  {item[0]}  Pre-train={item[1]}  New model={item[2]}")

# save checkpoint
os.makedirs("checkpoints_hg38", exist_ok=True)
save_path = "checkpoints_hg38/hg38_init.pth"
torch.save({
    "model": hg38_model.state_dict(),
    "cfg": cfg_dict,
}, save_path)
print(f"\n saved: {save_path}")
print("next train：python run_train.py --config-name config_hg38")