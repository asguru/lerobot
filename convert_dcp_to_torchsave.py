import os
import torch
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
# Model folder should have the distcp files and .metadata in it
base_path = "/home/scratch.driveix_50t_4/aguru/lerobot_results/pi0_aloha_adapt12/last/pretrained_model/"
dcp_to_torch_save(os.path.join(base_path, "model/"), os.path.join(base_path, "pi0_aloha_12bs.pt"))