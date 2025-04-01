import PIL
import torch
import numpy as np 
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from timm.data.transforms_factory import transforms_noaug_train

# convert all constants to torch tensors 
IMAGENET_DEFAULT_MEAN = torch.tensor(IMAGENET_DEFAULT_MEAN)
IMAGENET_DEFAULT_STD = torch.tensor(IMAGENET_DEFAULT_STD)
OPENAI_CLIP_MEAN = torch.tensor(OPENAI_CLIP_MEAN)
OPENAI_CLIP_STD = torch.tensor(OPENAI_CLIP_STD)

def undo_vision_transform(obs : torch.Tensor, mean : torch.Tensor = IMAGENET_DEFAULT_MEAN, std : torch.Tensor = IMAGENET_DEFAULT_STD):
    """
    Undo the vision transform applied to the observations.
    torch tensor has shape H, 3, H, W
    return np.ndarray with shape H, W, 3 at np.uint8
    """
    # undo normalization
    obs = obs.cpu()
    mean, std = torch.tensor(mean), torch.tensor(std)
    obs = obs.permute(0, 2, 3, 1)
    obs = obs * std + mean
    obs = obs.numpy()
    obs = np.clip(obs * 255, 0, 255).astype(np.uint8)
    return obs

def vision_transform(obs : torch.Tensor, mean : torch.Tensor = IMAGENET_DEFAULT_MEAN, std : torch.Tensor = IMAGENET_DEFAULT_STD):
    """
    Apply the vision transform to the observations.
    torch tensor has shape *, 3, H, W dtype=torch.uint8
    return torch tensor with shape *, 3, H, W dtype=torch.float32
    """
    # normalize
    if not torch.is_floating_point(obs) or obs.max() > 1.0: # convert to float
        obs = obs.float() / 255.0
    device = obs.device
    # update conversion to avoid warnings
    mean = mean.clone().detach().to(device)
    std = std.clone().detach().to(device)
    # Reshape mean and std to broadcast for any leading dimensions:
    expand_shape = [1] * (obs.dim() - 3) + [3, 1, 1]
    mean = mean.view(*expand_shape)
    std = std.view(*expand_shape)
    obs = (obs - mean) / std
    return obs

def imagenet_transform(obs : torch.Tensor):
    return vision_transform(obs, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

def clip_transform(obs : torch.Tensor):
    return vision_transform(obs, OPENAI_CLIP_MEAN, OPENAI_CLIP_STD)

def undo_imagenet_transform(obs : torch.Tensor):
    return undo_vision_transform(obs, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

def undo_clip_transform(obs : torch.Tensor):
    return undo_vision_transform(obs, OPENAI_CLIP_MEAN, OPENAI_CLIP_STD)

def crop2square(pil_img : PIL.Image):
    W, H = pil_img.size
    if H == W:
        return pil_img
    elif H > W:
        return pil_img.crop((0, (H - W) // 2, W, W + (H - W) // 2))
    else:
        return pil_img.crop(((W - H) // 2, 0, H + (W - H) // 2, H))
