import os
import torch
import numpy as np
import tyro
from pathlib import Path
import yaml 
from typing import Dict, Any, Union
import json
from scipy.spatial.transform import Rotation
import PIL
from PIL import Image
import torchvision.transforms.functional as transforms_f
import clip

from otter.util.args import ExperimentConfig
from otter.policy.otter import OTTER, create_text_mask
from otter.util import misc
from otter.dataset.utils import convert_abs_action
from .vision_tf import clip_transform, crop2square

class OtterInference():
    def __init__(
        self, 
        model_ckpt_folder : str, 
        ckpt_id : int, 
        device : str = 'cuda',
    ) -> None:
        # parse the paths 
        train_yaml_path = os.path.join(model_ckpt_folder, 'run.yaml')
        model_ckpt_name = os.path.join(model_ckpt_folder, f'checkpoint_{ckpt_id}.pt')
        
        # load config
        self.args : ExperimentConfig = yaml.load(Path(train_yaml_path).read_text(), Loader=yaml.Loader)
        self.device = device

        # construct model
        self.model = OTTER(
            model_config=self.args.model_cfg,
            shared_config=self.args.shared_cfg,
        )
        print(f"Loading model from checkpoint: {model_ckpt_name}")
        self.model = misc.load_state_dict_flexible(self.model, torch.load(model_ckpt_name, map_location='cpu'))
        
        # freeze all model parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Move model to device
        self.model = self.model.to(device)

        # model to eval mode
        self.model.eval()

        # current proprioceptive state, updated in _proprocess_proprio, shape: (self.model.proprio_input_dim)
        self.current_proprio = None

        # print the statistics of the model before eval
        print(self.model)
    
    def _proprocess_proprio(self, proprio : np.ndarray, gripper : np.ndarray) -> torch.Tensor:
        """
        Preprocess proprioception data.
        Args: 
            proprio : np.ndarray of shape (6): [x, y, z, roll, pitch, yaw]
            gripper : np.ndarray of shape (1): [gripper]
        
        Returns:
            torch.Tensor of shape (self.model.proprio_input_dim)
        """
        trans = proprio[:3]
        rot = proprio[3:]
        # droid and real (franka) use "XYZ" convention, robomimic uses "xyz"
        rot_mat = Rotation.from_euler("XYZ", rot, degrees=False).as_matrix()
        # select out the first two rows of the rotation matrix
        rot = rot_mat[:2, :].reshape((6,))
        ret = np.concatenate([trans, rot], axis=-1)
        
        gripper = np.clip(gripper, 0, 1)
        proprio_data = np.concatenate((ret, gripper), axis=-1)

        # update the current proprio state 
        self.current_proprio = proprio_data.copy()

        return torch.tensor(proprio_data).float().to(self.device)

    def _preprocess_images(self, images : Dict[str, PIL.Image.Image]) -> torch.Tensor:
        """
        Preprocess images.
        Args:
            images : Dict[str, PIL.Image.Image] with keys: ['image_primary', 'image_wrist']
        
        Returns:
            a tensor of shape (2, 3, self.args.shared_cfg.image_size, self.args.shared_cfg.image_size)
        """
        # first make sure the keys are matching 
        assert set(images.keys()) == set(self.model.camera_keys)

        out_images = []
        for key in self.model.camera_keys:
            image = crop2square(images[key])
            image = image.resize(
                (self.args.shared_cfg.image_size, self.args.shared_cfg.image_size), 
                # Image.Resampling.LANCZOS
                Image.Resampling.BICUBIC,
            )
            image = transforms_f.to_tensor(image)
            image = clip_transform(image)
            out_images.append(image)

        out_images = torch.stack(out_images, dim=0).float().to(self.device)
        return out_images
            
    def _decode_actions(
        self, 
        actions : Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        """
        Decode actions.
        Args:
            delta actions : np.ndarray or tensor of shape (T, D)
        
        Returns:
            absolute action in np.ndarray of shape (T, D)
        """
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        
        actions = actions[None, ...] # (1, T, D)
        proprio = self.current_proprio[None, None, ...] # (1, 1, D)

        # convert to absolute actions
        actions = convert_abs_action(actions, proprio) # (1, T, D)
        return actions[0]

    def reset(self):
        """
        Reset the current proprio state.
        """
        self.current_proprio = None
        self.model.reset_cache()

    def __call__(
        self, 
        images : Dict[str, PIL.Image.Image],
        text : str,
        proprio : np.ndarray,
        gripper : np.ndarray,
    ) :
        """
        Perform inference.
        Args:
            images : Dict[str, PIL.Image.Image] with keys: ['image_primary', 'image_wrist', 'proprio', 'gripper']
            text : str : text instruction
            proprio : np.ndarray of shape (6): [x, y, z, roll, pitch, yaw]
            gripper : np.ndarray of shape (1): [gripper]
        
        Returns:
            action of shape (self.model.action_horizon, self.model.action_dim)
        """
        # preprocess proprioception data
        proprio = self._proprocess_proprio(proprio, gripper) # shape: (self.model.proprio_input_dim)

        # preprocess text 
        text = clip.tokenize([text]).squeeze().to(self.device) # shape: (L)

        # create text mask 
        if self.model.pool_true_text:
            text_mask = create_text_mask(text, self.model.sot_token, self.model.eot_token, self.model.first_k_tokens).to(self.device)
        else:
            text_mask = None

        # preprocess images
        images = self._preprocess_images(images) # shape: (2, 3, H, W)

        # perform inference
        action = self.model.forward_inference(
            images=images, 
            text=text,
            proprio=proprio,
            text_mask=text_mask,
        )

        # decode actions 
        action = self._decode_actions(action) # (T, D)
        return action