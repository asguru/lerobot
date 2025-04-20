# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

from lerobot.common.optim.optimizers import AdamWConfig
from lerobot.common.optim.schedulers import (
    CosineDecayWithWarmupSchedulerConfig,
)
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature


@PreTrainedConfig.register_subclass("otter")
@dataclass
class OtterConfig(PreTrainedConfig):
    # Input / output structure.
    # CLIP model architecture to use for vision encoding
    clip_model: str = "ViT-L/14"

    # Action chunk size
    chunk_size: int = 12

    # Image size for computing num image patches
    image_size: int = 224

    # num_readouts for attention pooling
    num_readouts: int = 4
    
    # Input dimension for proprioception data
    proprio_input_dim: int = 14 # Changed to 14 from 10 to accommodate dataset
    
    # Hidden dimension for proprioception processing
    proprio_hidden_dim: int = 256
    
    # Output dimension for proprioception features
    proprio_output_dim: int = 64
    
    # Output dimension for text pooling layer
    text_pooling_output_dim: int = 128

    # number of tokens used for CLIP 
    first_k_tokens : int = 15
    
    # Output dimension for vision pooling layer (combined value for all cameras)
    vision_pooling_output_dim: int = 512
    
    # Number of attention heads in pooling layers
    pooling_heads: int = 8
    
    # Number of pooling layers
    pooling_layers: int = 2
    
    # Dimension of action space
    action_dim: int = 14

    # max position embeddings
    max_position_embeddings : int = 32

    # Number of transformer layers in the model
    transformer_layers: int = 8
    
    # Number of attention heads in transformer layers
    transformer_heads: int = 8
    
    # Hidden dimension size in transformer layers
    transformer_dim: int = 768

    # Transformer expansion factor more consistent with paper
    transformer_expansion_factor: int = 1

    # attention dropout probability
    attention_probs_dropout_prob : int = 0.0

    # dropout probability
    hidden_dropout_prob : int = 0.0

    # only true text token are used for attention pooling 
    pool_true_text : bool = False

    # Training presets
    optimizer_lr: float = 2.5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    # TODO: Add EMA

    def __post_init__(self):
        super().__post_init__()


    def validate_features(self) -> None:
        # TODO: implement value error
        # if not self.image_features and not self.env_state_feature:
        #     raise ValueError("You must provide at least one image or the environment state among the inputs.")

        for i in range(self.empty_cameras):
            key = f"observation.images.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 480, 640),
            )
            self.input_features[key] = empty_camera

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
