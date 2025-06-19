import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from typing import Optional, Tuple, List, Union
from lerobot.common.policies.otter.configuration_otter import OtterConfig
from .transformer import AttentionPooling, CausalTransformer
from .models import TextAwareVisualExtraction, ProprioceptionEncoder, ActionHead
from .vision_tf import clip_transform
from clip.simple_tokenizer import SimpleTokenizer
from lerobot.common.constants import OBS_ROBOT, ACTION
from lerobot.common.policies.pretrained import PreTrainedPolicy
from collections import deque

# Create mask for SOS, EOS, and padding tokens
def create_text_mask(text: torch.Tensor, sot_token: int, eot_token: int, first_k_tokens: int) -> torch.Tensor:
    """
    text: batch of text tokens (B, 77)
    sot_token: start of text token
    eot_token: end of text token
    first_k_tokens: number of tokens to consider

    For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
    the purpose of attention. For a float mask, it will be directly added to the corresponding ``key`` value.

    mask will be true at start of text, end of text, and padding tokens
    mask will be false at actual tokens
    """
    # make sure text is either dim 1 or 2 
    text_dim = len(text.shape)
    assert text_dim in (1, 2), "text must be either 1D or 2D"
    text_mask = torch.zeros_like(text, dtype=torch.bool)
    text_mask[text == sot_token] = True  # Start token
    text_mask[text == eot_token] = True  # End token
    text_mask[text == 0] = True  # Padding token
    # Only consider first k tokens
    if text_dim == 1:
        text_mask = text_mask[:first_k_tokens]    
    else:
        text_mask = text_mask[:, :first_k_tokens]  
    return text_mask

class OtterPolicy(PreTrainedPolicy):
    """Full Otter model with temporal sequence handling"""
    
    config_class = OtterConfig
    name = "otter"

    def __init__(
        self,
        config: OtterConfig,
        dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    ):
        super().__init__(config)
        # Extract model parameters from config
        self.clip_model_name = config.clip_model
        self.proprio_input_dim = config.proprio_input_dim 
        proprio_hidden_dim = config.proprio_hidden_dim
        proprio_output_dim = config.proprio_output_dim
        text_pooling_output_dim = config.text_pooling_output_dim
        vision_pooling_output_dim = config.vision_pooling_output_dim
        pooling_heads = config.pooling_heads
        pooling_layers = config.pooling_layers
        self.action_dim = config.action_dim
        self.first_k_tokens = config.first_k_tokens
        self.num_readouts = config.num_readouts
        self.pool_true_text = config.pool_true_text
        
        # extract shared config parameters
        self.action_horizon = config.chunk_size
        
        # Get camera keys from dataset_stats
        if dataset_stats is None:
            raise ValueError("dataset_stats must be provided to extract camera keys")
        # Find all keys that start with "observation.image"
        self.camera_keys = sorted([k for k in dataset_stats.keys() if k.startswith("observation.image")])
        if not self.camera_keys:
            raise ValueError("No camera keys found in dataset_stats")
        num_cameras = len(self.camera_keys)
        
        # FIXME: should this be a separate param?
        self.seq_length = config.chunk_size
        self.image_size = config.image_size
        
        # Load CLIP model
        self.clip_model, self.image_preprocess = clip.load(self.clip_model_name)
        self.tokenizer = SimpleTokenizer()
        self.sot_token : int = self.tokenizer.encoder["<|startoftext|>"]
        self.eot_token : int = self.tokenizer.encoder["<|endoftext|>"]

        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False
            param.data = param.data.float() # default uses half
            
        # Text and vision feature dimensions from CLIP
        text_dim = self.clip_model.text_projection.shape[1]
        vision_dim = self.clip_model.visual.output_dim
        
        # Components
        self.text_pooling = AttentionPooling(
            text_dim, 
            text_pooling_output_dim,
            pooling_heads,
            pooling_layers, 
            num_readouts=self.num_readouts,
        )
        
        self.visual_patch_size = self.clip_model.visual.conv1.kernel_size[0]
        self.num_img_patches = (self.image_size // self.visual_patch_size) ** 2

        self.visual_extraction = nn.ModuleList([
            TextAwareVisualExtraction(num_img_patches=self.num_img_patches, vision_dim=vision_dim) for _ in self.camera_keys
        ])
        
        # assert that the vision_pooling output dim is divisible by the number of cameras
        assert vision_pooling_output_dim % num_cameras == 0, "Vision pooling output dim must be divisible by number of cameras"
        # Create one vision pooling module per camera, using camera keys
        self.vision_poolings = nn.ModuleList([
            AttentionPooling(
            vision_dim,
            vision_pooling_output_dim // num_cameras,
            pooling_heads, 
            pooling_layers, 
            num_readouts=self.num_readouts
            ) for _ in self.camera_keys
        ])
        
        self.proprio_output_dim = proprio_output_dim
        self.proprio_encoder = ProprioceptionEncoder(
            self.proprio_input_dim,
            proprio_hidden_dim,
            proprio_output_dim
        )
        
        self.f_t_dim = text_pooling_output_dim + vision_pooling_output_dim + proprio_output_dim
        self.input_projection = nn.Linear(self.f_t_dim, config.transformer_dim)
        
        self.policy = CausalTransformer(config)
        
        self.action_head = ActionHead(config.transformer_dim, self.action_dim, self.action_horizon)

        # Store activations dict at instance level
        self.activation = {}
        
        def get_activation(name):
            def hook(model, input, output):
                self.activation[name] = output
            return hook
        
        # Register hooks once
        self.hooks = [
            self.clip_model.visual.transformer.resblocks[-1].attn.register_forward_hook(
                get_activation('image_patches')
            ),
            self.clip_model.transformer.register_forward_hook(
                get_activation('text_features')
            )
        ]

        # Add cache for transformer inputs
        self.cached_transformer_input = None
        self.cache_size = 0
        
        # Initialize action queue
        self._action_queue = deque([], maxlen=self.action_horizon)
    
    def reset(self):
        """Reset the model state when environment is reset.
        
        This includes:
        1. Clearing the transformer input cache
        2. Clearing the action queue
        """
        self.reset_cache()
        self._action_queue.clear()

    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Select a single action given environment observations.
        
        This method manages an action queue and returns one action at a time.
        When the queue is empty, it generates a new sequence of actions.
        
        Args:
            batch: Dictionary containing:
                - observation.image.*: Image tensors from multiple cameras
                - task: Text input
                - OBS_ROBOT: Proprioceptive state
                
        Returns:
            Single action tensor of shape (action_dim,)
        """
        # If action queue is empty, generate new actions
        if len(self._action_queue) == 0:
            # Prepare inputs
            images = self.prepare_images(batch)
            text = self.prepare_language(batch)
            proprio = batch[OBS_ROBOT]
            
            # Create text mask if needed
            if self.pool_true_text:
                text_mask = create_text_mask(text, self.sot_token, self.eot_token, self.first_k_tokens)
                text_mask = text_mask.to(images.device)
            else:
                text_mask = None
                
            # Generate actions
            actions = self.forward_actions(images, text, proprio, text_mask)
            
            # Add actions to queue
            # actions shape: (B, T, action_horizon, action_dim)
            # We want to queue them as (action_horizon, B, action_dim)
            actions = actions.transpose(0, 2)  # (action_horizon, B, T, action_dim)
            for action in actions:
                self._action_queue.append(action[0, 0])  # Take first batch, first timestep
                
        # Return next action from queue
        return self._action_queue.popleft()

    def reset_cache(self):
        """Reset the transformer input cache"""
        self.cached_transformer_input = None
        self.cache_size = 0

    def get_optim_params(self) -> dict:
        """Get parameters for optimization.
        
        Returns:
            All model parameters that can be optimized.
        """
        return self.parameters()

    def update_cache(self, transformer_input: torch.Tensor):
        """
        Update the transformer input cache
        Args:
            transformer_input (torch.Tensor): Input to the transformer (B, T, transformer_dim
        """
        if self.cached_transformer_input is None:
            self.cached_transformer_input = transformer_input # (B, T, transformer_dim)
        else:
            self.cached_transformer_input = torch.cat([
                self.cached_transformer_input,
                transformer_input
            ], dim=1)[:, -self.seq_length:]
            
        self.cache_size = min(
            self.cache_size + transformer_input.shape[1],
            self.seq_length
        )

    def extract_clip_features(
        self, 
        images: torch.Tensor,  # (B, T, num_cameras, C, H, W)
        text: torch.Tensor,    # (B, max_text_len)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, num_cameras = images.shape[:3]
        
        # Reshape images to process all views at once
        images = images.view(B * T * num_cameras, *images.shape[3:])  # (B*T*num_cameras, C, H, W)
        
        # Get features
        with torch.no_grad():
            _ = self.clip_model.encode_text(text)
            _ = self.clip_model.encode_image(images)
        
        # Process text features. self.activation is a hook!!!
        text_features = self.activation['text_features'].permute(1, 0, 2)[:, :self.first_k_tokens]
        text_features = self.clip_model.ln_final(text_features).type(self.clip_model.dtype) @ self.clip_model.text_projection
        text_features = text_features / text_features.norm(dim=-1, keepdim=True) # (B, first_k_tokens, text_dim)
        
        # repeat text features T times 
        # should be equivalent to 
        # text_features = text_features.unsqueeze(1).repeat(1, 3, 1, 1).view(B*T, self.first_k_tokens, text_features.shape[-1])
        text_features = text_features.repeat_interleave(T, dim=0) # (B*T, first_k_tokens, text_dim)
        
        # Process patch features. self.activation is a hook!!!
        patch_features = self.activation['image_patches'][0]
        patch_features = patch_features.permute(1, 0, 2)
        
        # get rid of the cls token 
        patch_features = patch_features[:, 1:]

        patch_features = self.clip_model.visual.ln_post(patch_features)
        
        if self.clip_model.visual.proj is not None:
            patch_features = patch_features @ self.clip_model.visual.proj
            
        patch_features = patch_features / patch_features.norm(dim=-1, keepdim=True)
        
        # Reshape patch features back to separate cameras
        patch_features = patch_features.view(B * T, num_cameras, *patch_features.shape[1:])
        
        # Split for each camera
        patch_features_per_camera = [patch_features[:, i] for i in range(num_cameras)]
        
        return patch_features_per_camera, text_features
    
    def forward_encoder(
        self, 
        images: torch.Tensor,      # (B, T, num_cameras, C, H, W)
        text: torch.Tensor,        # (B, max_text_len)
        proprio: torch.Tensor,     # (B, T, proprio_dim)
        text_mask: Optional[torch.Tensor] = None,   # (B, first_k_tokens)
    ):
        B, T = images.shape[:2]
        # patch_features_per_camera: List of num_cameras tensors of shape (B * T, num_patches, vision_dim)
        # text_features: Tensor of shape (B * T, self.first_k_tokens, text_dim)
        patch_features_per_camera, text_features = self.extract_clip_features(images, text)

        # process text mask to match T
        if self.pool_true_text:
            text_mask = text_mask.repeat_interleave(T, dim=0)
        
        # Process each camera's features
        vision_tokens = []
        for camera_idx, camera_features in enumerate(patch_features_per_camera):
            # Get text-aware features for this camera
            text_aware_features = self.visual_extraction[camera_idx](camera_features, text_features)
            
            # Pool features for this camera
            vision_token = self.vision_poolings[camera_idx](text_aware_features, text_mask)
            vision_tokens.append(vision_token)
        
        # Combine vision tokens from all cameras
        vision_token = torch.cat(vision_tokens, dim=-1)  # (B*T, vision_dim)
        
        # Pool text features (constant across time steps)
        text_token = self.text_pooling(text_features, text_mask)  # (B*T, text_dim)
        
        # Process proprioception
        proprio = proprio.view(B * T, -1)  # (B*T, proprio_dim)
        proprio_token = self.proprio_encoder(proprio)  # (B*T, proprio_dim)
        
        # Combine all features
        combined = torch.cat([text_token, vision_token, proprio_token], dim=-1)
        transformer_input = self.input_projection(combined)
        
        # Reshape back to sequence for transformer
        transformer_input = transformer_input.view(B, T, -1)  # (B, T, transformer_dim)

        return transformer_input

    def forward_actions(
        self,
        images: torch.Tensor,                       # (B, T, num_cameras, C, H, W)
        text: torch.Tensor,                         # (B, max_text_len)
        proprio: torch.Tensor,                      # (B, T, proprio_dim)
        text_mask: Optional[torch.Tensor] = None,   # (B, first_k_tokens)
    ) -> torch.Tensor:
        # get necessary shapes for forward pass
        B, T = images.shape[:2]
        
        # Pass through encoders
        transformer_input = self.forward_encoder(images, text, proprio, text_mask)
        
        # Pass through transformer policy
        policy_output = self.policy(transformer_input)
        
        # Predict actions for current timestep
        actions = self.action_head(policy_output) # (B, T, action_dim * action_horizon)

        # view it as (B, T, action_horizon, action_dim)
        actions = actions.view(B, T, self.action_horizon, self.action_dim)

        return actions

    @torch.no_grad()
    def forward_inference(
        self, 
        images: torch.Tensor,                       # (num_cameras, C, H, W)
        text: torch.Tensor,                         # (max_text_len)
        proprio: torch.Tensor,                      # (proprio_dim)
        text_mask: Optional[torch.Tensor] = None,   # (first_k_tokens)
    ) -> torch.Tensor:
        """
        Perform inference with the model
        Args:
            images (torch.Tensor): Images from all cameras (num_cameras, C, H, W)
            text (torch.Tensor): Text input (max_text_len) tokenization is taken care of outside
            proprio (torch.Tensor): Proprioceptive input (proprio_dim)
        Returns:
            torch.Tensor: Predicted actions (action_horizon, action_dim)
        """
    
        # Reshape images to process all views at once
        images = images.unsqueeze(0).unsqueeze(0) # (1, 1, num_cameras, C, H, W)
        text = text.unsqueeze(0) # (1, max_text_len)
        proprio = proprio.unsqueeze(0).unsqueeze(0) # (1, 1, proprio_dim)
        if self.pool_true_text:
            text_mask = text_mask.unsqueeze(0) # (1, first_k_tokens)
        
        # Pass through encoders
        transformer_input = self.forward_encoder(images, text, proprio, text_mask)

        # update cache 
        self.update_cache(transformer_input)
        
        # Pass through transformer policy
        policy_output = self.policy(self.cached_transformer_input) # B, T', transformer_dim
        
        # we only process the output of the last timestep
        policy_output = policy_output[:, self.cache_size - 1] # (B, transformer_dim)

        # Predict actions for current timestep
        actions = self.action_head(policy_output).squeeze() # (action_dim * action_horizon)

        # view it as (action_horizon, action_dim)
        actions = actions.view(self.action_horizon, self.action_dim)

        return actions

    def prepare_language(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Tokenize the text input using CLIP's tokenizer.
        
        Args:
            batch: Dictionary containing task key with text input
            
        Returns:
            Tokenized text tensor ready for CLIP encoding
        """
        device = batch[OBS_ROBOT].device
        tasks = batch["task"]

        # Tokenize using CLIP's tokenizer
        if isinstance(tasks, list):
            text = clip.tokenize(tasks).to(device)
        else:
            text = tasks.to(device)

        return text

    def resize_with_pad(self, img: torch.Tensor, width: int, height: int, pad_value: float = 0.0) -> torch.Tensor:
        """Resize image maintaining aspect ratio and pad to target size.
        
        Args:
            img: Image tensor of shape (B, C, H, W)
            width: Target width
            height: Target height
            pad_value: Value to use for padding
            
        Returns:
            Resized and padded image tensor
        """
        if img.ndim != 4:
            raise ValueError(f"Expected 4D (B,C,H,W) tensor, but got shape {img.shape}")

        cur_height, cur_width = img.shape[2:]

        ratio = max(cur_width / width, cur_height / height)
        resized_height = int(cur_height / ratio)
        resized_width = int(cur_width / ratio)
        resized_img = F.interpolate(
            img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
        )

        pad_height = max(0, int(height - resized_height))
        pad_width = max(0, int(width - resized_width))

        # pad on left and top of image
        padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
        return padded_img

    def prepare_images(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Prepare images for CLIP encoding.
        
        Args:
            batch: Dictionary containing image tensors of shape (B, T, C, H, W)
            
        Returns:
            Preprocessed images ready for CLIP encoding
        """
        # Extract and stack images from all cameras
        image_tensors = []
        print("camera keys", self.camera_keys)
        for key in self.camera_keys:
            if key in batch:
                img = batch[key]  # (B, T, C, H, W)
                print("img for key", key, img.shape)
                # Handle channel-last format if needed
                if img.shape[-1] == 3:  # (*, H, W, 3)
                    img = img.permute(0, 1, 4, 2, 3)  # (B, T, 3, H, W)
                image_tensors.append(img)
        
        if not image_tensors:
            raise ValueError(f"No camera images found in batch. Expected keys: {self.camera_keys}")
            
        # Stack images along camera dimension
        images = torch.stack(image_tensors, dim=2)  # (B, T, num_cameras, C, H, W)
        
        # Ensure images are float and in correct range
        if not torch.is_floating_point(images):
            images = images.float() / 255.0
            
        # Resize images to 224x224 maintaining aspect ratio
        B, T, num_cameras = images.shape[:3]
        images = images.view(B * T * num_cameras, *images.shape[3:])  # (B*T*num_cameras, C, H, W)
        print("images shape before resize", images.shape)
        images = self.resize_with_pad(images, 224, 224, pad_value=0.0)
        print("images shape after resize", images.shape)
        images = images.view(B, T, num_cameras, *images.shape[1:])  # (B, T, num_cameras, C, 224, 224)
            
        # Apply CLIP preprocessing
        images = clip_transform(images)
        
        return images

    def forward(
        self,
        batch: dict[str, torch.Tensor],  # Dictionary containing observation.image.*, task, OBS_ROBOT, and ACTION keys
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Forward pass for training.
        
        Args:
            batch: Dictionary containing:
                - observation.image.*: Image tensors from multiple cameras (B, T, C, H, W)
                - task: Text input (B, max_text_len)
                - OBS_ROBOT: Proprioceptive state (B, T, proprio_dim)
                - ACTION: Ground truth actions (B, T, action_horizon, action_dim)
        
        Returns:
            Tuple of (loss, loss_dict) where:
                - loss: The computed loss tensor
                - loss_dict: Dictionary containing loss values for logging
        """
        # Prepare images
        images = self.prepare_images(batch)
        
        # Extract and prepare other inputs
        text = self.prepare_language(batch)  # (B, max_text_len)
        proprio = batch[OBS_ROBOT]  # (B, T, proprio_dim)
        print("proprio shape", proprio.shape)
        gt_actions = batch[ACTION]  # (B, T, action_horizon, action_dim)
        
        # Create mask for SOS, EOS, and padding tokens
        if self.pool_true_text:
            text_mask = create_text_mask(text, self.sot_token, self.eot_token, self.first_k_tokens)
            text_mask = text_mask.to(images.device)
        else:
            text_mask = None

        # perform action generation
        # actions shape: (B, T, action_horizon, action_dim)
        actions = self.forward_actions(images, text, proprio, text_mask)
        
        # Calculate loss
        loss = F.l1_loss(actions, gt_actions)
        loss_dict = {"l1_loss": loss.item()}
        
        return loss, loss_dict
        
    def __repr__(self):
        """
        Return a string representation of the OTTER model.
        """
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters())
        
        def format_number(num):
            if num >= 1_000_000:
                return f"{num/1_000_000:.2f}M"
            elif num >= 1_000:
                return f"{num/1_000:.2f}K"
            return str(num)

        width = 70  # Slightly narrower for better display in notebooks/console
        border = "=" * width
        section_border = "-" * width
        
        # Collect all information
        model_info = [
            ("Model Architecture", [
                ("CLIP Model", self.clip_model_name),
                ("Cameras", f"{len(self.camera_keys)} ({', '.join(self.camera_keys)})"),
                ("Seq Length", self.seq_length),
                ("Action Horizon", self.action_horizon)
            ]),
            ("Dimensions", [
                ("Proprio Input", self.proprio_input_dim),
                ("Action Dim", self.action_dim),
                ("Num Readouts", self.num_readouts),
                ("First K Tokens", self.first_k_tokens),
                ("Vision Pool Out (per camera)", self.vision_poolings[0].output_dim),
                ("Text Pool Out", self.text_pooling.output_dim),
                ("Proprio Out", self.proprio_output_dim),
                ("FT Dim", self.f_t_dim),
                ("Transformer In", self.input_projection.out_features)
            ]),
            ("Parameters", [
                ("Total", format_number(count_parameters(self))),
                ("CLIP", format_number(count_parameters(self.clip_model))),
                ("Policy", format_number(count_parameters(self.policy))),
                ("Trainable (incl. modality enc.)", format_number(sum(p.numel() for p in self.parameters() if p.requires_grad)))
            ])
        ]

        # Build the string representation
        lines = [
            border,
            f"{'OTTER Model':^{width}}",
            border,
            ""
        ]

        for section_title, items in model_info:
            lines.append(f"{section_title:^{width}}")
            lines.append(section_border)
            
            # Find maximum length for alignment
            max_key_length = max(len(k) for k, _ in items)
            
            for key, value in items:
                # Add each line with proper spacing
                lines.append(f"{key:<{max_key_length}} : {value:>}")
            lines.append("")

        lines.append(border)

        # Join all lines with newlines
        return "\n".join(lines)