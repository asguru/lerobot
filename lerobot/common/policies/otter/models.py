import torch 
import torch.nn as nn
import torch.nn.functional as F

def sincos_position_embedding(seq_len: int, dim: int) -> torch.Tensor:
    """Generate sinusoidal positional embedding"""
    # shape: [seq_len, dim]
    pos = torch.arange(seq_len).float()
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    sinusoid_inp = torch.einsum("i,j->ij", pos, inv_freq)
    emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
    return emb

class TextAwareVisualExtraction(nn.Module):
    """Extract text-aware visual features using CLIP, following ClearCLIP approach"""
    def __init__(self, num_img_patches : int, vision_dim : int, temperature: float = 0.07):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))
        # mark position embedding as non-trainable param 
        self.register_buffer('pos_emb', sincos_position_embedding(num_img_patches, vision_dim))
        
    def forward(self, image_patch_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        # Calculate similarity between text and patch features
        # image_patch_features: (batch_size, num_patches, embedding_dim)
        # text_features: (batch_size, num_tokens, embedding_dim)
        similarity = torch.einsum('bij,bkj->bik', text_features, image_patch_features)
        
        # Apply temperature scaling and softmax
        attention = F.softmax(similarity / self.temperature.clamp(0, 100), dim=-1)
        
        # add position embedding to image patch features 
        pe_image_patch_features = image_patch_features + self.pos_emb
        # Get text-aware visual features by combining patch features according to attention (with position embedding)
        text_aware_features = torch.einsum('bik,bkj->bij', attention, pe_image_patch_features)
        
        return text_aware_features

class ProprioceptionEncoder(nn.Module):
    """Encode robot proprioception data"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

class ActionHead(nn.Module):
    """Predict actions from transformer outputs"""
    def __init__(self, input_dim: int, action_dim: int, horizon: int = 12):
        super().__init__()
        self.horizon = horizon
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, action_dim * horizon)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        actions = self.mlp(x)
        return actions.reshape(batch_size, self.horizon, -1)