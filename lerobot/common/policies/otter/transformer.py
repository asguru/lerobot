import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
# from otter.util.args import ModelConfig
from lerobot.common.policies.otter.configuration_otter import OtterConfig
from timm.layers.mlp import Mlp

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"Dimension {dim} should be divisible by 2")
            
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Create position indices for the sequence length
        pos = torch.arange(max_seq_len).float()
        sinusoid_inp = torch.einsum("i,j->ij", pos, inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        self.register_buffer("emb", emb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return self.emb[:seq_len, :]  # [seq_len, dim]

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, pos_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # q, k: [batch, n_heads, seq_len, head_dim]
    # pos_emb: [seq_len, head_dim]
    seq_len = q.shape[2]
    head_dim = q.shape[3]
    
    # Ensure pos_emb has the right sequence length
    pos_emb = pos_emb[:seq_len, :]
    
    # Split into real and imaginary parts
    cos = pos_emb[..., :head_dim//2]  # [seq_len, head_dim//2]
    sin = pos_emb[..., head_dim//2:]  # [seq_len, head_dim//2]
    
    # Reshape for broadcasting
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim//2]
    sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim//2]
    
    # Split q and k into real and imaginary parts
    q1, q2 = q[..., :head_dim//2], q[..., head_dim//2:]
    k1, k2 = k[..., :head_dim//2], k[..., head_dim//2:]
    
    # Apply rotation
    q_out = torch.cat([
        q1 * cos - q2 * sin,
        q2 * cos + q1 * sin
    ], dim=-1)
    
    k_out = torch.cat([
        k1 * cos - k2 * sin,
        k2 * cos + k1 * sin
    ], dim=-1)
    
    return q_out, k_out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, config : OtterConfig):
        super().__init__()
        if config.transformer_dim % config.transformer_heads != 0:
            raise ValueError(
                f"Hidden size {config.transformer_dim} is not divisible by number of attention heads {config.transformer_heads}"
            )
        self.transformer_heads = config.transformer_heads
        self.attention_head_size = int(config.transformer_dim / config.transformer_heads)
        self.all_head_size = self.transformer_heads * self.attention_head_size

        self.query = nn.Linear(config.transformer_dim, self.all_head_size)
        self.key = nn.Linear(config.transformer_dim, self.all_head_size)
        self.value = nn.Linear(config.transformer_dim, self.all_head_size)
        self.rope = RotaryPositionalEmbedding(self.attention_head_size, config.max_position_embeddings)
        
        self.dropout = config.attention_probs_dropout_prob
        self.out = nn.Linear(config.transformer_dim, config.transformer_dim)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.transformer_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # Apply RoPE to query and key
        pos_emb = self.rope(hidden_states)  # [seq_len, head_dim]
        query_layer, key_layer = apply_rotary_pos_emb(query_layer, key_layer, pos_emb)

        # Convert attention mask to proper format if provided
        if attention_mask is not None:
            # attention_mask should be broadcastable to [batch, num_heads, seq_len, seq_len]
            attention_mask = attention_mask.unsqueeze(1)  # [batch, 1, seq_len, seq_len]

        # Use torch's scaled_dot_product_attention
        context_layer = F.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        return self.out(context_layer)
    

class TransformerBlock(nn.Module):
    def __init__(self, config : OtterConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        intermediate_size = config.transformer_dim * config.transformer_expansion_factor
        self.intermediate = nn.Linear(config.transformer_dim, intermediate_size)
        self.output = nn.Linear(intermediate_size, config.transformer_dim)
        self.layernorm1 = nn.LayerNorm(config.transformer_dim)
        self.layernorm2 = nn.LayerNorm(config.transformer_dim)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.activation = nn.GELU()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        
        if attention_mask is not None:
            assert not is_causal, "Causal mask not supported with attention mask"

        attention_output = self.attention(
            self.layernorm1(hidden_states),
            attention_mask,
            is_causal,
        )
        hidden_states = hidden_states + self.dropout(attention_output)

        layer_output = self.layernorm2(hidden_states)
        layer_output = self.intermediate(layer_output)
        layer_output = self.activation(layer_output)
        layer_output = self.output(layer_output)
        
        return hidden_states + self.dropout(layer_output)

class CausalTransformer(nn.Module):
    """Main policy transformer with RoPE and causal attention"""
    def __init__(
        self,
        config: OtterConfig,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(config)
            for _ in range(config.transformer_layers)
        ])
        
    def forward(
        self, 
        x: torch.Tensor,
    ) -> torch.Tensor:
        # Forward through each transformer block
        for layer in self.layers:
            x = layer.forward(x, is_causal=True)
        return x

class CrossAttentionBlock(nn.Module):
    def __init__(self, kv_input_dim : int, q_dim : int, mlp_dim : int, num_heads : int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            q_dim, 
            num_heads, 
            batch_first=True, 
            kdim=kv_input_dim, 
            vdim=kv_input_dim,
        )
        self.mlp = Mlp(
            in_features=q_dim, 
            hidden_features=mlp_dim, 
            out_features=q_dim,
        )
        self.q_layer_norm = nn.LayerNorm(q_dim)
        self.layer_norm = nn.LayerNorm(q_dim)
    
    def forward(self, q: torch.Tensor, kv: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
        the purpose of attention. For a float mask, it will be directly added to the corresponding ``key`` value.

        mask will be true at start of text, end of text, and padding tokens
        """
        # mask: [batch, seq_len], True for tokens to attend to
        q = self.q_layer_norm(q)
        attn_out, _ = self.attention(q, kv, kv, key_padding_mask=mask)
        q = q + attn_out  # Add residual connection
        q = self.layer_norm(q)
        x = self.mlp(q)
        return q + x

class AttentionPooling(nn.Module):
    """Attention pooling layer to compress sequence of tokens into a single token"""
    def __init__(self, input_dim: int, output_dim: int, num_heads: int = 8, num_layers: int = 2, num_readouts: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_readouts = num_readouts
        self.intermediate_dim = self.output_dim // self.num_readouts
        assert self.intermediate_dim * self.num_readouts == self.output_dim, "Output dim must be divisible by num readouts"
        self.query = nn.Parameter(torch.randn(1, num_readouts, self.intermediate_dim))
        self.layer_norm = nn.LayerNorm(self.intermediate_dim)
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(
                kv_input_dim=input_dim, 
                q_dim=self.intermediate_dim, 
                mlp_dim=self.output_dim, 
                num_heads=num_heads
            )
            for _ in range(num_layers)
        ])
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
        the purpose of attention. For a float mask, it will be directly added to the corresponding ``key`` value.

        mask will be true at start of text, end of text, and padding tokens
        """
        # mask: [batch, seq_len], True for tokens to attend to
        batch_size = x.shape[0]
        query = self.query.expand(batch_size, -1, -1)
            
        for layer in self.blocks:
            query = layer(query, x, mask)
        
        query = self.layer_norm(query)
        return query.reshape(batch_size, -1)  # dimension: [batch, output_dim]