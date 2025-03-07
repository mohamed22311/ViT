import torch
from torch import nn
import numpy as np

class MultiHeadSelfAttention(nn.Module):
    """Creates a multi-head self-attention block ("MSA block" for short)."""
    def __init__(self, embed_dim: int=768, num_heads:int=12, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads."
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(normalized_shape=embed_dim)
        
    def forward(self, x, mask=None):
        batch_size, seq_length, embed_dim = x.size()
        
        x = self.layer_norm(x)

        # Linear projection to obtain queries, keys, and values
        qkv = self.qkv_proj(x)  # (batch_size, seq_length, 3 * embed_dim)
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1)  # (num_heads, batch_size, 3 * head_dim, seq_length)
        queries, keys, values = torch.chunk(qkv, 3, dim=2)  # Each is (num_heads, batch_size, head_dim, seq_length)
        
        # Scaled dot-product attention
        scores = torch.matmul(queries.transpose(-2, -1), keys) / (self.head_dim ** 0.5)  # (num_heads, batch_size, seq_length, seq_length)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        
        attn_weights = torch.softmax(scores, dim=-1)  # (num_heads, batch_size, seq_length, seq_length)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, values.transpose(-2, -1))  # (num_heads, batch_size, seq_length, head_dim)
        attn_output = attn_output.permute(1, 2, 0, 3).contiguous()  # (batch_size, seq_length, num_heads, head_dim)
        attn_output = attn_output.view(batch_size, seq_length, self.embed_dim)  # (batch_size, seq_length, embed_dim)
        
        output = self.o_proj(attn_output)  # (batch_size, seq_length, embed_dim)
        
        return output
