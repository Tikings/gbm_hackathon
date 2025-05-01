import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


# Helper functions
def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def pad_to_multiple(tensor, multiple, dim=-1, value=0):
    """Pad tensor to be multiple along dimension"""
    seqlen = tensor.shape[dim]
    m = seqlen % multiple
    if m == 0:
        return tensor
    padding = multiple - m
    padded_tensor = F.pad(tensor, [0, 0] * (tensor.dim() - dim - 1) + [0, padding] + [0, 0] * dim, value=value)
    return padded_tensor


def look_around(x, backward=1, forward=0, pad_value=-1, dim=2):
    """
    Extend the local window attention by looking backward and forward
    """
    t = x.shape[dim]
    dims = (len(x.shape) - dim) * (0, 0)
    padded_x = F.pad(x, (*dims, backward, forward), value=pad_value)
    tensors = [padded_x.narrow(dim, i, t) for i in range(backward + forward + 1)]
    return torch.cat(tensors, dim=dim + 1)


# Positional embedding for local attention
class RelativePositionalEmbedding(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.weight = nn.Parameter(torch.randn(heads, dim * 2 - 1))
        
    def forward(self, q_len, k_len):
        # Generate relative position indices
        seq_len = max(q_len, k_len)
        context_position = torch.arange(k_len, device=self.weight.device)[:, None]
        memory_position = torch.arange(q_len, device=self.weight.device)[None, :]
        relative_position = memory_position - context_position + q_len - 1  # Shift to get positive indices
        
        # Get embeddings
        embeddings = self.weight[:, relative_position.flatten()].view(self.heads, k_len, q_len)
        return embeddings.permute(0, 2, 1)  # [heads, q_len, k_len]


class LocalAttention(nn.Module):
    def __init__(
        self,
        window_size,
        causal=False,
        autopad=False,
        dropout=0.,
        look_forward=0,
        rel_pos_emb_config=None,
        use_mask=True,
    ):
        super().__init__()
        self.window_size = window_size
        self.causal = causal
        self.autopad = autopad
        self.look_forward = look_forward
        self.dropout = nn.Dropout(dropout)
        self.use_mask = use_mask
        
        # Set up relative positional embedding if specified
        self.has_rel_pos_emb = exists(rel_pos_emb_config)
        if self.has_rel_pos_emb:
            dim_head, heads = rel_pos_emb_config
            self.rel_pos_emb = RelativePositionalEmbedding(dim_head, heads)
        
        # Determine look around parameters based on causal setting
        self.look_backward = 1
        self.look_forward = 0 if causal else look_forward

    def forward(self, q, k, v, input_mask=None):
        """
        q, k, v: [batch, heads, seq_len, dim_head]
        input_mask: [batch, seq_len]
        """
        b, h, t, d = q.shape
        device = q.device
        
        # Autopad if needed to multiple of window_size
        if self.autopad and t % self.window_size != 0:
            excess = self.window_size - (t % self.window_size)
            q = F.pad(q, (0, 0, 0, excess), value=0)
            k = F.pad(k, (0, 0, 0, excess), value=0)
            v = F.pad(v, (0, 0, 0, excess), value=0)
            if exists(input_mask):
                input_mask = F.pad(input_mask, (0, excess), value=False)
            t = q.shape[2]  # Update sequence length
        
        # Set window size
        window_size = self.window_size
        assert t % window_size == 0, f'sequence length {t} must be divisible by window size {window_size}'
        
        # Reshape into windows
        # [batch, heads, windows, window_size, dim_head]
        windows = t // window_size
        q = q.reshape(b, h, windows, window_size, d)
        k = k.reshape(b, h, windows, window_size, d)
        v = v.reshape(b, h, windows, window_size, d)
        
        # Handle looking around (expanding window context)
        if self.look_backward > 0 or self.look_forward > 0:
            k_expanded = look_around(k, self.look_backward, self.look_forward, pad_value=0, dim=2)
            v_expanded = look_around(v, self.look_backward, self.look_forward, pad_value=0, dim=2)
            
            # [batch, heads, windows, window_size, window_size * (1 + look_backward + look_forward)]
            expand_size = window_size * (1 + self.look_backward + self.look_forward)
            k_expanded = k_expanded.reshape(b, h, windows, window_size, expand_size)
            v_expanded = v_expanded.reshape(b, h, windows, window_size, expand_size)
        else:
            k_expanded = k
            v_expanded = v
        
        # Apply attention
        # Calculate similarity
        sim = torch.einsum('bhwid,bhwjd->bhwij', q, k_expanded)
        
        # Add relative positional embeddings if enabled
        if self.has_rel_pos_emb:
            pos_emb = self.rel_pos_emb(window_size, k_expanded.shape[3])
            pos_emb = repeat(pos_emb, 'h i j -> b h w i j', b=b, w=windows)
            sim = sim + pos_emb
        
        # Apply causal mask if needed
        if self.causal:
            causal_mask = torch.ones(window_size, window_size, device=device).triu_(1).bool()
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)
        
        # Apply input mask if provided
        if exists(input_mask) and self.use_mask:
            input_mask = input_mask.reshape(b, 1, windows, window_size, 1)
            mask_value = -torch.finfo(sim.dtype).max
            sim = sim.masked_fill(~input_mask, mask_value)
        
        # Calculate attention weights
        attn = F.softmax(sim, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.einsum('bhwij,bhwjd->bhwid', attn, v_expanded)
        
        # Reshape back
        out = out.reshape(b, h, t, d)
        
        # Remove padding if needed
        if self.autopad and t != q.shape[2]:
            out = out[:, :, :t - excess]
            
        return out