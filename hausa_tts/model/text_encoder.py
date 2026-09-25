import math
import torch
import torch.nn as nn
from hausa_tts.model.modules import LayerNorm

def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)

class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional relative positional bias."""
    def __init__(self, channels, out_channels, n_heads, p_dropout=0.0, 
                 window_size=None, heads_share=True):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.heads_share = heads_share
        
        self.k_channels = channels // n_heads
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.drop = nn.Dropout(p_dropout)
        
        if window_size is not None:
            self.emb_rel_k = nn.Parameter(torch.randn(1, window_size * 2 + 1, self.k_channels))
            self.emb_rel_v = nn.Parameter(torch.randn(1, window_size * 2 + 1, self.k_channels))
        
    def forward(self, x, c, attn_mask=None):
        b, c_, t = x.size()
        q = self.conv_q(x).view(b, self.n_heads, self.k_channels, t)
        k = self.conv_k(c).view(b, self.n_heads, self.k_channels, t)
        v = self.conv_v(c).view(b, self.n_heads, self.k_channels, t)
        
        scores = torch.einsum("b h d t, b h d s -> b h t s", q, k) / math.sqrt(self.k_channels)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)
        p_attn = torch.softmax(scores, dim=-1)
        p_attn = self.drop(p_attn)
        output = torch.einsum("b h t s, b h d s -> b h d t", p_attn, v)
        output = output.reshape(b, c_, t)
        return self.conv_o(output)

class FFN(nn.Module):
    """Position-wise Feed-Forward Network with conv1d."""
    def __init__(self, in_channels, out_channels, filter_channels, kernel_size, 
                 p_dropout=0.0):
        super().__init__()
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size//2)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x, x_mask):
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        return x * x_mask

class Encoder(nn.Module):
    """Transformer encoder stack with relative positional encoding."""
    def __init__(self, hidden_channels, filter_channels, n_heads, n_layers, 
                 kernel_size=1, p_dropout=0.0, window_size=4):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.drop = nn.Dropout(p_dropout)
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        for _ in range(n_layers):
            self.attn_layers.append(MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout, window_size))
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout))
            self.norm_layers_2.append(LayerNorm(hidden_channels))
            
    def forward(self, x, x_mask):
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        for i in range(self.n_layers):
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)
            
            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)
        return x * x_mask

class TextEncoder(nn.Module):
    """Transformer text encoder with parallel tone embedding stream."""
    def __init__(self, 
                 n_vocab: int,           
                 out_channels: int,      
                 hidden_channels: int,   
                 filter_channels: int,   
                 n_heads: int,           
                 n_layers: int,          
                 kernel_size: int,       
                 p_dropout: float,       
                 n_tones: int = 4,       
                 tone_embedding_dim: int = 32,  
                 gin_channels: int = 0   
                 ):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        
        self.emb = nn.Embedding(n_vocab, hidden_channels)
        self.tone_emb = nn.Embedding(n_tones, tone_embedding_dim)
        self.tone_proj = nn.Linear(tone_embedding_dim, hidden_channels)
        
        self.encoder = Encoder(hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
        
        if gin_channels > 0:
            self.cond = nn.Conv1d(gin_channels, hidden_channels, 1)
        else:
            self.cond = None
            
    def forward(self, x: torch.Tensor, x_lengths: torch.Tensor, 
                tones: torch.Tensor = None, g: torch.Tensor = None):
        """Args:
            x: (B, T) phoneme IDs
            x_lengths: (B,) lengths
            tones: (B, T) tone IDs, optional
            g: (B, gin_channels, 1) speaker embedding, optional
        Returns:
            x: (B, hidden_channels, T) - encoder hidden states
            m: (B, out_channels, T) - prior mean
            logs: (B, out_channels, T) - prior log variance
            x_mask: (B, 1, T) - mask
        """
        x_mask = sequence_mask(x_lengths, x.size(1)).unsqueeze(1).to(x.dtype)
        
        x = self.emb(x) * math.sqrt(self.hidden_channels)
        if tones is not None:
            t_emb = self.tone_emb(tones)
            t_emb = self.tone_proj(t_emb)
            x = x + t_emb
            
        x = torch.transpose(x, 1, 2)
        
        if g is not None and self.cond is not None:
            x = x + self.cond(g)
            
        x = self.encoder(x, x_mask)
        stats = self.proj(x) * x_mask
        
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs, x_mask
