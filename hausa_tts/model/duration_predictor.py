import torch
import torch.nn as nn
from hausa_tts.model.modules import DDSConv, ConvFlow, Log, Flip, ElementwiseAffine
import math

def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)

class StochasticDurationPredictor(nn.Module):
    """Flow-based stochastic duration predictor."""
    def __init__(self,
                 in_channels: int,
                 filter_channels: int,
                 kernel_size: int = 3,
                 p_dropout: float = 0.5,
                 n_flows: int = 4,
                 gin_channels: int = 0):
        super().__init__()
        self.pre = nn.Conv1d(in_channels, filter_channels, 1)
        self.dds = DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
        
        self.flows = nn.ModuleList()
        self.flows.append(Log())
        self.flows.append(ElementwiseAffine(2))
        for _ in range(n_flows):
            self.flows.append(ConvFlow(2, filter_channels, kernel_size, n_layers=3))
            self.flows.append(Flip())
            
        self.post = nn.Conv1d(filter_channels, 2, 1)
        
        if gin_channels > 0:
            self.cond = nn.Conv1d(gin_channels, filter_channels, 1)
        else:
            self.cond = None
            
    def forward(self, x, x_mask, w=None, g=None, reverse=False, noise_scale=1.0):
        """Args:
            x: (B, in_channels, T) hidden states from text encoder
            x_mask: (B, 1, T)
            w: (B, 1, T) ground truth durations (training only, when reverse=False)
            g: (B, gin_channels, 1) speaker embedding
            reverse: If True, sample durations (inference)
            noise_scale: noise scaling for sampling
        Returns:
            If reverse=False (training): negative log likelihood
            If reverse=True (inference): (B, 1, T) predicted durations
        """
        x = self.pre(x) * x_mask
        if g is not None and self.cond is not None:
            x = x + self.cond(g)
        x = self.dds(x, x_mask)
        
        if not reverse:
            h = self.post(x) * x_mask
            return h
        else:
            z = torch.randn(x.size(0), 2, x.size(2)).to(x.device) * noise_scale
            for flow in reversed(self.flows):
                z, _ = flow(z, x_mask, g=x, reverse=reverse)
            return z[:, 0, :].unsqueeze(1) * x_mask

class DurationPredictor(nn.Module):
    """Deterministic duration predictor (simpler alternative)."""
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size//2)
        self.norm1 = nn.LayerNorm(filter_channels)
        self.conv2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size//2)
        self.norm2 = nn.LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)
        self.drop = nn.Dropout(p_dropout)
        if gin_channels > 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)
        else:
            self.cond = None

    def forward(self, x, x_mask, g=None):
        if g is not None and self.cond is not None:
            x = x + self.cond(g)
        x = self.conv1(x * x_mask)
        x = torch.relu(x)
        x = self.norm1(x.transpose(1, 2)).transpose(1, 2)
        x = self.drop(x)
        x = self.conv2(x * x_mask)
        x = torch.relu(x)
        x = self.norm2(x.transpose(1, 2)).transpose(1, 2)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask
