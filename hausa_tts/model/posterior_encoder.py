import torch
import torch.nn as nn
from hausa_tts.model.modules import WN

def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)

class PosteriorEncoder(nn.Module):
    """WaveNet-based posterior encoder.
    Encodes linear spectrogram into latent representation during training.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 hidden_channels: int,
                 kernel_size: int = 5,
                 dilation_rate: int = 1,
                 n_layers: int = 16,
                 gin_channels: int = 0):
        super().__init__()
        self.out_channels = out_channels
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
        
    def forward(self, x, x_lengths, g=None):
        """Args:
            x: (B, in_channels, T) linear spectrogram
            x_lengths: (B,) lengths
            g: (B, gin_channels, 1) speaker embedding
        Returns:
            z: (B, out_channels, T) - sampled latent (reparameterization trick)
            m: (B, out_channels, T) - posterior mean
            logs: (B, out_channels, T) - posterior log variance
            x_mask: (B, 1, T)
        """
        x_mask = sequence_mask(x_lengths, x.size(2)).unsqueeze(1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask
