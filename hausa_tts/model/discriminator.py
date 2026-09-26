import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm, spectral_norm

from hausa_tts.model.modules import get_padding


class DiscriminatorP(nn.Module):
    """Single-period sub-discriminator (MPD component).
    Reshapes 1D audio into 2D using a specific period, then applies 2D convs.
    Channel sizes: 1 -> 32 -> 128 -> 512 -> 1024 -> 1024  (standard VITS).
    """
    def __init__(self, period: int, kernel_size: int = 5, stride: int = 3,
                 use_spectral_norm: bool = False):
        super().__init__()
        self.period = period
        norm_f = spectral_norm if use_spectral_norm else weight_norm

        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(1,    32,   (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(nn.Conv2d(32,   128,  (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(nn.Conv2d(128,  512,  (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(nn.Conv2d(512,  1024, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(nn.Conv2d(1024, 1024, (kernel_size, 1), 1,           padding=(get_padding(kernel_size, 1), 0))),
        ])
        self.conv_post = norm_f(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: torch.Tensor):
        """Returns: (output, list_of_feature_maps)"""
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)
        for layer in self.convs:
            x = F.leaky_relu(layer(x), 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class DiscriminatorS(nn.Module):
    """Single-scale sub-discriminator (MSD component).
    Standard 1D grouped conv discriminator.
    Channel sizes: 1 -> 16 -> 64 -> 256 -> 1024 -> 1024 -> 1024  (standard VITS).
    Groups must evenly divide in_channels:
        16 / 4 = 4   ✓
        64 / 16 = 4  ✓
       256 / 64 = 4  ✓
      1024 / 256 = 4 ✓
    """
    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()
        norm_f = spectral_norm if use_spectral_norm else weight_norm

        self.convs = nn.ModuleList([
            norm_f(nn.Conv1d(1,    16,   15, 1,           padding=7)),
            norm_f(nn.Conv1d(16,   64,   41, 4, groups=4,   padding=20)),
            norm_f(nn.Conv1d(64,   256,  41, 4, groups=16,  padding=20)),
            norm_f(nn.Conv1d(256,  1024, 41, 4, groups=64,  padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 5,  1,             padding=2)),
        ])
        self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x: torch.Tensor):
        """Returns: (output, list_of_feature_maps)"""
        fmap = []
        for layer in self.convs:
            x = F.leaky_relu(layer(x), 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiPeriodDiscriminator(nn.Module):
    """Ensemble of 5 period-based discriminators: periods [2, 3, 5, 7, 11].
    Each sub-disc is a DiscriminatorP (2D conv on period-reshaped audio).
    """
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(2),
            DiscriminatorP(3),
            DiscriminatorP(5),
            DiscriminatorP(7),
            DiscriminatorP(11),
        ])

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor):
        """
        Args:
            y:     (B, 1, T) real audio
            y_hat: (B, 1, T) generated audio
        Returns:
            y_d_rs, y_d_gs: discriminator logits (list)
            fmap_rs, fmap_gs: feature maps (list of lists)
        """
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for d in self.discriminators:
            y_d_r,  fmap_r = d(y)
            y_d_g,  fmap_g = d(y_hat)
            y_d_rs.append(y_d_r);  y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r); fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class MultiScaleDiscriminator(nn.Module):
    """3 DiscriminatorS at different audio scales (full, /2, /4).
    First disc uses spectral_norm (more stable at full resolution).
    """
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),   # full resolution
            DiscriminatorS(),                          # 2x downsampled
            DiscriminatorS(),                          # 4x downsampled
        ])
        self.meanpools = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2),
        ])

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor):
        """Same return format as MultiPeriodDiscriminator."""
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y     = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            y_d_r,  fmap_r = d(y)
            y_d_g,  fmap_g = d(y_hat)
            y_d_rs.append(y_d_r);  y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r); fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
