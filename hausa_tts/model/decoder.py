import torch
import torch.nn as nn
from torch.nn.utils import weight_norm, remove_weight_norm
from hausa_tts.model.modules import ResBlock1, ResBlock2, init_weights, get_padding, LRELU_SLOPE

class Generator(nn.Module):
    """HiFi-GAN v1 generator. Converts latent z to raw waveform."""
    def __init__(self,
                 initial_channel: int,
                 resblock: str = '1',
                 resblock_kernel_sizes: list = [3, 7, 11],
                 resblock_dilation_sizes: list = [[1,3,5], [1,3,5], [1,3,5]],
                 upsample_rates: list = [8, 8, 2, 2],
                 upsample_initial_channel: int = 512,
                 upsample_kernel_sizes: list = [16, 16, 4, 4],
                 gin_channels: int = 0):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = weight_norm(nn.Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3))
        resblock_cls = ResBlock1 if resblock == '1' else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                nn.ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)),
                                   k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock_cls(ch, k, d))

        self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3))
        
        if gin_channels > 0:
            self.cond = nn.Conv1d(gin_channels, initial_channel, 1)
        else:
            self.cond = None
            
        self.apply(init_weights)

    def forward(self, x, g=None):
        """Args:
            x: (B, initial_channel, T) latent representation
            g: (B, gin_channels, 1) speaker embedding
        Returns:
            x: (B, 1, T*prod(upsample_rates)) raw waveform
        """
        if g is not None and self.cond is not None:
            x = x + self.cond(g)
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = torch.nn.functional.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = torch.nn.functional.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
