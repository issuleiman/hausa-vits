import torch
import torch.nn as nn
from hausa_tts.model.modules import ResidualCouplingLayer, Flip

class ResidualCouplingBlock(nn.Module):
    """Stack of residual coupling layers for normalizing flows.
    Transforms the prior distribution to match the posterior.
    """
    def __init__(self,
                 channels: int,
                 hidden_channels: int,
                 kernel_size: int = 5,
                 dilation_rate: int = 1,
                 n_layers: int = 4,
                 n_flows: int = 4,
                 gin_channels: int = 0):
        super().__init__()
        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels, mean_only=True))
            self.flows.append(Flip())
            
    def forward(self, x, x_mask, g=None, reverse=False):
        """Args:
            x: (B, channels, T)
            x_mask: (B, 1, T)
            g: (B, gin_channels, 1)
            reverse: If True, run flows in reverse (for inference)
        Returns:
            x: (B, channels, T) - transformed
        """
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        return x
