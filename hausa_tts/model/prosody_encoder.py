import torch
import torch.nn as nn

class ProsodyEncoder(nn.Module):
    """Reference encoder for capturing utterance-level prosody."""
    def __init__(self,
                 n_mel_channels: int = 80,
                 prosody_embedding_dim: int = 64,
                 ref_enc_filters: list = [32, 32, 64, 64, 128, 128],
                 ref_enc_gru_size: int = 128):
        super().__init__()
        
        in_channels = 1
        convs = []
        for out_channels in ref_enc_filters:
            convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1))
            convs.append(nn.BatchNorm2d(out_channels))
            convs.append(nn.ReLU())
            in_channels = out_channels
            
        self.convs = nn.Sequential(*convs)
        
        out_mel_dim = n_mel_channels
        for _ in ref_enc_filters:
            out_mel_dim = (out_mel_dim + 2*1 - 3) // 2 + 1
            
        self.gru = nn.GRU(input_size=ref_enc_filters[-1] * out_mel_dim,
                          hidden_size=ref_enc_gru_size,
                          batch_first=True)
                          
        self.proj = nn.Linear(ref_enc_gru_size, prosody_embedding_dim)
        
    def forward(self, mel: torch.Tensor, mel_lengths: torch.Tensor = None) -> torch.Tensor:
        """Args:
            mel: (B, n_mel_channels, T) mel spectrogram
            mel_lengths: (B,) optional lengths for packing
        Returns:
            prosody_embedding: (B, prosody_embedding_dim, 1) - broadcastable
        """
        b, c, t = mel.size()
        x = mel.unsqueeze(1)
        x = self.convs(x)
        
        x = x.transpose(1, 3).contiguous()
        b, t_prime, f, c_prime = x.size()
        x = x.view(b, t_prime, f * c_prime)
        
        self.gru.flatten_parameters()
        _, hidden = self.gru(x)
        
        hidden = hidden.squeeze(0)
        prosody = self.proj(hidden)
        
        return prosody.unsqueeze(-1)

class ProsodyPredictor(nn.Module):
    """Predicts prosody embedding from text."""
    def __init__(self, in_channels: int = 192, prosody_embedding_dim: int = 64,
                 n_layers: int = 2, kernel_size: int = 3):
        super().__init__()
        convs = []
        for _ in range(n_layers):
            convs.append(nn.Conv1d(in_channels, in_channels, kernel_size, padding=kernel_size//2))
            convs.append(nn.ReLU())
            convs.append(nn.BatchNorm1d(in_channels))
            
        self.convs = nn.Sequential(*convs)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(in_channels, prosody_embedding_dim)
        
    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (B, in_channels, T) text encoder hidden states
            x_mask: (B, 1, T)
        Returns:
            prosody_embedding: (B, prosody_embedding_dim, 1)
        """
        x = self.convs(x * x_mask)
        x = self.pool(x).squeeze(-1)
        prosody = self.proj(x)
        return prosody.unsqueeze(-1)
