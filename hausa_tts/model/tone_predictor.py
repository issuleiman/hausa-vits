import torch
import torch.nn as nn
import math

class TonePredictor(nn.Module):
    """Predicts per-phoneme tones from text context."""
    def __init__(self,
                 n_vocab: int,
                 n_tones: int = 4,
                 hidden_channels: int = 192,
                 n_layers: int = 3,
                 n_heads: int = 2,
                 filter_channels: int = 512,
                 kernel_size: int = 3,
                 p_dropout: float = 0.1):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.emb = nn.Embedding(n_vocab, hidden_channels)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_channels,
            nhead=n_heads,
            dim_feedforward=filter_channels,
            dropout=p_dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.proj = nn.Linear(hidden_channels, n_tones)
        
    def _get_sinusoidal_positional_encoding(self, seq_len: int, d_model: int, device: torch.device):
        position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=device) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, seq_len, d_model, device=device)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x: torch.Tensor, x_lengths: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (B, T) phoneme IDs
            x_lengths: (B,) lengths
        Returns:
            tone_logits: (B, T, n_tones) - logits for each position
        """
        b, t = x.size()
        x_emb = self.emb(x) * math.sqrt(self.hidden_channels)
        pe = self._get_sinusoidal_positional_encoding(t, self.hidden_channels, x.device)
        x_emb = x_emb + pe
        
        pad_mask = torch.arange(t, device=x.device).unsqueeze(0) >= x_lengths.unsqueeze(1)
        
        out = self.transformer(x_emb, src_key_padding_mask=pad_mask)
        logits = self.proj(out)
        return logits
    
    @torch.no_grad()
    def predict(self, x: torch.Tensor, x_lengths: torch.Tensor) -> torch.Tensor:
        """Predict tone IDs (argmax of logits).
        Returns:
            tone_ids: (B, T) predicted tone IDs
        """
        logits = self.forward(x, x_lengths)
        return torch.argmax(logits, dim=-1)
