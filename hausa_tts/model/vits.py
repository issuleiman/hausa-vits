import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict

from hausa_tts.model.text_encoder import TextEncoder
from hausa_tts.model.posterior_encoder import PosteriorEncoder
from hausa_tts.model.flow import ResidualCouplingBlock
from hausa_tts.model.decoder import Generator
from hausa_tts.model.duration_predictor import StochasticDurationPredictor, DurationPredictor
from hausa_tts.model.tone_predictor import TonePredictor
from hausa_tts.model.prosody_encoder import ProsodyEncoder, ProsodyPredictor
from hausa_tts.model.discriminator import MultiPeriodDiscriminator, MultiScaleDiscriminator
from hausa_tts.model.mas import monotonic_alignment_search
from hausa_tts.model.modules import sequence_mask, generate_path

def rand_slice_segments(x: torch.Tensor, x_lengths: torch.Tensor, 
                        segment_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Randomly slice segments from a batch for decoder training.
    Args:
        x: (B, C, T) 
        x_lengths: (B,)
        segment_size: length of segment to slice
    Returns:
        segments: (B, C, segment_size)
        ids_start: (B,) start indices
    """
    b, c, t = x.size()
    max_start = torch.clamp(x_lengths - segment_size, min=0).to(dtype=torch.long)
    ids_start = (torch.rand(b, device=x.device) * (max_start.float() + 1)).to(dtype=torch.long)
    ids_start = torch.clamp(ids_start, max=t - segment_size) if t >= segment_size else ids_start
    
    segments = slice_segments(x, ids_start, segment_size)
    return segments, ids_start

def slice_segments(x: torch.Tensor, ids_str: torch.Tensor, 
                   segment_size: int) -> torch.Tensor:
    """Slice segments given start indices."""
    b, c, t = x.size()
    ret = torch.zeros((b, c, segment_size), dtype=x.dtype, device=x.device)
    for i in range(b):
        start = ids_str[i].item()
        end = min(start + segment_size, t)
        length = end - start
        ret[i, :, :length] = x[i, :, start:end]
    return ret

class HausaVITS(nn.Module):
    """Main HausaVITS model for Hausa Text-to-Speech.
    
    Supports:
    - Single-speaker and multi-speaker modes
    - Tone-aware text encoding (with explicit tones or predicted tones)
    - Prosody encoding for utterance-level style
    - Stochastic duration prediction
    - End-to-end waveform generation
    """
    def __init__(self,
                 n_vocab: int,
                 spec_channels: int,        # Linear spec channels (n_fft//2 + 1)
                 segment_size: int,          # Segment size for decoder training
                 inter_channels: int = 192,  # Latent channels
                 hidden_channels: int = 192, # Transformer hidden
                 filter_channels: int = 768, # FFN intermediate
                 n_heads: int = 2,
                 n_layers: int = 6,
                 kernel_size: int = 3,
                 p_dropout: float = 0.1,
                 resblock: str = '1',
                 resblock_kernel_sizes: list = [3, 7, 11],
                 resblock_dilation_sizes: list = [[1,3,5], [1,3,5], [1,3,5]],
                 upsample_rates: list = [8, 8, 2, 2],
                 upsample_initial_channel: int = 512,
                 upsample_kernel_sizes: list = [16, 16, 4, 4],
                 # Hausa-specific:
                 n_tones: int = 4,
                 tone_embedding_dim: int = 32,
                 use_tone_predictor: bool = True,
                 use_prosody_encoder: bool = True,
                 prosody_embedding_dim: int = 64,
                 n_mel_channels: int = 80,
                 # Multi-speaker:
                 n_speakers: int = 0,       # 0 = single speaker
                 gin_channels: int = 0,     # Speaker embedding dim
                 # Duration:
                 use_sdp: bool = True,
                 **kwargs):
        super().__init__()
        self.n_vocab = n_vocab
        self.spec_channels = spec_channels
        self.segment_size = segment_size
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.use_tone_predictor = use_tone_predictor
        self.use_prosody_encoder = use_prosody_encoder
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels
        self.use_sdp = use_sdp

        # --- Text Encoder (with tone embeddings) ---
        self.enc_p = TextEncoder(
            n_vocab=n_vocab,
            out_channels=inter_channels,
            hidden_channels=hidden_channels,
            filter_channels=filter_channels,
            n_heads=n_heads,
            n_layers=n_layers,
            kernel_size=kernel_size,
            p_dropout=p_dropout,
            n_tones=n_tones,
            tone_embedding_dim=tone_embedding_dim,
        )

        # --- Posterior Encoder ---
        self.enc_q = PosteriorEncoder(
            in_channels=spec_channels,
            out_channels=inter_channels,
            hidden_channels=hidden_channels,
            kernel_size=5,
            dilation_rate=1,
            n_layers=16,
            gin_channels=gin_channels,
        )

        # --- Normalizing Flows ---
        self.flow = ResidualCouplingBlock(
            channels=inter_channels,
            hidden_channels=hidden_channels,
            kernel_size=5,
            dilation_rate=1,
            n_layers=4,
            gin_channels=gin_channels,
        )

        # --- HiFi-GAN Decoder ---
        self.dec = Generator(
            initial_channel=inter_channels,
            resblock=resblock,
            resblock_kernel_sizes=resblock_kernel_sizes,
            resblock_dilation_sizes=resblock_dilation_sizes,
            upsample_rates=upsample_rates,
            upsample_initial_channel=upsample_initial_channel,
            upsample_kernel_sizes=upsample_kernel_sizes,
            gin_channels=gin_channels,
        )

        # --- Duration Predictor ---
        if use_sdp:
            self.dp = StochasticDurationPredictor(
                in_channels=hidden_channels,
                filter_channels=192,
                kernel_size=3,
                p_dropout=0.5,
                n_flows=4,
                gin_channels=gin_channels,
            )
        else:
            self.dp = DurationPredictor(
                in_channels=hidden_channels,
                filter_channels=256,
                kernel_size=3,
                p_dropout=0.5,
                gin_channels=gin_channels,
            )

        # --- Tone Predictor (Hausa-specific) ---
        if use_tone_predictor:
            self.tone_predictor = TonePredictor(
                n_vocab=n_vocab,
                n_tones=n_tones,
                hidden_channels=hidden_channels,
                n_layers=3,
                n_heads=n_heads,
                filter_channels=filter_channels,
                p_dropout=p_dropout,
            )

        # --- Prosody Encoder (Hausa-specific) ---
        if use_prosody_encoder:
            self.prosody_encoder = ProsodyEncoder(
                n_mel_channels=n_mel_channels,
                prosody_embedding_dim=prosody_embedding_dim,
            )
            self.prosody_predictor = ProsodyPredictor(
                in_channels=hidden_channels,
                prosody_embedding_dim=prosody_embedding_dim,
            )
            # Project prosody to conditioning dimension
            target_dim = gin_channels if gin_channels > 0 else hidden_channels
            self.prosody_proj = nn.Linear(prosody_embedding_dim, target_dim)

        # --- Speaker Embedding ---
        if n_speakers > 0:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)

    def forward(self, 
                x: torch.Tensor,           # (B, T_text) phoneme IDs
                x_lengths: torch.Tensor,   # (B,)
                y: torch.Tensor,           # (B, spec_channels, T_mel) linear spec
                y_lengths: torch.Tensor,   # (B,)
                tones: Optional[torch.Tensor] = None,  # (B, T_text) tone IDs
                sid: Optional[torch.Tensor] = None,    # (B,) speaker IDs
                mel: Optional[torch.Tensor] = None     # (B, n_mels, T_mel) for prosody
                ) -> Dict[str, torch.Tensor]:
        """Training forward pass.
        
        Returns dict with all values needed for loss computation.
        """
        # --- Speaker & prosody conditioning ---
        g = None
        if self.n_speakers > 0 and sid is not None:
            g = self.emb_g(sid).unsqueeze(-1)  # (B, gin_channels, 1)

        if self.use_prosody_encoder and mel is not None:
            prosody_emb = self.prosody_encoder(mel)  # (B, prosody_dim, 1)
            p_proj = self.prosody_proj(prosody_emb.squeeze(-1)).unsqueeze(-1)  # (B, target_dim, 1)
            g = g + p_proj if g is not None else p_proj

        # --- Tone prediction (when no tone labels provided) ---
        tone_logits = None
        if tones is None and self.use_tone_predictor:
            tone_logits = self.tone_predictor(x, x_lengths)  # (B, T, n_tones)
            tones = torch.argmax(tone_logits, dim=-1)        # (B, T)

        # --- Text encoding with tone embeddings ---
        x_enc, m_p, logs_p, x_mask = self.enc_p(x, x_lengths, tones)

        # --- Get tone logits from hidden states if we had explicit tones ---
        if self.use_tone_predictor and tone_logits is None:
            tone_logits = self.tone_predictor(x, x_lengths)

        # --- Posterior encoding ---
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)

        # --- Flow forward: posterior → prior space ---
        z_p = self.flow(z, y_mask, g=g, reverse=False)

        # --- Monotonic Alignment Search ---
        with torch.no_grad():
            # Compute log probability of attention (negative distance in latent space)
            s_p_sq_r = torch.exp(-2 * logs_p)  # 1/sigma_p^2
            neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True)
            neg_cent2 = torch.matmul(-0.5 * (z_p ** 2).transpose(1, 2), s_p_sq_r)
            neg_cent3 = torch.matmul(z_p.transpose(1, 2), (m_p * s_p_sq_r))
            neg_cent4 = torch.sum(-0.5 * (m_p ** 2) * s_p_sq_r, [1], keepdim=True)
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4

            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            attn = monotonic_alignment_search(neg_cent, attn_mask)

        # --- Duration predictor ---
        w = attn.sum(2)  # (B, 1, T_text) — duration per phoneme
        if self.use_sdp:
            l_length = self.dp(x_enc, x_mask, w, g=g)
        else:
            logw_ = torch.log(w + 1e-6) * x_mask
            logw = self.dp(x_enc, x_mask, g=g)
            l_length = torch.sum((logw - logw_) ** 2, [1, 2]) / torch.sum(x_mask)

        # --- Expand prior statistics using alignment ---
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        # --- Random segment slicing for decoder training ---
        z_slice, ids_slice = rand_slice_segments(z, y_lengths, self.segment_size)
        y_hat = self.dec(z_slice, g=g)

        return {
            'y_hat': y_hat,
            'ids_slice': ids_slice,
            'x_mask': x_mask,
            'y_mask': y_mask,
            'z': z,
            'z_p': z_p,
            'm_p': m_p,
            'logs_p': logs_p,
            'm_q': m_q,
            'logs_q': logs_q,
            'l_length': l_length,
            'attn': attn,
            'tone_logits': tone_logits,
        }

    @torch.no_grad()
    def infer(self,
              x: torch.Tensor,
              x_lengths: torch.Tensor,
              tones: Optional[torch.Tensor] = None,
              sid: Optional[torch.Tensor] = None,
              noise_scale: float = 0.667,
              length_scale: float = 1.0,
              noise_scale_w: float = 0.8,
              max_len: Optional[int] = None
              ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Inference: text → waveform.
        
        Args:
            x: (B, T_text) phoneme IDs
            x_lengths: (B,) lengths
            tones: (B, T_text) tone IDs (optional — predicted if not given)
            sid: (B,) speaker IDs (optional — for multi-speaker)
            noise_scale: Noise scale for sampling from prior (controls expressiveness)
            length_scale: Duration scaling (>1 = slower, <1 = faster)
            noise_scale_w: Noise scale for stochastic duration predictor
            max_len: Maximum output length (optional)
        
        Returns:
            waveform: (B, 1, T_audio)
            attn: alignment
            durations: predicted durations
        """
        # --- Speaker conditioning ---
        g = None
        if self.n_speakers > 0 and sid is not None:
            g = self.emb_g(sid).unsqueeze(-1)

        # --- Predict tones if not provided ---
        if tones is None and self.use_tone_predictor:
            tones = self.tone_predictor.predict(x, x_lengths)

        # --- Text encoding ---
        x_enc, m_p, logs_p, x_mask = self.enc_p(x, x_lengths, tones)

        # --- Prosody prediction from text (no reference audio at inference) ---
        if self.use_prosody_encoder:
            p_emb = self.prosody_predictor(x_enc, x_mask)  # (B, prosody_dim, 1)
            p_proj = self.prosody_proj(p_emb.squeeze(-1)).unsqueeze(-1)
            g = g + p_proj if g is not None else p_proj

        # --- Duration prediction ---
        if self.use_sdp:
            logw = self.dp(x_enc, x_mask, g=g, reverse=True, noise_scale=noise_scale_w)
        else:
            logw = self.dp(x_enc, x_mask, g=g)

        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = sequence_mask(y_lengths, None).unsqueeze(1).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)

        # --- Expand prior using predicted durations ---
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        # --- Sample from prior & reverse flow ---
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)

        # --- Decode to waveform ---
        waveform = self.dec(z * y_mask, g=g)

        return waveform, attn, w_ceil

