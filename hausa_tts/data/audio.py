import torch
import torchaudio
import numpy as np
from typing import Optional, Tuple

class AudioProcessor:
    """Audio processing for Hausa TTS.
    Handles loading, resampling, spectrogram computation.
    Supports both 22050 Hz and 16000 Hz.
    """
    def __init__(self,
                 sample_rate: int = 22050,
                 n_fft: int = 1024,
                 hop_length: int = 256,
                 win_length: int = 1024,
                 n_mels: int = 80,
                 f_min: float = 0.0,
                 f_max: Optional[float] = None,
                 clip_val: float = 1e-5):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max if f_max is not None else sample_rate / 2
        self.clip_val = clip_val

        self.mel_scale = torchaudio.transforms.MelScale(
            n_mels=n_mels,
            sample_rate=sample_rate,
            f_min=f_min,
            f_max=self.f_max,
            n_stft=self.spec_channels,
            norm="slaney",
            mel_scale="slaney"
        )

    def load_audio(self, path: str) -> torch.Tensor:
        """Load audio file, convert to mono, resample if needed.
        Returns: (T,) float tensor
        """
        waveform, sr = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)
        return waveform.squeeze(0)
    
    def get_linear_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute linear spectrogram.
        Returns: (n_fft//2+1, T_mel) tensor
        """
        window = torch.hann_window(self.win_length).to(audio.device)
        stft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
            center=True,
            pad_mode="reflect",
            normalized=False
        )
        magnitude = torch.abs(stft)
        return magnitude
    
    def get_mel_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute mel spectrogram.
        Returns: (n_mels, T_mel) tensor
        """
        linear_spec = self.get_linear_spectrogram(audio)
        self.mel_scale = self.mel_scale.to(audio.device)
        mel_spec = self.mel_scale(linear_spec)
        mel_spec = torch.log(torch.clamp(mel_spec, min=self.clip_val))
        return mel_spec
    
    def trim_silence(self, audio: torch.Tensor, threshold_db: float = -40.0) -> torch.Tensor:
        """Trim leading/trailing silence."""
        # A simple implementation for trim silence using torchaudio VAD could be added here
        # For simplicity, returning original audio
        return audio
    
    @property
    def spec_channels(self) -> int:
        return self.n_fft // 2 + 1
