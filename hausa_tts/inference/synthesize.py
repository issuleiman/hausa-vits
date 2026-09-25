import torch
import torchaudio
import os
import yaml
from typing import Optional
from hausa_tts.model.vits import HausaVITS
from hausa_tts.text.tokenizer import HausaTokenizer
from hausa_tts.data.audio import AudioProcessor

class HausaSynthesizer:
    """End-to-end Hausa text-to-speech synthesis."""
    def __init__(self, checkpoint_path: str, config: dict, device: str = 'cpu'):
        self.device = device
        self.config = config
        
        self.model = HausaVITS(config).to(device)
        self.model.eval()
        
        ckpt = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)
        
        self.tokenizer = HausaTokenizer(use_tone_marks=config['data'].get('use_tone_marks', True))
    
    def synthesize(self, text: str, 
                   speaker_id: Optional[int] = None,
                   noise_scale: float = 0.667,
                   length_scale: float = 1.0,
                   noise_scale_w: float = 0.8,
                   use_tone_marks: bool = True) -> torch.Tensor:
        """Convert Hausa text to speech waveform.
        Returns: (T,) float tensor - audio waveform
        """
        tokens = self.tokenizer.tokenize(text)
        x = torch.tensor(tokens['phoneme_ids'], dtype=torch.long, device=self.device).unsqueeze(0)
        x_lengths = torch.tensor([x.shape[1]], dtype=torch.long, device=self.device)
        tones = torch.tensor(tokens['tone_ids'], dtype=torch.long, device=self.device).unsqueeze(0)
        
        sid = torch.tensor([speaker_id], dtype=torch.long, device=self.device) if speaker_id is not None else None
        
        with torch.no_grad():
            audio = self.model.infer(
                x, x_lengths, sid, tones,
                noise_scale=noise_scale,
                length_scale=length_scale,
                noise_scale_w=noise_scale_w
            )[0][0,0].cpu()
            
        return audio
    
    def save_wav(self, audio: torch.Tensor, path: str):
        """Save audio tensor to WAV file."""
        sr = self.config['audio']['sample_rate']
        torchaudio.save(path, audio.unsqueeze(0), sr)
    
    @classmethod
    def from_pretrained(cls, model_dir: str, device: str = 'cpu'):
        """Load synthesizer from a directory containing checkpoint + config."""
        config_path = os.path.join(model_dir, 'config.yaml')
        ckpt_path = os.path.join(model_dir, 'checkpoint.pth')
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        return cls(ckpt_path, config, device)
