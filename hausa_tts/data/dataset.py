import torch
from torch.utils.data import Dataset
import csv
import os
from typing import Optional, Dict, List
from hausa_tts.data.audio import AudioProcessor
from hausa_tts.text.tokenizer import HausaTokenizer

class HausaTTSDataset(Dataset):
    """Dataset for Hausa TTS training.
    
    Expected data format (CSV with header):
    audio,text,speaker_id,gender
    path/to/audio1.wav,Hausa text here,spk_001,M
    path/to/audio2.wav,Another text,spk_002,F
    """
    def __init__(self,
                 metadata_path: str,
                 audio_dir: str,
                 audio_processor: AudioProcessor,
                 tokenizer: HausaTokenizer,
                 max_audio_len: Optional[int] = None,
                 min_audio_len: int = 0,
                 use_tone_marks: bool = True,
                 precomputed_dir: Optional[str] = None):
        self.audio_dir = audio_dir
        self.audio_processor = audio_processor
        self.tokenizer = tokenizer
        self.max_audio_len = max_audio_len
        self.min_audio_len = min_audio_len
        self.use_tone_marks = use_tone_marks
        self.precomputed_dir = precomputed_dir
        
        self.items = []
        unique_speakers = set()
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.items.append(row)
                unique_speakers.add(row['speaker_id'])
                
        # Build speaker map
        self._speaker_map = {spk: idx for idx, spk in enumerate(sorted(unique_speakers))}
    
    def __len__(self) -> int:
        return len(self.items)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.items[idx]
        audio_path = os.path.join(self.audio_dir, row['audio'])
        text = row['text']
        speaker_id = self._speaker_map[row['speaker_id']]
        
        # Tokenize text
        token_output = self.tokenizer.tokenize(text)
        phoneme_ids = torch.tensor(token_output['phoneme_ids'], dtype=torch.long)
        tone_ids = torch.tensor(token_output['tone_ids'], dtype=torch.long)
        
        # Load audio
        wav = self.audio_processor.load_audio(audio_path)
        
        # Trim or pad logic can be applied here based on min/max len
        
        # Compute spectrograms
        linear_spec = self.audio_processor.get_linear_spectrogram(wav)
        mel = self.audio_processor.get_mel_spectrogram(wav)
        
        return {
            'phoneme_ids': phoneme_ids,
            'tone_ids': tone_ids,
            'mel': mel,
            'linear_spec': linear_spec,
            'wav': wav,
            'speaker_id': torch.tensor(speaker_id, dtype=torch.long),
            'text': text,
            'audio_path': audio_path
        }
    
    @property
    def n_speakers(self) -> int:
        return len(self._speaker_map)
    
    @property
    def speaker_map(self) -> Dict[str, int]:
        return self._speaker_map
