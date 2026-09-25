import torch
from typing import List, Tuple
from hausa_tts.text.symbols import SYMBOL_TO_ID, ID_TO_SYMBOL, TONE_TO_ID, ID_TO_TONE, n_symbols, n_tones
from hausa_tts.text.cleaners import hausa_cleaners
from hausa_tts.text.hausa_phonemizer import HausaPhonemizer

class HausaTokenizer:
    def __init__(self):
        """Initialize tokenizer using symbols from symbols.py"""
        self.phonemizer = HausaPhonemizer()
        
    def encode(self, phonemes: List[str], tones: List[str]) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """Convert phoneme and tone lists to integer tensors.
        Returns:
            phoneme_ids: LongTensor of shape (seq_len,)
            tone_ids: LongTensor of shape (seq_len,)
        """
        phoneme_ids = [SYMBOL_TO_ID.get(p, 0) for p in phonemes]
        tone_ids = [TONE_TO_ID.get(t, 0) for t in tones]
        
        return torch.LongTensor(phoneme_ids), torch.LongTensor(tone_ids)
    
    def decode(self, phoneme_ids: torch.LongTensor, tone_ids: torch.LongTensor) -> Tuple[List[str], List[str]]:
        """Convert integer tensors back to phoneme and tone lists."""
        phonemes = [ID_TO_SYMBOL.get(p.item(), '_') for p in phoneme_ids]
        tones = [ID_TO_TONE.get(t.item(), '<TONE_PAD>') for t in tone_ids]
        
        return phonemes, tones
    
    def text_to_sequence(self, text: str, use_tone_marks: bool = True) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """Full pipeline: raw text → cleaned → phonemized → encoded."""
        self.phonemizer.use_tone_marks = use_tone_marks
        cleaned_text = hausa_cleaners(text)
        phonemes, tones = self.phonemizer.phonemize(cleaned_text)
        return self.encode(phonemes, tones)
    
    @property
    def vocab_size(self) -> int:
        """Number of phoneme symbols."""
        return n_symbols
    
    @property
    def n_tones(self) -> int:
        """Number of tone symbols."""
        return n_tones
