import torch
from typing import Dict, List

class HausaCollate:
    """Custom collate function for HausaTTSDataset.
    Pads all sequences in a batch to the maximum length.
    """
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Sort batch by text length descending
        batch.sort(key=lambda x: x['phoneme_ids'].shape[0], reverse=True)
        
        # Get max lengths
        max_text_len = max([x['phoneme_ids'].shape[0] for x in batch])
        max_mel_len = max([x['mel'].shape[1] for x in batch])
        max_wav_len = max([x['wav'].shape[0] for x in batch])
        
        B = len(batch)
        n_mels = batch[0]['mel'].shape[0]
        spec_channels = batch[0]['linear_spec'].shape[0]
        
        # Initialize padded tensors
        phoneme_ids_padded = torch.zeros((B, max_text_len), dtype=torch.long)
        tone_ids_padded = torch.zeros((B, max_text_len), dtype=torch.long)
        mel_padded = torch.zeros((B, n_mels, max_mel_len), dtype=torch.float)
        linear_spec_padded = torch.zeros((B, spec_channels, max_mel_len), dtype=torch.float)
        wav_padded = torch.zeros((B, 1, max_wav_len), dtype=torch.float)
        
        phoneme_lengths = torch.zeros((B,), dtype=torch.long)
        mel_lengths = torch.zeros((B,), dtype=torch.long)
        wav_lengths = torch.zeros((B,), dtype=torch.long)
        speaker_ids = torch.zeros((B,), dtype=torch.long)
        
        for i, item in enumerate(batch):
            # Text lengths
            t_len = item['phoneme_ids'].shape[0]
            phoneme_lengths[i] = t_len
            phoneme_ids_padded[i, :t_len] = item['phoneme_ids']
            tone_ids_padded[i, :t_len] = item['tone_ids']
            
            # Audio lengths
            m_len = item['mel'].shape[1]
            mel_lengths[i] = m_len
            mel_padded[i, :, :m_len] = item['mel']
            linear_spec_padded[i, :, :m_len] = item['linear_spec']
            
            w_len = item['wav'].shape[0]
            wav_lengths[i] = w_len
            wav_padded[i, 0, :w_len] = item['wav']
            
            # Speaker
            speaker_ids[i] = item['speaker_id']
            
        return {
            'phoneme_ids': phoneme_ids_padded,
            'phoneme_lengths': phoneme_lengths,
            'tone_ids': tone_ids_padded,
            'mel': mel_padded,
            'linear_spec': linear_spec_padded,
            'wav': wav_padded,
            'wav_lengths': wav_lengths,
            'speaker_ids': speaker_ids,
            'mel_lengths': mel_lengths
        }
