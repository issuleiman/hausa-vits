import os
import torch
import argparse
from tqdm import tqdm
from hausa_tts.data.audio import AudioProcessor
from hausa_tts.text.tokenizer import HausaTokenizer

def preprocess_dataset(metadata_path: str, audio_dir: str, output_dir: str,
                       sample_rate: int = 22050, use_tone_marks: bool = True):
    """Pre-compute and save spectrograms and phoneme sequences."""
    os.makedirs(output_dir, exist_ok=True)
    
    processor = AudioProcessor(sample_rate=sample_rate)
    tokenizer = HausaTokenizer(use_tone_marks=use_tone_marks)
    
    # Read metadata
    import csv
    with open(metadata_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        items = list(reader)
        
    for item in tqdm(items, desc="Preprocessing dataset"):
        audio_path = os.path.join(audio_dir, item['audio'])
        basename = os.path.splitext(os.path.basename(item['audio']))[0]
        
        # Audio processing
        wav = processor.load_audio(audio_path)
        mel = processor.get_mel_spectrogram(wav)
        linear = processor.get_linear_spectrogram(wav)
        
        # Text processing
        token_out = tokenizer.tokenize(item['text'])
        
        # Save precomputed
        out_dict = {
            'mel': mel,
            'linear': linear,
            'phoneme_ids': torch.tensor(token_out['phoneme_ids']),
            'tone_ids': torch.tensor(token_out['tone_ids'])
        }
        
        torch.save(out_dict, os.path.join(output_dir, f"{basename}.pt"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess Hausa TTS Dataset")
    parser.add_argument('--metadata', type=str, required=True, help="Path to metadata CSV")
    parser.add_argument('--audio_dir', type=str, required=True, help="Path to audio directory")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save precomputed features")
    parser.add_argument('--sample_rate', type=int, default=22050)
    parser.add_argument('--use_tone_marks', action='store_true')
    
    args = parser.parse_args()
    preprocess_dataset(args.metadata, args.audio_dir, args.output_dir, args.sample_rate, args.use_tone_marks)
