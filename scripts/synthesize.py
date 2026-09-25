#!/usr/bin/env python3
import sys
import os
import argparse
import yaml
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hausa_tts.inference.synthesize import HausaSynthesizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True, help='Directory containing checkpoint and config')
    parser.add_argument('--text', type=str, help='Text to synthesize')
    parser.add_argument('--text_file', type=str, help='File containing texts to synthesize (one per line)')
    parser.add_argument('--output', type=str, required=True, help='Output WAV path or directory')
    parser.add_argument('--speaker_id', type=int, default=None, help='Speaker ID for multi-speaker model')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    synth = HausaSynthesizer.from_pretrained(args.model_dir, device=device)
    
    texts = []
    if args.text:
        texts.append((args.text, args.output))
    elif args.text_file:
        os.makedirs(args.output, exist_ok=True)
        with open(args.text_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line:
                    texts.append((line, os.path.join(args.output, f'sample_{i}.wav')))
                    
    for text, out_path in texts:
        print(f"Synthesizing: {text}")
        audio = synth.synthesize(text, speaker_id=args.speaker_id)
        synth.save_wav(audio, out_path)
        print(f"Saved to: {out_path}")

if __name__ == '__main__':
    main()
