#!/usr/8bin/env python3
import sys
import os

# Add root directory to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hausa_tts.data.preprocess import preprocess_dataset
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess Hausa TTS Dataset")
    parser.add_argument('--metadata', type=str, required=True, help="Path to metadata CSV")
    parser.add_argument('--audio_dir', type=str, required=True, help="Path to audio directory")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save precomputed features")
    parser.add_argument('--sample_rate', type=int, default=22050)
    parser.add_argument('--use_tone_marks', action='store_true', help="Use tone marks for TTS")
    
    args = parser.parse_args()
    preprocess_dataset(
        metadata_path=args.metadata,
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
        use_tone_marks=args.use_tone_marks
    )
