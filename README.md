# HausaVITS

A comprehensive text-to-speech framework for the Hausa language, based on VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech).

## Features
- End-to-end TTS architecture based on VITS
- Dedicated Hausa phonemizer and tone handling
- Multi-speaker support
- Fast inference
- PyTorch DDP training support
- Mixed precision training

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Data Preparation
Data should be in a Common Voice style format with `audio,text,speaker_id,gender` columns.

```
audio,text,speaker_id,gender
path/to/audio1.wav,Ina kwana,spk_001,M
path/to/audio2.wav,Lafiya lau,spk_002,F
```

## Preprocessing
Pre-compute spectrograms and phoneme sequences to speed up training:

```bash
python scripts/preprocess.py --metadata data/train.csv --audio_dir data/wavs --output_dir data/processed
```

## Training

**Single GPU:**
```bash
python scripts/train.py --config configs/base.yaml --log_dir logs/hausa_base
```

**Multi-GPU (DDP):**
```bash
torchrun --nproc_per_node=4 scripts/train.py --config configs/base.yaml --log_dir logs/hausa_base
```

## Inference

**CLI:**
```bash
python scripts/synthesize.py --model_dir logs/hausa_base --text "Sannu da zuwa" --output result.wav
```

**Python API:**
```python
from hausa_tts import HausaSynthesizer

synth = HausaSynthesizer.from_pretrained('logs/hausa_base')
audio = synth.synthesize("Sannu da zuwa")
synth.save_wav(audio, "output.wav")
```

## Architecture Overview
The framework implements a Hausa-specific G2P (Grapheme-to-Phoneme) and tone handling system alongside the VITS architecture, consisting of a Posterior Encoder, Prior Encoder, Decoder, Stochastic Duration Predictor, and Discriminators.

## License
MIT
