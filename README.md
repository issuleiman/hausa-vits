<div align="center">

# 🎙️ HausaVITS

### A Natural Text-to-Speech Framework for the Hausa Language

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/issuleiman/hausa-vits?style=social)](https://github.com/issuleiman/hausa-vits)

**HausaVITS** is the first comprehensive, linguistically-aware Text-to-Speech training framework designed specifically for the **Hausa language** — spoken by over **80 million people** across West Africa.

Built on top of the **VITS** (Variational Inference with adversarial learning for end-to-end TTS) architecture, this framework introduces targeted innovations that address the unique phonological, tonal, and prosodic complexity of Hausa — producing natural, expressive, and intelligible speech.

[**Quickstart**](#-quickstart) • [**Architecture**](#-architecture) • [**Innovations**](#-innovations-over-standard-vits) • [**Data Format**](#-data-preparation) • [**Training**](#-training) • [**Inference**](#-inference)

</div>

---

## 🌍 Why Hausa Needs a Dedicated TTS Framework

Hausa is not just another language — it is one of the most phonologically complex languages in Africa. Standard TTS systems (including generic VITS) fail on Hausa because they ignore:

| Challenge | What Standard TTS Does | What HausaVITS Does |
|-----------|------------------------|----------------------|
| **Tonal system** (H/L/F tones change word meaning) | Ignores tones entirely | Dedicated tone embedding stream + tone predictor |
| **Implosive consonants** (ɓ, ɗ) | Maps to wrong phonemes | Full inventory of 51 Hausa phonemes |
| **Ejective consonants** (ƙ, tsʼ) | Not supported | Native support with correct IPA mapping |
| **Phonemic vowel length** (a vs aː) | Treats all vowels equally | Separate short/long vowel symbols |
| **Pitch declination (downdrift)** | Not modeled | Prosody encoder captures utterance-level F0 patterns |
| **Ajami script (Arabic-based)** | No support | Built-in Ajami → Boko transliteration |
| **Hausa number/date reading** | Falls back to English | Full Hausa numeral expansion (ɗaya...dubu) |

---

## 🚀 Innovations Over Standard VITS

### 1. 🎵 Parallel Tone Embedding Stream
The biggest innovation. Hausa is a **tonal language** — the same word pronounced with different pitch means something completely different (e.g., *fita* H-L = "go out" vs H-H = "whistle"). Standard VITS has no concept of tones.

HausaVITS adds a **dedicated tone embedding** that runs in parallel with the phoneme embedding and is summed before the Transformer encoder:

```
Phoneme IDs  →  Phoneme Embedding (192d)  ─┐
                                             ⊕  →  Transformer Encoder
Tone IDs     →  Tone Embedding (32d)      ─┘
                  └→ Linear Projection (32d → 192d)
```

This allows the model to learn tone-conditioned acoustic features — the pitch of each phoneme is directly influenced by whether it carries a High (H), Low (L), or Falling (HL) tone.

---

### 2. 🧠 Neural Tone Predictor
Standard Hausa writing (**Boko script**) does **not** mark tones. This means a sentence like *"ya zo"* is ambiguous without context. For a real-world TTS system, you cannot assume tone-annotated input.

HausaVITS includes a **3-layer Transformer Tone Predictor** that:
- Reads the phoneme sequence
- Predicts per-phoneme tone (H/L/F) using contextual information
- Can be trained jointly with the TTS model or pre-trained on annotated corpora
- Falls back gracefully when tone marks **are** provided in the input

```python
# Works with unmarked Boko text (tone predicted automatically)
audio = synth.synthesize("Barka da safe")

# Works with tone-marked text (tones used directly, higher quality)
audio = synth.synthesize("Bàrkà dà sàfe")
```

---

### 3. 🎭 Hausa Prosody Encoder
Hausa has distinctive sentence-level prosody patterns that no phoneme-level model can capture:

- **Downdrift**: High tones progressively lower in pitch across a sentence
- **Phrase-boundary reset**: Pitch snaps back up at clause boundaries
- **Question intonation**: Yes/No questions suspend downdrift with a rising final contour

The **Reference Encoder** (Conv2D → GRU → Linear) extracts a 64-dim prosody embedding from a reference mel spectrogram during training. At inference, a **Prosody Predictor** generates this embedding from the text encoder hidden states — no reference audio required.

---

### 4. 🔤 Complete Hausa Phoneme Inventory (51 symbols)
Purpose-built for Hausa, going far beyond standard Latin phoneme sets:

| Category | Symbols |
|----------|---------|
| **Short vowels** | a, e, i, o, u |
| **Long vowels** | aː, eː, iː, oː, uː |
| **Diphthongs** | ai, au |
| **Implosives** | ɓ, ɗ |
| **Ejective** | ƙ, tsʼ |
| **Labialized** | kʷ, gʷ |
| **Palatalized** | kʲ, gʲ |
| **Retroflex flap** | ɽ |
| **Affricates** | dʒ, tʃ |
| **Glottal stop** | ʔ (auto-inserted word-initially) |
| **Tone markers** | `<H>`, `<L>`, `<F>`, `<TONE_PAD>` |

---

### 5. 📜 Ajami + Boko Dual Script Support
HausaVITS accepts both:
- **Boko** (Latin-based, standard): `"Ina kwana"`
- **Ajami** (Arabic-based): automatically detected and transliterated to Boko before G2P

```python
synth.synthesize("ina kwana")           # Boko
synth.synthesize("اِنا كوانا")           # Ajami (auto-converted)
```

---

### 6. 🔢 Native Hausa Number Expansion
Numbers are expanded into grammatically correct Hausa words:

```
1      → ɗaya
10     → goma
100    → ɗari
123    → ɗari da ashirin da uku
5000   → dubu biyar
```

---

### 7. 🎛️ Stochastic Duration Predictor
Unlike deterministic duration models that produce robotic, fixed-rhythm speech, HausaVITS uses a **flow-based Stochastic Duration Predictor**. This models duration as a *distribution* rather than a point estimate — capturing the natural variability in how fast or slow different speakers say the same phoneme. This is especially important for Hausa's **syllable-timed rhythm**.

---

## 📐 Architecture

```
                    ┌─────────────────────────────────────┐
                    │         HausaVITS Model              │
                    │                                      │
  Hausa Text ──►  Text Processing  ──►  Phoneme IDs       │
                    │                   Tone IDs           │
                    │                       │              │
                    │            ┌──────────▼──────────┐   │
                    │            │    Text Encoder      │   │
                    │            │  (Phoneme Emb        │   │
                    │            │   + Tone Emb         │   │
                    │            │   → Transformer)     │   │
                    │            └──────────┬───────────┘   │
                    │                       │               │
                    │         ┌─────────────▼─────────────┐ │
                    │         │   Prior Distribution       │ │
                    │         │   μ_p, σ_p  (per frame)   │ │
                    │         └─────────────┬─────────────┘ │
                    │                       │               │
  Reference Mel ─►  Prosody Encoder         │               │
  (training only)        │                 │               │
                    │    ▼                 │               │
                    │  Prosody Emb ──► Conditioning g      │
                    │                       │               │
  Linear Spec ──►  Posterior Encoder ──► z (latent)        │
  (training only)   │         │                            │
                    │         ▼                            │
                    │    Norm. Flows ◄──── z_p             │
                    │    (4 coupling                       │
                    │     layers)                          │
                    │         │                            │
                    │         ▼                            │
                    │  MAS Alignment (training)            │
                    │  Duration Predictor                  │
                    │         │                            │
                    │         ▼                            │
                    │  HiFi-GAN Decoder ──► 🔊 Waveform   │
                    └─────────────────────────────────────┘
```

### Components

| Module | Description |
|--------|-------------|
| `TextEncoder` | 6-layer Transformer with phoneme + tone dual embedding |
| `TonePredictor` | 3-layer Transformer predicting H/L/F per phoneme |
| `ProsodyEncoder` | Conv2D + GRU reference encoder for utterance prosody |
| `ProsodyPredictor` | Conv1D + pooling for inference-time prosody estimation |
| `PosteriorEncoder` | WaveNet (16-layer dilated conv) on linear spectrogram |
| `ResidualCouplingBlock` | 4-layer normalizing flow with affine coupling |
| `StochasticDurationPredictor` | Flow-based duration distribution model |
| `Generator` | HiFi-GAN v1 with multi-receptive field fusion |
| `MultiPeriodDiscriminator` | 5 sub-discriminators at periods [2,3,5,7,11] |
| `MultiScaleDiscriminator` | 3-scale waveform discriminator |
| `MAS` | Monotonic Alignment Search (Viterbi-based) |

---

## ⚡ Quickstart

### Install
```bash
# From GitHub (for Colab/Kaggle)
pip install git+https://github.com/issuleiman/hausa-vits.git

# For local development (editable)
git clone https://github.com/issuleiman/hausa-vits.git
cd hausa-vits
pip install -e .
```

### Synthesize (Python API)
```python
from hausa_tts import HausaSynthesizer

synth = HausaSynthesizer.from_pretrained("path/to/checkpoint/")

# Basic synthesis
audio = synth.synthesize("Sannu da zuwa")
synth.save_wav(audio, "output.wav")

# Control speaking style
audio = synth.synthesize(
    "Barka da safe",
    noise_scale=0.667,    # expressiveness (0=monotone, 1=expressive)
    length_scale=1.0,     # speed (>1 slower, <1 faster)
    noise_scale_w=0.8,    # duration variation
)

# Multi-speaker
audio = synth.synthesize("Ina kwana", speaker_id=3)

# Play in Colab/Jupyter
from IPython.display import Audio
Audio(audio.numpy(), rate=22050)
```

---

## 📦 Data Preparation

HausaVITS uses **Common Voice style CSV format**:

```csv
audio,text,speaker_id,gender
wavs/clip001.wav,Ina son kasuwar garin nan,spk_001,M
wavs/clip002.wav,Yaro ya tafi makaranta yau,spk_002,F
wavs/clip003.wav,Ruwan sama ya fadi jiya dare,spk_001,M
```

| Column | Description |
|--------|-------------|
| `audio` | Path to `.wav` file (relative to `audio_dir`) |
| `text` | Hausa transcript (Boko or Ajami, with or without tone marks) |
| `speaker_id` | Speaker identifier string (used for multi-speaker embedding) |
| `gender` | `M` or `F` (used for analysis; optional for training) |

### Recommended Audio Specs
- Format: **WAV, mono**
- Sample rate: **22050 Hz** (or 16000 Hz — both supported)
- Duration: **2–15 seconds** per clip (5–10s optimal)
- Clean speech, minimal background noise

### Preprocess
```bash
python scripts/preprocess.py \
  --metadata data/train.csv \
  --audio_dir data/wavs/ \
  --output_dir data/processed/ \
  --sample_rate 22050
```

---

## 🏋️ Training

### Single-Speaker (Single GPU)
```bash
python scripts/train.py --config configs/single_speaker.yaml
```

### Multi-Speaker (Single GPU)
```bash
python scripts/train.py --config configs/multi_speaker.yaml
```

### Multi-GPU (DDP — recommended for faster training)
```bash
torchrun --nproc_per_node=4 scripts/train.py --config configs/multi_speaker.yaml
```

### Google Colab / Kaggle
```python
# Mount Drive for checkpoint saving
from google.colab import drive
drive.mount('/content/drive')

# Install framework
!pip install git+https://github.com/issuleiman/hausa-vits.git

# Clone for config access
!git clone https://github.com/issuleiman/hausa-vits.git
%cd hausa-vits

# Edit config then train
import yaml
with open("configs/single_speaker.yaml") as f:
    config = yaml.safe_load(f)

config["data"]["metadata_path"] = "/content/data/metadata.csv"
config["data"]["audio_dir"] = "/content/data/wavs/"

from hausa_tts.training import Trainer
trainer = Trainer(config)
trainer.train()
```

### Key Hyperparameters (`configs/base.yaml`)
```yaml
audio:
  sample_rate: 22050
  n_mels: 80

model:
  hidden_channels: 192   # Transformer hidden dim
  n_layers: 6            # Transformer layers
  use_sdp: true          # Stochastic duration predictor

tone:
  n_tones: 4             # TONE_PAD, H, L, F
  tone_embedding_dim: 32
  use_tone_predictor: true

prosody:
  use_prosody_encoder: true
  prosody_embedding_dim: 64

training:
  batch_size: 32
  learning_rate_g: 2.0e-4
  fp16: true             # Mixed precision
  epochs: 1000
```

---

## 🔊 Inference

### CLI
```bash
python scripts/synthesize.py \
  --checkpoint checkpoints/model.pt \
  --config configs/base.yaml \
  --text "Sannu da zuwa" \
  --output output.wav \
  --speaker_id 0
```

### Python API
```python
from hausa_tts import HausaSynthesizer

synth = HausaSynthesizer.from_pretrained("checkpoints/")
audio = synth.synthesize("Yaro ya tafi makaranta")
synth.save_wav(audio, "output.wav")
```

---

## 📁 Project Structure

```
hausa-vits/
├── hausa_tts/
│   ├── text/
│   │   ├── symbols.py          # 51-symbol Hausa phoneme inventory
│   │   ├── cleaners.py         # Text normalization + Ajami support
│   │   ├── hausa_phonemizer.py # G2P with tone parsing
│   │   └── tokenizer.py        # Text → tensor pipeline
│   ├── model/
│   │   ├── vits.py             # Main HausaVITS model
│   │   ├── text_encoder.py     # Transformer + tone embeddings
│   │   ├── tone_predictor.py   # Neural tone prediction
│   │   ├── prosody_encoder.py  # Prosody reference encoder
│   │   ├── posterior_encoder.py
│   │   ├── flow.py             # Normalizing flows
│   │   ├── decoder.py          # HiFi-GAN decoder
│   │   ├── duration_predictor.py
│   │   ├── discriminator.py    # MPD + MSD
│   │   ├── mas.py              # Monotonic Alignment Search
│   │   ├── modules.py          # WN, ResBlocks, splines
│   │   └── losses.py           # All loss functions
│   ├── data/
│   │   ├── dataset.py          # Common Voice CSV loader
│   │   ├── audio.py            # Mel/linear spectrogram
│   │   ├── collate.py          # Batch padding
│   │   └── preprocess.py
│   ├── training/
│   │   ├── trainer.py          # DDP + AMP training loop
│   │   └── scheduler.py
│   └── inference/
│       └── synthesize.py       # End-to-end synthesis
├── configs/
│   ├── base.yaml
│   ├── single_speaker.yaml
│   └── multi_speaker.yaml
├── scripts/
│   ├── preprocess.py
│   ├── train.py
│   └── synthesize.py
└── tests/
```

---

## 📊 Loss Functions

| Loss | Weight | Purpose |
|------|--------|---------|
| Mel Reconstruction (L1) | 45 | Acoustic fidelity |
| KL Divergence | 1 | Align prior & posterior |
| Generator Adversarial | 1 | Naturalness via GAN |
| Feature Matching | 2 | Fine-grained waveform detail |
| Duration (NLL) | 1 | Phoneme timing |
| **Tone Prediction (CE)** | 1 | **Hausa-specific: correct tones** |

---

## 🛠️ Requirements

```
torch>=2.0.0
torchaudio>=2.0.0
numpy>=1.21.0
scipy>=1.7.0
librosa>=0.9.0
tensorboard>=2.10.0
PyYAML>=6.0
tqdm>=4.60.0
soundfile>=0.10.0
```

---

## 📜 Citation

If you use HausaVITS in your research, please cite:

```bibtex
@software{hausavits2024,
  author    = {Ismail Suleman},
  title     = {HausaVITS: A Natural Text-to-Speech Framework for the Hausa Language},
  year      = {2024},
  url       = {https://github.com/issuleiman/hausa-vits},
}
```

---

## 🙏 Acknowledgements

- [**VITS**](https://arxiv.org/abs/2106.06103) — Kim et al., 2021 — the foundation architecture
- [**HiFi-GAN**](https://arxiv.org/abs/2010.05646) — Kong et al., 2020 — the vocoder backbone
- The Hausa linguistic community and open speech datasets that make this work possible

---

<div align="center">

**Built with ❤️ for the Hausa-speaking world**

[⭐ Star this repo](https://github.com/issuleiman/hausa-vits) • [🐛 Report an issue](https://github.com/issuleiman/hausa-vits/issues) • [🤝 Contribute](https://github.com/issuleiman/hausa-vits/pulls)

</div>
