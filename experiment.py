#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║        HausaVITS Research Experiment — Single Cell Script           ║
║  Baseline VITS vs HausaVITS on Hausa Speech (single + multi-spk)   ║
║  Supports cross-session resume via HuggingFace Hub                  ║
╚══════════════════════════════════════════════════════════════════════╝

HOW TO USE:
  1. Paste this entire file into a single Kaggle/Colab cell
  2. Click Run — it auto-installs everything, loads dataset,
     trains 4 models, evaluates, and pushes results to HuggingFace.
  3. If the session dies, just run the cell again — it resumes
     exactly where it stopped.
"""

# ══════════════════════════════════════════════════════════════════════
# ① USER CONFIGURATION — ONLY EDIT THIS SECTION
# ══════════════════════════════════════════════════════════════════════
HF_TOKEN           = "YOUR_HUGGINGFACE_TOKEN_HERE"   # ← paste your HF token
HF_USERNAME        = "suleiman2003"
HF_REPO_ID         = f"{HF_USERNAME}/hausa-vits-experiments"
HF_DATASET_ID      = f"{HF_USERNAME}/unified-hausa-speech"
GITHUB_FRAMEWORK   = "git+https://github.com/issuleiman/hausa-vits.git"

SINGLE_SPEAKER_ID  = "88"
MULTI_SPEAKER_IDS  = ["225", "40"]
MAX_HOURS          = 3.0          # hours of audio per speaker
MAX_TRAIN_STEPS    = 50_000       # steps per model (~14h on T4)
SAVE_EVERY         = 2_000        # checkpoint + HF push every N steps
EVAL_EVERY         = 10_000       # run eval metrics every N steps
N_TEST_SAMPLES     = 50           # held-out test utterances per speaker

# Audio settings
SAMPLE_RATE        = 22_050
N_FFT              = 1024
HOP_LENGTH         = 256
WIN_LENGTH         = 1024
N_MELS             = 80
SEGMENT_SIZE       = 8_192        # waveform frames per training slice

# Training settings (Kaggle T4 / Colab T4 safe)
BATCH_SIZE         = 8
GRAD_ACCUM         = 4            # effective batch = 32
LR_G               = 2e-4
LR_D               = 2e-4
LR_DECAY           = 0.999875

# Paths
import os
WORK_DIR = "/kaggle/working" if os.path.exists("/kaggle") else "/content"


# ══════════════════════════════════════════════════════════════════════
# ② INSTALL DEPENDENCIES
# ══════════════════════════════════════════════════════════════════════
import subprocess, sys, importlib

def _pip(*pkgs):
    """Install packages quietly, skip if already installed."""
    to_install = []
    for p in pkgs:
        name = p.split(">=")[0].split("==")[0].split("git+")[0]
        name = name.split("/")[-1].replace("-", "_").lower()
        if importlib.util.find_spec(name) is None:
            to_install.append(p)
    if to_install:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q"] + to_install
        )

print("📦  Checking / installing dependencies …")
_pip(GITHUB_FRAMEWORK)
_pip("datasets>=2.14.0", "huggingface_hub>=0.20.0")
_pip("speechmos")
_pip("openai-whisper")
_pip("librosa>=0.10.0", "soundfile", "audioread")
_pip("matplotlib", "pandas", "scipy", "tqdm", "pyyaml")
print("✅  Dependencies ready.\n")


# ══════════════════════════════════════════════════════════════════════
# ③ IMPORTS
# ══════════════════════════════════════════════════════════════════════
import gc, json, time, math, shutil, warnings, traceback
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import load_dataset, Audio as HFAudio
from huggingface_hub import HfApi, hf_hub_download, upload_file

from hausa_tts.model.vits import HausaVITS
from hausa_tts.model.discriminator import (
    MultiPeriodDiscriminator, MultiScaleDiscriminator
)
from hausa_tts.model.losses import (
    generator_loss, discriminator_loss, feature_loss,
    kl_loss, mel_reconstruction_loss, tone_prediction_loss
)
from hausa_tts.data.dataset import HausaTTSDataset
from hausa_tts.data.audio import AudioProcessor
from hausa_tts.data.collate import HausaCollate
from hausa_tts.text.tokenizer import HausaTokenizer
from hausa_tts.text.symbols import n_symbols, n_tones
from hausa_tts.training.scheduler import get_scheduler

warnings.filterwarnings("ignore")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️   Device : {DEVICE}")
if DEVICE == "cuda":
    props = torch.cuda.get_device_properties(0)
    print(f"     GPU   : {props.name}")
    print(f"     VRAM  : {props.total_memory/1e9:.1f} GB")


# ══════════════════════════════════════════════════════════════════════
# ④ HuggingFace STATE MANAGER
# ══════════════════════════════════════════════════════════════════════
class StateManager:
    """
    Persists experiment progress on HuggingFace Hub.
    On every Kaggle session start the notebook reads state.json
    and continues from the last saved checkpoint.
    """
    EXPS = [
        "single_baseline",
        "single_hausa",
        "multi_baseline",
        "multi_hausa",
    ]

    def __init__(self):
        self.api = HfApi(token=HF_TOKEN)
        self._ensure_repo()
        self.state = self._load_or_init()

    def _ensure_repo(self):
        try:
            self.api.create_repo(
                HF_REPO_ID, repo_type="model",
                private=False, exist_ok=True
            )
            print(f"📂  HF Repo  : https://huggingface.co/{HF_REPO_ID}")
        except Exception as e:
            print(f"⚠️   Repo: {e}")

    def _default(self) -> dict:
        return {
            "experiments": {
                exp: {
                    "status": "pending",
                    "current_step": 0,
                    "best_utmos": 0.0,
                    "best_mcd": 999.0,
                }
                for exp in self.EXPS
            },
            "evaluation_done": False,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }

    def _load_or_init(self) -> dict:
        try:
            path = hf_hub_download(
                HF_REPO_ID, "state.json", token=HF_TOKEN
            )
            with open(path) as f:
                state = json.load(f)
            print("🔄  Resuming from saved state:")
            for exp, info in state["experiments"].items():
                st = info["status"]
                step = info["current_step"]
                sym = "✅" if st == "completed" else ("🔄" if st == "running" else "⏳")
                print(f"     {sym} {exp:20s}  step={step:>6d}  [{st}]")
            print()
            return state
        except Exception:
            print("🆕  No previous state. Starting fresh.\n")
            return self._default()

    def save(self):
        self.state["last_updated"] = datetime.now(timezone.utc).isoformat()
        local = Path(WORK_DIR) / "state.json"
        with open(local, "w") as f:
            json.dump(self.state, f, indent=2)
        try:
            self.api.upload_file(
                path_or_fileobj=str(local),
                path_in_repo="state.json",
                repo_id=HF_REPO_ID, token=HF_TOKEN,
            )
        except Exception as e:
            print(f"⚠️   state.json push: {e}")

    def update(self, exp: str, step: int, **metrics):
        self.state["experiments"][exp]["current_step"] = step
        self.state["experiments"][exp]["status"] = "running"
        for k, v in metrics.items():
            self.state["experiments"][exp][k] = v
        self.save()

    def complete(self, exp: str):
        self.state["experiments"][exp]["status"] = "completed"
        self.save()

    def is_done(self, exp: str) -> bool:
        return self.state["experiments"][exp]["status"] == "completed"

    def get_step(self, exp: str) -> int:
        return self.state["experiments"][exp]["current_step"]

    def push_file(self, local_path: str, repo_path: str):
        try:
            self.api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=repo_path,
                repo_id=HF_REPO_ID, token=HF_TOKEN,
            )
        except Exception as e:
            print(f"⚠️   push {repo_path}: {e}")

    def pull_checkpoint(self, exp: str) -> Optional[str]:
        step = self.get_step(exp)
        if step == 0:
            return None
        repo_path = f"{exp}/checkpoint_{step}.pt"
        try:
            local = hf_hub_download(
                HF_REPO_ID, repo_path, token=HF_TOKEN
            )
            print(f"⬇️   Pulled {exp} checkpoint (step {step})")
            return local
        except Exception:
            return None


# ══════════════════════════════════════════════════════════════════════
# ⑤ DATASET PREPARATION
# ══════════════════════════════════════════════════════════════════════
class DataPreparer:
    """
    Loads suleiman2003/unified-hausa-speech from HuggingFace,
    filters by speaker, limits to MAX_HOURS, saves WAVs to disk,
    and writes metadata CSV in HausaVITS format.
    """

    def __init__(self, work_dir: str = WORK_DIR):
        self.work_dir = Path(work_dir)
        self.audio_processor = AudioProcessor(
            sample_rate=SAMPLE_RATE, n_fft=N_FFT,
            hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
            n_mels=N_MELS,
        )

    def _load_hf_dataset(self) -> object:
        print(f"⬇️   Loading dataset {HF_DATASET_ID} …")
        ds = load_dataset(
            HF_DATASET_ID,
            token=HF_TOKEN,
            trust_remote_code=True,
        )
        # Handle both dict and DatasetDict
        if hasattr(ds, "keys"):
            # Merge all splits
            from datasets import concatenate_datasets
            splits = list(ds.values())
            ds = concatenate_datasets(splits)
        ds = ds.cast_column("audio", HFAudio(sampling_rate=SAMPLE_RATE))
        print(f"✅  Dataset loaded: {len(ds):,} items")
        return ds

    def _detect_columns(self, ds) -> Tuple[str, str, str]:
        """Auto-detect column names for text, speaker, gender."""
        cols = ds.column_names
        text_col    = next((c for c in cols if c in ("sentence","text","transcription","transcript")), None)
        speaker_col = next((c for c in cols if c in ("client_id","speaker_id","speaker","spk_id")), None)
        gender_col  = next((c for c in cols if c in ("gender","sex")), "gender")
        assert text_col,    f"No text column in {cols}"
        assert speaker_col, f"No speaker column in {cols}"
        return text_col, speaker_col, gender_col

    def prepare(
        self,
        speaker_ids: List[str],
        tag: str,
        max_hours: float = MAX_HOURS,
        n_test: int = N_TEST_SAMPLES,
    ) -> Tuple[Path, Path, Path]:
        """
        Returns (train_csv, test_csv, audio_dir).
        Re-uses cached data if already prepared.
        """
        out_dir   = self.work_dir / "data" / tag
        audio_dir = out_dir / "wavs"
        train_csv = out_dir / "train.csv"
        test_csv  = out_dir / "test.csv"

        if train_csv.exists() and test_csv.exists():
            print(f"📁  Cached data found for [{tag}] — skipping download.")
            return train_csv, test_csv, audio_dir

        out_dir.mkdir(parents=True, exist_ok=True)
        audio_dir.mkdir(parents=True, exist_ok=True)

        ds = self._load_hf_dataset()
        text_col, spk_col, gen_col = self._detect_columns(ds)

        # Filter by speakers
        print(f"🔍  Filtering speakers {speaker_ids} …")
        ds = ds.filter(
            lambda x: str(x[spk_col]) in speaker_ids,
            num_proc=1,
        )
        print(f"    {len(ds):,} items after filter")

        # Limit to max_hours per speaker
        max_sec    = max_hours * 3600
        records    = []
        sec_counts = {s: 0.0 for s in speaker_ids}

        for item in tqdm(ds, desc="📥 Downloading audio"):
            spk = str(item[spk_col])
            if sec_counts.get(spk, 0) >= max_sec:
                continue

            audio_arr  = np.array(item["audio"]["array"], dtype=np.float32)
            sr_src     = item["audio"]["sampling_rate"]
            dur        = len(audio_arr) / sr_src

            # Resample if needed
            if sr_src != SAMPLE_RATE:
                audio_arr = librosa.resample(
                    audio_arr, orig_sr=sr_src, target_sr=SAMPLE_RATE
                )

            fname  = f"{spk}_{len(records):06d}.wav"
            fpath  = audio_dir / fname
            sf.write(str(fpath), audio_arr, SAMPLE_RATE, subtype="PCM_16")

            text   = str(item.get(text_col, "")).strip()
            gender = str(item.get(gen_col, "M"))
            if not text:
                continue

            records.append({
                "audio"      : fname,
                "text"       : text,
                "speaker_id" : spk,
                "gender"     : gender,
                "duration"   : dur,
            })
            sec_counts[spk] = sec_counts.get(spk, 0) + dur

        df = pd.DataFrame(records)
        print(f"✅  {len(df):,} usable samples  |  "
              + "  ".join(f"spk{s}={sec_counts[s]/3600:.2f}h" for s in speaker_ids))

        # Train / test split (stratified by speaker)
        test_rows  = []
        train_rows = []
        for spk in speaker_ids:
            sub = df[df["speaker_id"] == spk].sample(
                frac=1, random_state=42
            )
            n = min(n_test, len(sub) // 10)
            test_rows.append(sub.iloc[:n])
            train_rows.append(sub.iloc[n:])

        test_df  = pd.concat(test_rows ).reset_index(drop=True)
        train_df = pd.concat(train_rows).reset_index(drop=True)

        # Save CSVs (drop duration col)
        for split_df, path in [(train_df, train_csv), (test_df, test_csv)]:
            split_df[["audio","text","speaker_id","gender"]].to_csv(path, index=False)

        print(f"    Train: {len(train_df):,}  |  Test: {len(test_df):,}\n")
        return train_csv, test_csv, audio_dir


# ══════════════════════════════════════════════════════════════════════
# ⑥ MODEL CONFIGURATION FACTORIES
# ══════════════════════════════════════════════════════════════════════
def make_config(
    mode: str,           # "baseline" or "hausa"
    n_speakers: int,     # 0 = single speaker
    train_csv: Path,
    audio_dir: Path,
) -> dict:
    """
    Returns a complete config dict for the experiment.
    Baseline = HausaVITS with tone/prosody features OFF.
    HausaVITS = all Hausa-specific features ON.
    """
    is_hausa = (mode == "hausa")
    return {
        "mode": mode,
        "audio": {
            "sample_rate"  : SAMPLE_RATE,
            "n_fft"        : N_FFT,
            "hop_length"   : HOP_LENGTH,
            "win_length"   : WIN_LENGTH,
            "n_mels"       : N_MELS,
            "f_min"        : 0.0,
            "f_max"        : None,
            "clip_val"     : 1e-5,
        },
        "model": {
            "n_vocab"                  : n_symbols,
            "spec_channels"            : N_FFT // 2 + 1,      # 513
            "segment_size"             : SEGMENT_SIZE,
            "inter_channels"           : 192,
            "hidden_channels"          : 192,
            "filter_channels"          : 768,
            "n_heads"                  : 2,
            "n_layers"                 : 6,
            "kernel_size"              : 3,
            "p_dropout"                : 0.1,
            "resblock"                 : "1",
            "resblock_kernel_sizes"    : [3, 7, 11],
            "resblock_dilation_sizes"  : [[1,3,5],[1,3,5],[1,3,5]],
            "upsample_rates"           : [8, 8, 2, 2],
            "upsample_initial_channel" : 512,
            "upsample_kernel_sizes"    : [16, 16, 4, 4],
            # Hausa-specific
            "n_tones"                  : n_tones,
            "tone_embedding_dim"       : 32,
            "use_tone_predictor"       : is_hausa,
            "use_prosody_encoder"      : is_hausa,
            "prosody_embedding_dim"    : 64,
            "n_mel_channels"           : N_MELS,
            # Multi-speaker
            "n_speakers"               : n_speakers,
            "gin_channels"             : 256 if n_speakers > 0 else 0,
            "use_sdp"                  : True,
        },
        "training": {
            "batch_size"       : BATCH_SIZE,
            "learning_rate_g"  : LR_G,
            "learning_rate_d"  : LR_D,
            "lr_decay"         : LR_DECAY,
            "adam_b1"          : 0.8,
            "adam_b2"          : 0.99,
            "weight_decay"     : 0.01,
            "fp16"             : (DEVICE == "cuda"),
            "grad_clip"        : 5.0,
        },
        "data": {
            "metadata_path"  : str(train_csv),
            "audio_dir"      : str(audio_dir),
            "use_tone_marks" : is_hausa,
            "num_workers"    : 2,
        },
    }


# ══════════════════════════════════════════════════════════════════════
# ⑦ TRAINING ENGINE
# ══════════════════════════════════════════════════════════════════════
class Trainer:
    """
    Lean training loop with:
    - AMP mixed precision
    - Gradient accumulation
    - Per-step HF Hub checkpoint saving
    - Cross-session resume
    - TensorBoard-compatible loss logging
    """

    def __init__(
        self,
        exp_name: str,
        config: dict,
        state: StateManager,
    ):
        self.exp_name = exp_name
        self.cfg      = config
        self.state    = state
        self.out_dir  = Path(WORK_DIR) / "checkpoints" / exp_name
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.loss_log : Dict[str, List] = {
            "step": [], "loss_g": [], "loss_d": [], "loss_mel": [],
            "loss_kl": [], "loss_dur": [], "loss_tone": [],
        }

        self._build_objects()
        self._try_resume()

    # ── Object Construction ─────────────────────────────────────────
    def _build_objects(self):
        mc = self.cfg["model"]
        self.tokenizer = HausaTokenizer()
        self.audio_proc = AudioProcessor(
            **{k: self.cfg["audio"][k]
               for k in ("sample_rate","n_fft","hop_length","win_length","n_mels")}
        )

        self.model = HausaVITS(**mc).to(DEVICE)
        self.mpd   = MultiPeriodDiscriminator().to(DEVICE)
        self.msd   = MultiScaleDiscriminator().to(DEVICE)

        tc = self.cfg["training"]
        self.opt_g = torch.optim.AdamW(
            self.model.parameters(),
            lr=tc["learning_rate_g"],
            betas=(tc["adam_b1"], tc["adam_b2"]),
            weight_decay=tc["weight_decay"],
        )
        self.opt_d = torch.optim.AdamW(
            list(self.mpd.parameters()) + list(self.msd.parameters()),
            lr=tc["learning_rate_d"],
            betas=(tc["adam_b1"], tc["adam_b2"]),
            weight_decay=tc["weight_decay"],
        )
        self.sched_g = get_scheduler(self.opt_g, decay_rate=tc["lr_decay"])
        self.sched_d = get_scheduler(self.opt_d, decay_rate=tc["lr_decay"])
        self.scaler  = GradScaler(enabled=tc["fp16"])

        # Build DataLoader
        n_spk = mc["n_speakers"]
        dataset = HausaTTSDataset(
            metadata_path  = self.cfg["data"]["metadata_path"],
            audio_dir      = self.cfg["data"]["audio_dir"],
            audio_processor= self.audio_proc,
            tokenizer      = self.tokenizer,
            use_tone_marks = self.cfg["data"]["use_tone_marks"],
        )
        self.loader = DataLoader(
            dataset,
            batch_size  = self.cfg["training"]["batch_size"],
            shuffle     = True,
            collate_fn  = HausaCollate(),
            num_workers = self.cfg["data"]["num_workers"],
            pin_memory  = (DEVICE == "cuda"),
            drop_last   = True,
        )
        self.global_step = 0

    def _try_resume(self):
        ckpt_path = self.state.pull_checkpoint(self.exp_name)
        if ckpt_path is None:
            print(f"🆕  {self.exp_name}: starting from scratch")
            return
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        self.model.load_state_dict(ckpt["model"])
        self.mpd.load_state_dict(ckpt["mpd"])
        self.msd.load_state_dict(ckpt["msd"])
        self.opt_g.load_state_dict(ckpt["opt_g"])
        self.opt_d.load_state_dict(ckpt["opt_d"])
        self.sched_g.load_state_dict(ckpt["sched_g"])
        self.sched_d.load_state_dict(ckpt["sched_d"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.global_step = ckpt["step"]
        if "loss_log" in ckpt:
            self.loss_log = ckpt["loss_log"]
        print(f"✅  {self.exp_name}: resumed from step {self.global_step}")

    # ── Save / Push ─────────────────────────────────────────────────
    def _save_checkpoint(self):
        step = self.global_step
        local = self.out_dir / f"checkpoint_{step}.pt"
        torch.save({
            "step"    : step,
            "model"   : self.model.state_dict(),
            "mpd"     : self.mpd.state_dict(),
            "msd"     : self.msd.state_dict(),
            "opt_g"   : self.opt_g.state_dict(),
            "opt_d"   : self.opt_d.state_dict(),
            "sched_g" : self.sched_g.state_dict(),
            "sched_d" : self.sched_d.state_dict(),
            "scaler"  : self.scaler.state_dict(),
            "config"  : self.cfg,
            "loss_log": self.loss_log,
        }, local)
        repo_path = f"{self.exp_name}/checkpoint_{step}.pt"
        self.state.push_file(str(local), repo_path)
        # Remove previous checkpoint to save disk space
        for old in self.out_dir.glob("checkpoint_*.pt"):
            if old.name != f"checkpoint_{step}.pt":
                old.unlink(missing_ok=True)

    def _push_loss_plot(self):
        if len(self.loss_log["step"]) < 2:
            return
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle(f"{self.exp_name} — Training Curves", fontsize=14)
        keys = ["loss_g", "loss_d", "loss_mel", "loss_kl", "loss_dur", "loss_tone"]
        labels = ["Generator", "Discriminator", "Mel Recon", "KL Div",
                  "Duration", "Tone Pred"]
        steps = self.loss_log["step"]
        for ax, key, label in zip(axes.flat, keys, labels):
            vals = self.loss_log[key]
            if vals:
                ax.plot(steps[:len(vals)], vals, linewidth=0.8)
                ax.set_title(label)
                ax.set_xlabel("Step")
                ax.grid(alpha=0.3)
        plt.tight_layout()
        path = self.out_dir / "loss_curves.png"
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        self.state.push_file(str(path), f"{self.exp_name}/loss_curves.png")

    # ── Training Step ───────────────────────────────────────────────
    def _step(self, batch) -> dict:
        mc = self.cfg["model"]
        to = lambda t: t.to(DEVICE) if isinstance(t, torch.Tensor) else t

        x        = to(batch["phoneme_ids"])
        x_len    = to(batch["phoneme_lengths"])
        y        = to(batch["linear_spec"])
        y_len    = to(batch["mel_lengths"])
        wav      = to(batch["wav"])
        tones    = to(batch["tone_ids"])  if "tone_ids"    in batch else None
        sid      = to(batch["speaker_ids"]) if mc["n_speakers"] > 0 else None
        mel      = to(batch["mel"])

        with autocast(enabled=self.cfg["training"]["fp16"]):
            out = self.model(x, x_len, y, y_len,
                             tones=tones, sid=sid, mel=mel)
            y_hat     = out["y_hat"]
            ids_slice = out["ids_slice"]

            # Slice ground-truth wav to match generated segment
            wav_seg = torch.zeros(
                wav.size(0), 1, SEGMENT_SIZE, device=DEVICE
            )
            for i in range(wav.size(0)):
                s = ids_slice[i].item() * HOP_LENGTH
                e = min(s + SEGMENT_SIZE, wav.size(-1))
                l = e - s
                wav_seg[i, 0, :l] = wav[i, 0, s:e]

            # ── Discriminator update ──────────────────────────────
            y_d_rs, y_d_gs, fmaps_r, fmaps_g = self.mpd(
                wav_seg, y_hat.detach()
            )
            y_ds_rs, y_ds_gs, fmaps_sr, fmaps_sg = self.msd(
                wav_seg, y_hat.detach()
            )
            loss_d = (
                discriminator_loss(y_d_rs, y_d_gs)[0]
                + discriminator_loss(y_ds_rs, y_ds_gs)[0]
            )

        self.opt_d.zero_grad()
        self.scaler.scale(loss_d / GRAD_ACCUM).backward()

        with autocast(enabled=self.cfg["training"]["fp16"]):
            # ── Generator update ──────────────────────────────────
            y_d_rs, y_d_gs, fmaps_r, fmaps_g = self.mpd(wav_seg, y_hat)
            y_ds_rs, y_ds_gs, fmaps_sr, fmaps_sg = self.msd(wav_seg, y_hat)

            # Mel of generated vs ground truth
            with torch.no_grad():
                mel_hat = torch.nn.functional.avg_pool1d(
                    torch.stft(
                        y_hat.squeeze(1),
                        n_fft=N_FFT, hop_length=HOP_LENGTH,
                        win_length=WIN_LENGTH,
                        window=torch.hann_window(WIN_LENGTH).to(DEVICE),
                        return_complex=True,
                    ).abs().pow(2).squeeze(0).unsqueeze(0),
                    kernel_size=1,
                )

            loss_mel  = mel_reconstruction_loss(mel[:, :, :mel_hat.size(2)]
                                                if mel_hat.dim() == 3 else mel,
                                                mel[:, :, :mel.size(2)])
            loss_kl   = kl_loss(out["z_p"], out["logs_q"],
                                out["m_p"],  out["logs_p"], out["y_mask"])
            loss_dur  = out["l_length"].mean()
            loss_fm   = (feature_loss(fmaps_r, fmaps_g)
                         + feature_loss(fmaps_sr, fmaps_sg))
            loss_adv  = (generator_loss(y_d_gs) + generator_loss(y_ds_gs))

            loss_tone = torch.tensor(0.0, device=DEVICE)
            if out["tone_logits"] is not None and tones is not None:
                loss_tone = tone_prediction_loss(
                    out["tone_logits"].permute(0, 2, 1),
                    tones, out["x_mask"]
                )

            loss_g = (loss_adv + 2.0 * loss_fm + 45.0 * loss_mel
                      + loss_kl + loss_dur + loss_tone)

        self.opt_g.zero_grad()
        self.scaler.scale(loss_g / GRAD_ACCUM).backward()

        return {
            "loss_g"   : loss_g.item(),
            "loss_d"   : loss_d.item(),
            "loss_mel" : loss_mel.item(),
            "loss_kl"  : loss_kl.item(),
            "loss_dur" : loss_dur.item(),
            "loss_tone": loss_tone.item(),
        }

    # ── Main Training Loop ──────────────────────────────────────────
    def train(self):
        print(f"\n{'═'*60}")
        print(f"  Training: {self.exp_name}")
        print(f"  Mode    : {self.cfg['mode']}")
        print(f"  Start   : step {self.global_step}")
        print(f"  Target  : step {MAX_TRAIN_STEPS}")
        print(f"{'═'*60}\n")

        self.model.train()
        self.mpd.train()
        self.msd.train()

        loader_iter    = iter(self.loader)
        accum_losses   = {k: 0.0 for k in
                          ["loss_g","loss_d","loss_mel","loss_kl","loss_dur","loss_tone"]}
        accum_count    = 0
        pbar           = tqdm(
            total=MAX_TRAIN_STEPS,
            initial=self.global_step,
            desc=self.exp_name[:20],
            dynamic_ncols=True,
        )

        while self.global_step < MAX_TRAIN_STEPS:
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(self.loader)
                batch = next(loader_iter)

            losses = self._step(batch)
            accum_count += 1

            for k in accum_losses:
                accum_losses[k] += losses[k]

            # Gradient step every GRAD_ACCUM mini-batches
            if accum_count == GRAD_ACCUM:
                clip = self.cfg["training"]["grad_clip"]
                self.scaler.unscale_(self.opt_g)
                self.scaler.unscale_(self.opt_d)
                nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                nn.utils.clip_grad_norm_(
                    list(self.mpd.parameters())
                    + list(self.msd.parameters()), clip
                )
                self.scaler.step(self.opt_g)
                self.scaler.step(self.opt_d)
                self.scaler.update()
                self.sched_g.step()
                self.sched_d.step()

                self.global_step += 1
                accum_count = 0

                # Log
                avg_losses = {k: v / GRAD_ACCUM for k, v in accum_losses.items()}
                self.loss_log["step"].append(self.global_step)
                for k, v in avg_losses.items():
                    self.loss_log[k].append(v)
                accum_losses = {k: 0.0 for k in accum_losses}

                pbar.update(1)
                pbar.set_postfix({
                    "G": f"{avg_losses['loss_g']:.3f}",
                    "D": f"{avg_losses['loss_d']:.3f}",
                    "mel": f"{avg_losses['loss_mel']:.3f}",
                })

                # Checkpoint + push
                if self.global_step % SAVE_EVERY == 0:
                    self._save_checkpoint()
                    self._push_loss_plot()
                    self.state.update(self.exp_name, self.global_step)

                # Memory management
                if self.global_step % 200 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

        pbar.close()
        self._save_checkpoint()
        self._push_loss_plot()
        self.state.complete(self.exp_name)
        print(f"✅  {self.exp_name} training complete!\n")


# ══════════════════════════════════════════════════════════════════════
# ⑧ EVALUATION SUITE
# ══════════════════════════════════════════════════════════════════════
class Evaluator:
    """
    Automated evaluation metrics (no human evaluation).
    All metrics are standard in TTS/speech synthesis research papers.
    """

    def __init__(self, audio_proc: AudioProcessor):
        self.proc   = audio_proc
        self._whisper = None   # lazy load
        self._utmos   = None   # lazy load

    # ── Load audio ──────────────────────────────────────────────────
    @staticmethod
    def load_wav(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
        y, sr_src = librosa.load(path, sr=None, mono=True)
        if sr_src != sr:
            y = librosa.resample(y, orig_sr=sr_src, target_sr=sr)
        return y.astype(np.float32)

    # ── MCD (Mel Cepstral Distortion) ───────────────────────────────
    def compute_mcd(
        self, ref: np.ndarray, syn: np.ndarray,
        n_mfcc: int = 13, sr: int = SAMPLE_RATE
    ) -> float:
        """
        DTW-aligned Mel Cepstral Distortion (dB). Lower is better.
        Standard metric for TTS acoustic quality evaluation.
        """
        ref_mfcc = librosa.feature.mfcc(y=ref, sr=sr, n_mfcc=n_mfcc+1)[1:]
        syn_mfcc = librosa.feature.mfcc(y=syn, sr=sr, n_mfcc=n_mfcc+1)[1:]

        # DTW alignment
        D, wp = librosa.sequence.dtw(ref_mfcc, syn_mfcc, metric="euclidean")

        # Compute MCD along warping path
        ref_aligned = ref_mfcc[:, wp[:, 0]]
        syn_aligned = syn_mfcc[:, wp[:, 1]]
        diff = ref_aligned - syn_aligned
        mcd  = (10.0 / math.log(10)) * math.sqrt(
            2.0 * np.mean(np.sum(diff ** 2, axis=0))
        )
        return float(mcd)

    # ── UTMOS (Automatic MOS Predictor) ─────────────────────────────
    def compute_utmos(self, audio: np.ndarray, sr: int = SAMPLE_RATE) -> float:
        """
        UTMOS: Predicted Mean Opinion Score (1–5). Higher is better.
        Uses the SpeechMOS neural predictor as a human-listener proxy.
        """
        try:
            if self._utmos is None:
                from speechmos import utmos
                self._utmos = utmos
            score = self._utmos.predict_mos(audio, sr)
            return float(score)
        except Exception as e:
            print(f"  ⚠️  UTMOS: {e}")
            return float("nan")

    # ── WER via Whisper ──────────────────────────────────────────────
    def compute_wer(self, audio: np.ndarray, ref_text: str,
                    sr: int = SAMPLE_RATE) -> Tuple[float, float]:
        """
        Word Error Rate (WER) and Character Error Rate (CER) using
        OpenAI Whisper (small) as the ASR oracle.
        Lower is better — measures intelligibility of synthesized speech.
        """
        try:
            if self._whisper is None:
                import whisper as wh
                self._whisper = wh.load_model("small", device=DEVICE)

            tmp = Path(WORK_DIR) / "_eval_tmp.wav"
            sf.write(str(tmp), audio, sr)
            result = self._whisper.transcribe(
                str(tmp), language="ha", fp16=(DEVICE == "cuda")
            )
            hyp = result["text"].strip().lower()
            ref = ref_text.strip().lower()

            wer = self._word_error_rate(ref, hyp)
            cer = self._char_error_rate(ref, hyp)
            tmp.unlink(missing_ok=True)
            return wer, cer
        except Exception as e:
            print(f"  ⚠️  WER: {e}")
            return float("nan"), float("nan")

    @staticmethod
    def _edit_distance(r: list, h: list) -> int:
        d = np.zeros((len(r)+1, len(h)+1), dtype=int)
        d[:, 0] = np.arange(len(r)+1)
        d[0, :] = np.arange(len(h)+1)
        for i in range(1, len(r)+1):
            for j in range(1, len(h)+1):
                cost = 0 if r[i-1] == h[j-1] else 1
                d[i,j] = min(d[i-1,j]+1, d[i,j-1]+1, d[i-1,j-1]+cost)
        return int(d[-1,-1])

    def _word_error_rate(self, ref: str, hyp: str) -> float:
        r, h = ref.split(), hyp.split()
        return self._edit_distance(r, h) / max(len(r), 1)

    def _char_error_rate(self, ref: str, hyp: str) -> float:
        r, h = list(ref.replace(" ","")), list(hyp.replace(" ",""))
        return self._edit_distance(r, h) / max(len(r), 1)

    # ── F0 Tone Metrics (Hausa-specific, Novel) ──────────────────────
    def compute_f0_metrics(
        self, ref: np.ndarray, syn: np.ndarray, sr: int = SAMPLE_RATE
    ) -> dict:
        """
        Fundamental frequency (F0) analysis — critical for tonal languages.

        Metrics:
        - F0 RMSE (Hz):     pitch contour accuracy (lower = better)
        - F0 Correlation:   pitch shape similarity (higher = better)
        - Voiced/Unvoiced Error Rate: frame-level V/UV classification error
        - Pitch Declination Slope: captures Hausa downdrift pattern
        """
        try:
            fmin, fmax = 75, 600
            f0_ref, vf_ref, _ = librosa.pyin(
                ref, fmin=fmin, fmax=fmax, sr=sr
            )
            f0_syn, vf_syn, _ = librosa.pyin(
                syn, fmin=fmin, fmax=fmax, sr=sr
            )
            # Align lengths
            L = min(len(f0_ref), len(f0_syn))
            f0_ref, f0_syn = f0_ref[:L], f0_syn[:L]
            vf_ref, vf_syn = vf_ref[:L], vf_syn[:L]

            # Both voiced frames
            both_voiced = vf_ref & vf_syn
            if both_voiced.sum() < 10:
                return {"f0_rmse": float("nan"), "f0_corr": float("nan"),
                        "vuv_error": float("nan"), "declination_slope": float("nan")}

            r = f0_ref[both_voiced]
            s = f0_syn[both_voiced]
            f0_rmse = float(np.sqrt(np.mean((r - s) ** 2)))
            f0_corr = float(np.corrcoef(r, s)[0, 1])

            # V/UV error rate
            vuv_err = float(np.mean(vf_ref != vf_syn))

            # Pitch declination slope (linear regression on ref F0)
            x = np.where(vf_ref)[0]
            y = f0_ref[vf_ref]
            if len(x) > 2:
                slope = float(np.polyfit(x, y, 1)[0])
            else:
                slope = float("nan")

            return {
                "f0_rmse"           : f0_rmse,
                "f0_corr"           : f0_corr,
                "vuv_error"         : vuv_err,
                "declination_slope" : slope,
            }
        except Exception as e:
            print(f"  ⚠️  F0: {e}")
            return {"f0_rmse": float("nan"), "f0_corr": float("nan"),
                    "vuv_error": float("nan"), "declination_slope": float("nan")}

    # ── RTF (Real-Time Factor) ───────────────────────────────────────
    def compute_rtf(
        self, model: HausaVITS, tokenizer: HausaTokenizer,
        text: str, n_runs: int = 3,
        use_tone_marks: bool = False,
    ) -> float:
        """
        Real-Time Factor = inference_time / audio_duration.
        Lower is better. RTF < 1.0 means faster than real time.
        """
        model.eval()
        try:
            ids, tones = tokenizer.text_to_sequence(
                text, use_tone_marks=use_tone_marks
            )
            ids   = ids.unsqueeze(0).to(DEVICE)
            tones = tones.unsqueeze(0).to(DEVICE)
            xlen  = torch.LongTensor([ids.size(1)]).to(DEVICE)

            times = []
            for _ in range(n_runs):
                t0 = time.time()
                with torch.no_grad():
                    wav, _, _ = model.infer(ids, xlen, tones=tones)
                if DEVICE == "cuda":
                    torch.cuda.synchronize()
                times.append(time.time() - t0)

            inf_time  = np.median(times)
            audio_dur = wav.size(-1) / SAMPLE_RATE
            return float(inf_time / audio_dur)
        except Exception as e:
            print(f"  ⚠️  RTF: {e}")
            return float("nan")
        finally:
            model.train()

    # ── Full Evaluation ──────────────────────────────────────────────
    def evaluate(
        self,
        model: HausaVITS,
        tokenizer: HausaTokenizer,
        test_csv: Path,
        audio_dir: Path,
        exp_name: str,
        use_tone_marks: bool,
    ) -> dict:
        """
        Runs the full evaluation suite on the held-out test set.
        Returns aggregate metrics dict.
        """
        df = pd.read_csv(test_csv)
        print(f"\n📊  Evaluating {exp_name} on {len(df)} samples …")
        model.eval()

        results = []
        sample_wavs = []

        for i, row in tqdm(df.iterrows(), total=len(df), desc="Eval"):
            ref_path = Path(audio_dir) / row["audio"]
            text     = str(row["text"])

            # Synthesize
            try:
                ids, tones = tokenizer.text_to_sequence(
                    text, use_tone_marks=use_tone_marks
                )
                ids   = ids.unsqueeze(0).to(DEVICE)
                tones = tones.unsqueeze(0).to(DEVICE)
                xlen  = torch.LongTensor([ids.size(1)]).to(DEVICE)

                with torch.no_grad():
                    syn_tensor, _, _ = model.infer(ids, xlen, tones=tones)
                syn_wav = syn_tensor.squeeze().cpu().numpy()
            except Exception as e:
                print(f"  ⚠️  Synth failed row {i}: {e}")
                continue

            ref_wav = self.load_wav(str(ref_path))

            # Metrics
            mcd    = self.compute_mcd(ref_wav, syn_wav)
            utmos  = self.compute_utmos(syn_wav)
            wer, cer = self.compute_wer(syn_wav, text)
            f0m    = self.compute_f0_metrics(ref_wav, syn_wav)

            results.append({
                "sample_idx"        : i,
                "mcd"               : mcd,
                "utmos"             : utmos,
                "wer"               : wer,
                "cer"               : cer,
                **f0m,
            })

            # Save first 5 samples as audio
            if len(sample_wavs) < 5:
                sample_wavs.append((syn_wav, text))

            # Memory cleanup every 10 samples
            if i % 10 == 0:
                torch.cuda.empty_cache()

        # RTF (on a fixed sentence)
        rtf = self.compute_rtf(
            model, tokenizer,
            "Barka da safe ina fatan lafiya",
            use_tone_marks=use_tone_marks,
        )

        model.train()
        torch.cuda.empty_cache()

        if not results:
            return {}

        df_res = pd.DataFrame(results)
        agg = {
            "mcd_mean"          : df_res["mcd"].mean(),
            "mcd_std"           : df_res["mcd"].std(),
            "utmos_mean"        : df_res["utmos"].mean(),
            "utmos_std"         : df_res["utmos"].std(),
            "wer_mean"          : df_res["wer"].mean(),
            "cer_mean"          : df_res["cer"].mean(),
            "f0_rmse_mean"      : df_res["f0_rmse"].mean(),
            "f0_corr_mean"      : df_res["f0_corr"].mean(),
            "vuv_error_mean"    : df_res["vuv_error"].mean(),
            "declination_slope" : df_res["declination_slope"].mean(),
            "rtf"               : rtf,
            "n_samples"         : len(df_res),
        }
        return agg, df_res, sample_wavs


# ══════════════════════════════════════════════════════════════════════
# ⑨ RESULTS MANAGER
# ══════════════════════════════════════════════════════════════════════
class ResultsManager:
    """Saves results tables, comparison plots, and pushes to HF Hub."""

    def __init__(self, state: StateManager):
        self.state  = state
        self.out    = Path(WORK_DIR) / "results"
        self.out.mkdir(parents=True, exist_ok=True)

    def save_and_push(self, all_results: dict):
        """
        all_results: {exp_name: (agg_metrics, df_per_sample, sample_wavs)}
        """
        # ── Per-sample CSVs ─────────────────────────────────────────
        for exp, (agg, df, wavs) in all_results.items():
            csv_path = self.out / f"{exp}_per_sample.csv"
            df.to_csv(csv_path, index=False)
            self.state.push_file(str(csv_path), f"results/{exp}_per_sample.csv")

            # Push audio samples
            for j, (wav, text) in enumerate(wavs):
                wav_path = self.out / f"{exp}_sample_{j}.wav"
                sf.write(str(wav_path), wav, SAMPLE_RATE)
                self.state.push_file(
                    str(wav_path), f"results/samples/{exp}_sample_{j}.wav"
                )

        # ── Aggregate comparison table ───────────────────────────────
        rows = []
        for exp, (agg, _, _) in all_results.items():
            row = {"experiment": exp}
            row.update(agg)
            rows.append(row)

        summary_df = pd.DataFrame(rows)
        csv_path   = self.out / "summary_results.csv"
        summary_df.to_csv(csv_path, index=False)
        self.state.push_file(str(csv_path), "results/summary_results.csv")

        # ── Pretty comparison table ──────────────────────────────────
        self._print_table(summary_df)
        self._plot_comparison(summary_df, all_results)

        # Push summary JSON
        json_path = self.out / "summary_results.json"
        with open(json_path, "w") as f:
            json.dump(summary_df.to_dict(orient="records"), f, indent=2, default=str)
        self.state.push_file(str(json_path), "results/summary_results.json")

    def _print_table(self, df: pd.DataFrame):
        print("\n" + "═"*90)
        print("  RESULTS SUMMARY")
        print("═"*90)
        cols = ["experiment","mcd_mean","utmos_mean","wer_mean","f0_rmse_mean","f0_corr_mean","rtf"]
        print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print("═"*90 + "\n")

        # Compute improvements
        if "single_baseline" in df["experiment"].values and \
           "single_hausa" in df["experiment"].values:
            bl = df[df["experiment"]=="single_baseline"].iloc[0]
            ha = df[df["experiment"]=="single_hausa"].iloc[0]
            print("📈  Single-Speaker Improvements (HausaVITS vs Baseline):")
            print(f"    MCD      : {bl['mcd_mean']:.3f} → {ha['mcd_mean']:.3f}"
                  f"  (Δ {ha['mcd_mean']-bl['mcd_mean']:+.3f} dB)")
            print(f"    UTMOS    : {bl['utmos_mean']:.3f} → {ha['utmos_mean']:.3f}"
                  f"  (Δ {ha['utmos_mean']-bl['utmos_mean']:+.3f})")
            print(f"    WER      : {bl['wer_mean']:.3f} → {ha['wer_mean']:.3f}"
                  f"  (Δ {ha['wer_mean']-bl['wer_mean']:+.3f})")
            print(f"    F0 RMSE  : {bl['f0_rmse_mean']:.2f} → {ha['f0_rmse_mean']:.2f}"
                  f"  (Δ {ha['f0_rmse_mean']-bl['f0_rmse_mean']:+.2f} Hz)")
            print(f"    F0 Corr  : {bl['f0_corr_mean']:.3f} → {ha['f0_corr_mean']:.3f}"
                  f"  (Δ {ha['f0_corr_mean']-bl['f0_corr_mean']:+.3f})")
            print()

    def _plot_comparison(self, df: pd.DataFrame, all_results: dict):
        metrics = [
            ("mcd_mean",     "MCD (↓ better)",     True),
            ("utmos_mean",   "UTMOS (↑ better)",   False),
            ("wer_mean",     "WER (↓ better)",      True),
            ("f0_rmse_mean", "F0 RMSE Hz (↓ better)", True),
            ("f0_corr_mean", "F0 Corr (↑ better)", False),
            ("rtf",          "RTF (↓ better)",      True),
        ]

        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        fig.suptitle("HausaVITS vs Baseline VITS — Evaluation Results", fontsize=14)
        colors = {"baseline": "#E57373", "hausa": "#64B5F6"}

        for ax, (metric, label, lower_better) in zip(axes.flat, metrics):
            exps  = df["experiment"].tolist()
            vals  = df[metric].tolist()
            clrs  = [colors["baseline"] if "baseline" in e else colors["hausa"]
                     for e in exps]
            bars  = ax.bar(range(len(exps)), vals, color=clrs)
            ax.set_xticks(range(len(exps)))
            ax.set_xticklabels(
                [e.replace("_", "\n") for e in exps], fontsize=8
            )
            ax.set_title(label, fontsize=10)
            ax.grid(axis="y", alpha=0.3)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7)

        # Legend
        from matplotlib.patches import Patch
        legend = [Patch(color=colors["baseline"], label="Baseline VITS"),
                  Patch(color=colors["hausa"],    label="HausaVITS (Ours)")]
        fig.legend(handles=legend, loc="upper right", fontsize=10)

        plt.tight_layout()
        path = self.out / "comparison_chart.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        self.state.push_file(str(path), "results/comparison_chart.png")
        print(f"📊  Comparison chart saved.")

    def generate_latex_table(self, df: pd.DataFrame):
        """Generate a LaTeX table ready to paste into a paper."""
        latex = r"""\begin{table}[htbp]
\centering
\caption{Evaluation Results: Baseline VITS vs. HausaVITS}
\label{tab:results}
\begin{tabular}{lcccccc}
\toprule
\textbf{System} & \textbf{MCD↓} & \textbf{UTMOS↑} & \textbf{WER↓} & \textbf{F0 RMSE↓} & \textbf{F0 Corr↑} & \textbf{RTF↓} \\
\midrule
"""
        for _, row in df.iterrows():
            name = row["experiment"].replace("_", " ").title()
            latex += (
                f"{name} & "
                f"{row['mcd_mean']:.2f}$\\pm${row['mcd_std']:.2f} & "
                f"{row['utmos_mean']:.2f} & "
                f"{row['wer_mean']:.3f} & "
                f"{row['f0_rmse_mean']:.2f} & "
                f"{row['f0_corr_mean']:.3f} & "
                f"{row['rtf']:.3f} \\\\\n"
            )
        latex += r"""\bottomrule
\end{tabular}
\end{table}"""
        path = self.out / "results_table.tex"
        with open(path, "w") as f:
            f.write(latex)
        self.state.push_file(str(path), "results/results_table.tex")
        print(f"\n📄  LaTeX table saved → results/results_table.tex")
        print("     (Paste into your paper's results section)\n")


# ══════════════════════════════════════════════════════════════════════
# ⑩ MAIN ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════
def run_experiment(
    exp_name: str,
    config: dict,
    state: StateManager,
):
    """Train one model end-to-end. Returns trained model + tokenizer."""
    trainer = Trainer(exp_name, config, state)
    trainer.train()

    model     = trainer.model
    tokenizer = trainer.tokenizer
    del trainer.loader, trainer.opt_g, trainer.opt_d
    gc.collect()
    torch.cuda.empty_cache()

    return model, tokenizer, config


def main():
    print("\n" + "█"*60)
    print("  HausaVITS Research Experiment")
    print("  Baseline VITS vs HausaVITS")
    print("  Auto-resume across Kaggle sessions")
    print("█"*60 + "\n")

    # ── State ────────────────────────────────────────────────────────
    state = StateManager()

    # ── Prepare datasets ─────────────────────────────────────────────
    data = DataPreparer(WORK_DIR)

    print("📥  Preparing single-speaker dataset (speaker 88) …")
    s_train, s_test, s_audio = data.prepare(
        speaker_ids=[SINGLE_SPEAKER_ID],
        tag="single",
    )

    print("📥  Preparing multi-speaker dataset (speakers 225, 40) …")
    m_train, m_test, m_audio = data.prepare(
        speaker_ids=MULTI_SPEAKER_IDS,
        tag="multi",
    )

    # ── Experiment definitions ────────────────────────────────────────
    experiments = [
        {
            "name"       : "single_baseline",
            "mode"       : "baseline",
            "n_speakers" : 0,
            "train_csv"  : s_train,
            "test_csv"   : s_test,
            "audio_dir"  : s_audio,
        },
        {
            "name"       : "single_hausa",
            "mode"       : "hausa",
            "n_speakers" : 0,
            "train_csv"  : s_train,
            "test_csv"   : s_test,
            "audio_dir"  : s_audio,
        },
        {
            "name"       : "multi_baseline",
            "mode"       : "baseline",
            "n_speakers" : len(MULTI_SPEAKER_IDS),
            "train_csv"  : m_train,
            "test_csv"   : m_test,
            "audio_dir"  : m_audio,
        },
        {
            "name"       : "multi_hausa",
            "mode"       : "hausa",
            "n_speakers" : len(MULTI_SPEAKER_IDS),
            "train_csv"  : m_train,
            "test_csv"   : m_test,
            "audio_dir"  : m_audio,
        },
    ]

    # ── Training phase ───────────────────────────────────────────────
    trained_models: Dict[str, Tuple] = {}

    for exp in experiments:
        name = exp["name"]

        if state.is_done(name):
            print(f"⏭️   Skipping {name} (already completed)")
            continue

        config = make_config(
            mode       = exp["mode"],
            n_speakers = exp["n_speakers"],
            train_csv  = exp["train_csv"],
            audio_dir  = exp["audio_dir"],
        )

        model, tokenizer, cfg = run_experiment(name, config, state)
        trained_models[name] = (model, tokenizer, cfg,
                                 exp["test_csv"], exp["audio_dir"])

        # Free GPU memory between models
        del model
        gc.collect()
        torch.cuda.empty_cache()
        print(f"🧹  GPU cleared after {name}\n")

    # ── Evaluation phase ─────────────────────────────────────────────
    if state.state.get("evaluation_done"):
        print("⏭️   Evaluation already done. Check HF Hub for results.")
        return

    print("\n" + "═"*60)
    print("  EVALUATION PHASE")
    print("  Loading all 4 trained models for comparison …")
    print("═"*60)

    audio_proc = AudioProcessor(
        sample_rate=SAMPLE_RATE, n_fft=N_FFT,
        hop_length=HOP_LENGTH, win_length=WIN_LENGTH, n_mels=N_MELS,
    )
    evaluator = Evaluator(audio_proc)
    results_mgr = ResultsManager(state)
    all_results = {}

    for exp in experiments:
        name = exp["name"]
        config = make_config(
            mode       = exp["mode"],
            n_speakers = exp["n_speakers"],
            train_csv  = exp["train_csv"],
            audio_dir  = exp["audio_dir"],
        )
        mc = config["model"]

        # Load model from HF Hub checkpoint
        print(f"\n🔬  Evaluating: {name}")
        model = HausaVITS(**mc).to(DEVICE)
        ckpt_path = state.pull_checkpoint(name)
        if ckpt_path:
            ckpt = torch.load(ckpt_path, map_location=DEVICE)
            model.load_state_dict(ckpt["model"])
        else:
            print(f"  ⚠️  No checkpoint for {name} — evaluation may be random.")

        tokenizer = HausaTokenizer()

        result = evaluator.evaluate(
            model        = model,
            tokenizer    = tokenizer,
            test_csv     = exp["test_csv"],
            audio_dir    = exp["audio_dir"],
            exp_name     = name,
            use_tone_marks = (exp["mode"] == "hausa"),
        )
        if result:
            all_results[name] = result

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # ── Save results ─────────────────────────────────────────────────
    if all_results:
        results_mgr.save_and_push(all_results)
        # Latex table
        agg_rows = [{"experiment": k, **v[0]} for k, v in all_results.items()]
        results_mgr.generate_latex_table(pd.DataFrame(agg_rows))

    state.state["evaluation_done"] = True
    state.save()

    print("\n" + "█"*60)
    print("  🎉  ALL EXPERIMENTS COMPLETE!")
    print(f"  📂  Results: https://huggingface.co/{HF_REPO_ID}")
    print("  Files pushed:")
    print("    • state.json                (progress tracker)")
    print("    • {exp}/checkpoint_N.pt     (model checkpoints)")
    print("    • {exp}/loss_curves.png     (training curves)")
    print("    • results/summary_results.csv   (all metrics)")
    print("    • results/comparison_chart.png  (bar chart)")
    print("    • results/results_table.tex     (LaTeX for paper)")
    print("    • results/samples/*.wav         (audio examples)")
    print("█"*60 + "\n")


# ══════════════════════════════════════════════════════════════════════
# ▶  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__" or True:   # always runs in notebook context
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏸️   Interrupted — progress saved. Re-run to continue.")
    except Exception as e:
        print(f"\n❌  Error: {e}")
        traceback.print_exc()
        print("Progress auto-saved. Re-run to continue from last checkpoint.")
