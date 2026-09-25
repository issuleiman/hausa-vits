import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import os
import logging
from typing import Optional, Dict

from hausa_tts.model.vits import HausaVITS
from hausa_tts.model.discriminator import MultiPeriodDiscriminator, MultiScaleDiscriminator
from hausa_tts.model.losses import (
    generator_loss, discriminator_loss, feature_loss, 
    kl_loss, mel_reconstruction_loss, tone_prediction_loss
)
from hausa_tts.data.dataset import HausaTTSDataset
from hausa_tts.data.collate import HausaCollate
from hausa_tts.data.audio import AudioProcessor
from hausa_tts.text.tokenizer import HausaTokenizer
from hausa_tts.training.scheduler import get_scheduler

class Trainer:
    """HausaVITS trainer with DDP and mixed precision support."""
    def __init__(self, config: dict, rank: int = 0, world_size: int = 1):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
        
        # 1. Initialize models
        self.net_g = HausaVITS(config).to(self.device)
        self.net_d_mpd = MultiPeriodDiscriminator().to(self.device)
        self.net_d_msd = MultiScaleDiscriminator().to(self.device)
        
        # 2. Optimizers
        self.optim_g = torch.optim.AdamW(
            self.net_g.parameters(),
            config['training']['learning_rate_g'],
            betas=(config['training']['adam_b1'], config['training']['adam_b2']),
            eps=1e-9
        )
        self.optim_d = torch.optim.AdamW(
            list(self.net_d_mpd.parameters()) + list(self.net_d_msd.parameters()),
            config['training']['learning_rate_d'],
            betas=(config['training']['adam_b1'], config['training']['adam_b2']),
            eps=1e-9
        )
        
        # 3. Schedulers
        self.scheduler_g = get_scheduler(self.optim_g, config['training']['lr_decay'])
        self.scheduler_d = get_scheduler(self.optim_d, config['training']['lr_decay'])
        
        # 4. Scaler
        self.scaler = GradScaler(enabled=config['training'].get('fp16', False))
        
        # 5. DDP wrapping
        if world_size > 1:
            self.net_g = DDP(self.net_g, device_ids=[rank])
            self.net_d_mpd = DDP(self.net_d_mpd, device_ids=[rank])
            self.net_d_msd = DDP(self.net_d_msd, device_ids=[rank])
            
        # 6. TensorBoard
        if rank == 0:
            os.makedirs(config['log_dir'], exist_ok=True)
            self.writer = SummaryWriter(log_dir=config['log_dir'])
            
        self.step = 0
        self.epoch = 0

    def train(self, dataloader):
        """Main training loop."""
        self.net_g.train()
        self.net_d_mpd.train()
        self.net_d_msd.train()
        
        for epoch in range(self.epoch, self.config['training']['epochs']):
            if self.world_size > 1:
                dataloader.sampler.set_epoch(epoch)
            
            for batch in dataloader:
                # Move to device
                phonemes = batch['phoneme_ids'].to(self.device)
                phoneme_lengths = batch['phoneme_lengths'].to(self.device)
                tones = batch['tone_ids'].to(self.device)
                mels = batch['mel'].to(self.device)
                mel_lengths = batch['mel_lengths'].to(self.device)
                linear = batch['linear_spec'].to(self.device)
                wavs = batch['wav'].to(self.device)
                wav_lengths = batch['wav_lengths'].to(self.device)
                speakers = batch['speaker_ids'].to(self.device) if 'speaker_ids' in batch else None
                
                # Generator Forward
                with autocast(enabled=self.config['training'].get('fp16', False)):
                    y_hat, l_length, attn, ids_slice, x_mask, z_mask, \
                    (z, z_p, m_p, logs_p, m_q, logs_q), tone_preds = self.net_g(
                        phonemes, phoneme_lengths, linear, mel_lengths, tones, speakers
                    )
                    
                    # Slice real wav
                    mel_slice = slice(0, y_hat.shape[-1] // self.config['data']['hop_length'])
                    wav_real_sliced = wavs[:, 0:1, ids_slice.squeeze(1)[0] * self.config['data']['hop_length'] : (ids_slice.squeeze(1)[0] + y_hat.shape[-1] // self.config['data']['hop_length']) * self.config['data']['hop_length']]

                # Discriminator Step
                self.optim_d.zero_grad()
                with autocast(enabled=self.config['training'].get('fp16', False)):
                    y_d_rs, y_d_gs, fmap_rs, fmap_gs = self.net_d_mpd(wav_real_sliced, y_hat.detach())
                    y_d_rs_m, y_d_gs_m, fmap_rs_m, fmap_gs_m = self.net_d_msd(wav_real_sliced, y_hat.detach())
                    
                    loss_d, losses_d_r, losses_d_g = discriminator_loss(y_d_rs, y_d_gs, y_d_rs_m, y_d_gs_m)
                    
                self.scaler.scale(loss_d).backward()
                self.scaler.unscale_(self.optim_d)
                torch.nn.utils.clip_grad_norm_(self.net_d_mpd.parameters(), self.config['training'].get('grad_clip', 5.0))
                self.scaler.step(self.optim_d)
                
                # Generator Step
                self.optim_g.zero_grad()
                with autocast(enabled=self.config['training'].get('fp16', False)):
                    y_d_rs, y_d_gs, fmap_rs, fmap_gs = self.net_d_mpd(wav_real_sliced, y_hat)
                    y_d_rs_m, y_d_gs_m, fmap_rs_m, fmap_gs_m = self.net_d_msd(wav_real_sliced, y_hat)
                    
                    loss_mel = mel_reconstruction_loss(wav_real_sliced, y_hat)
                    loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask)
                    loss_fm = feature_loss(fmap_rs, fmap_gs, fmap_rs_m, fmap_gs_m)
                    loss_gen, losses_gen = generator_loss(y_d_gs, y_d_gs_m)
                    
                    loss_tone = tone_prediction_loss(tone_preds, tones) if tone_preds is not None else 0.0
                    loss_g_all = loss_gen + loss_fm + loss_mel + loss_kl + loss_tone
                    
                self.scaler.scale(loss_g_all).backward()
                self.scaler.unscale_(self.optim_g)
                torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), self.config['training'].get('grad_clip', 5.0))
                self.scaler.step(self.optim_g)
                self.scaler.update()
                
                self.step += 1
                
                if self.rank == 0 and self.step % self.config['training']['log_interval'] == 0:
                    self.writer.add_scalar('loss/g/total', loss_g_all, self.step)
                    self.writer.add_scalar('loss/d/total', loss_d, self.step)
                    
            self.scheduler_g.step()
            self.scheduler_d.step()
            self.epoch += 1
            
            if self.rank == 0 and self.epoch % self.config['training']['save_interval'] == 0:
                self.save_checkpoint(self.step)

    def validate(self, step: int):
        pass

    def save_checkpoint(self, step: int):
        path = os.path.join(self.config['log_dir'], f"G_{step}.pth")
        torch.save({'model': self.net_g.state_dict(), 'optim': self.optim_g.state_dict(), 'step': step}, path)
        
    def load_checkpoint(self, path: str):
        pass
    
    def _log_audio(self, tag: str, audio: torch.Tensor, step: int):
        pass
    
    def _log_mel(self, tag: str, mel: torch.Tensor, step: int):
        pass
