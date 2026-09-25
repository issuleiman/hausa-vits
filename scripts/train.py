#!/usr/bin/env python3
import sys
import os
import argparse
import yaml
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hausa_tts.training.trainer import Trainer
from hausa_tts.data.dataset import HausaTTSDataset
from hausa_tts.data.collate import HausaCollate
from hausa_tts.data.audio import AudioProcessor
from hausa_tts.text.tokenizer import HausaTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--log_dir', type=str, required=True, help='Directory to save logs and checkpoints')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    config['log_dir'] = args.log_dir

    if 'WORLD_SIZE' in os.environ:
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        dist.init_process_group('nccl', init_method='env://', world_size=world_size, rank=rank)
    else:
        world_size = 1
        rank = 0
        
    torch.cuda.set_device(args.local_rank)

    # Init dataset
    processor = AudioProcessor(
        sample_rate=config['audio']['sample_rate'],
        n_fft=config['audio']['n_fft'],
        hop_length=config['audio']['hop_length'],
        win_length=config['audio']['win_length'],
        n_mels=config['audio']['n_mels']
    )
    tokenizer = HausaTokenizer(use_tone_marks=config['data']['use_tone_marks'])
    
    dataset = HausaTTSDataset(
        metadata_path=config['data']['metadata_path'],
        audio_dir=config['data']['audio_dir'],
        audio_processor=processor,
        tokenizer=tokenizer,
        use_tone_marks=config['data']['use_tone_marks']
    )
    
    sampler = DistributedSampler(dataset) if world_size > 1 else None
    collate_fn = HausaCollate()
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=config['data']['num_workers'],
        collate_fn=collate_fn
    )

    trainer = Trainer(config, rank=args.local_rank, world_size=world_size)
    trainer.train(dataloader)

if __name__ == '__main__':
    main()
