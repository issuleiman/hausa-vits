import torch
from torch.optim.lr_scheduler import ExponentialLR, _LRScheduler

def get_scheduler(optimizer, decay_rate: float = 0.999875, last_epoch: int = -1):
    """Get exponential LR scheduler."""
    return ExponentialLR(optimizer, gamma=decay_rate, last_epoch=last_epoch)
