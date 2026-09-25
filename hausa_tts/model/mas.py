import torch
import numpy as np
from typing import Optional

@torch.no_grad()
def monotonic_alignment_search(log_p_attn: torch.Tensor, 
                                x_mask: torch.Tensor, 
                                y_mask: torch.Tensor) -> torch.Tensor:
    """Find the most probable monotonic alignment between text and mel frames.
    
    Uses dynamic programming (Viterbi-like) to find optimal path.
    
    Args:
        log_p_attn: (B, T_text, T_mel) - log probability of attention
        x_mask: (B, 1, T_text) - text mask
        y_mask: (B, 1, T_mel) - mel mask
    
    Returns:
        attn: (B, T_text, T_mel) - hard monotonic alignment (binary)
    """
    req_grad = log_p_attn.requires_grad
    if req_grad:
        log_p_attn = log_p_attn.detach()
        
    b, t_x, t_y = log_p_attn.shape
    path = torch.zeros_like(log_p_attn)
    
    log_p_attn = log_p_attn * y_mask.unsqueeze(1).transpose(1, 2) * x_mask.unsqueeze(2).transpose(2, 1)

    for i in range(b):
        max_y = int(y_mask[i].sum())
        max_x = int(x_mask[i].sum())

        v = torch.zeros(max_y, max_x, dtype=log_p_attn.dtype, device=log_p_attn.device)
        v[0, 0] = log_p_attn[i, 0, 0]
        for j in range(1, max_x):
            v[0, j] = -1e4
            
        for k in range(1, max_y):
            v[k, 0] = v[k - 1, 0] + log_p_attn[i, 0, k]
            for j in range(1, max_x):
                v[k, j] = max(v[k - 1, j - 1], v[k - 1, j]) + log_p_attn[i, j, k]
                
        # backtrack
        index = max_x - 1
        for k in range(max_y - 1, -1, -1):
            path[i, index, k] = 1
            if index > 0:
                if v[k - 1, index - 1] > v[k - 1, index]:
                    index -= 1
            if index == 0 and k == 0:
                break
                
    if req_grad:
        path.requires_grad_()
    return path
