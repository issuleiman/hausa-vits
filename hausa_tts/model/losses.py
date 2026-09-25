import torch
import torch.nn.functional as F

def generator_loss(disc_outputs: list) -> torch.Tensor:
    """Generator adversarial loss (hinge or MSE).
    Args:
        disc_outputs: list of discriminator outputs for generated audio
    Returns:
        loss: scalar tensor
    """
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        # Using MSE-based GAN loss as requested
        l = torch.mean((1 - dg)**2)
        gen_losses.append(l)
        loss += l
    return loss, gen_losses

def discriminator_loss(disc_real_outputs: list, 
                       disc_generated_outputs: list) -> tuple:
    """Discriminator loss.
    Args:
        disc_real_outputs: list of discriminator outputs for real audio
        disc_generated_outputs: list of discriminator outputs for generated audio  
    Returns:
        (loss, real_losses, generated_losses)
    """
    loss = 0
    real_losses = []
    generated_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        # Using MSE-based GAN loss
        r_loss = torch.mean((1 - dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        real_losses.append(r_loss.item())
        generated_losses.append(g_loss.item())
    return loss, real_losses, generated_losses

def feature_loss(fmap_r: list, fmap_g: list) -> torch.Tensor:
    """Feature matching loss between real and generated feature maps."""
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))
    return loss * 2

def kl_loss(z_p: torch.Tensor, logs_q: torch.Tensor, 
            m_p: torch.Tensor, logs_p: torch.Tensor,
            z_mask: torch.Tensor) -> torch.Tensor:
    """KL divergence loss between posterior and prior.
    KL(q||p) where q = N(z_p, exp(2*logs_q)), p = N(m_p, exp(2*logs_p))
    z_p is sampled from q.
    """
    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p)**2) * torch.exp(-2. * logs_p)
    kl = torch.sum(kl * z_mask)
    loss = kl / torch.sum(z_mask)
    return loss

def mel_reconstruction_loss(mel_pred: torch.Tensor, 
                            mel_target: torch.Tensor) -> torch.Tensor:
    """L1 mel spectrogram reconstruction loss."""
    loss = F.l1_loss(mel_pred, mel_target)
    return loss

def tone_prediction_loss(tone_logits: torch.Tensor, 
                         tone_targets: torch.Tensor,
                         mask: torch.Tensor) -> torch.Tensor:
    """Cross-entropy loss for tone prediction.
    Args:
        tone_logits: (B, n_tones, T) - predicted tone logits
        tone_targets: (B, T) - ground truth tone IDs
        mask: (B, 1, T) - valid position mask
    """
    B, n_tones, T = tone_logits.shape
    tone_logits = tone_logits.transpose(1, 2).reshape(B * T, n_tones)
    tone_targets = tone_targets.reshape(B * T)
    
    loss = F.cross_entropy(tone_logits, tone_targets, reduction='none')
    loss = loss.view(B, 1, T)
    loss = torch.sum(loss * mask) / torch.sum(mask)
    return loss
