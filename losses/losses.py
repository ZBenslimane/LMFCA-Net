# models/losses.py
import torch
from typing import Dict, Optional

def mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.mean((a - b) ** 2)

def mask_mag(x_2ft: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    x_2ft: [B, 2, F, T] -> magnitude [B, 1, F, T]
    """
    mag = torch.sqrt(torch.clamp(x_2ft[:, 0]**2 + x_2ft[:, 1]**2, min=0.0) + eps)
    return mag.unsqueeze(1)

def reim_to_complex(x_2ft: torch.Tensor) -> torch.Tensor:
    """
    x_2ft: [B, 2, F, T] float -> [B, F, T] complex
    """
    assert x_2ft.dim() == 4 and x_2ft.size(1) == 2, \
        f"Expected [B,2,F,T], got {tuple(x_2ft.shape)}"
    # Move the last-2 axis to the end so last dim==2 for view_as_complex
    z = x_2ft.permute(0, 2, 3, 1).contiguous()  # [B, F, T, 2]
    return torch.view_as_complex(z)

def si_sdr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    est/ref: [B, N] float -> SI-SDR [B] (higher is better)
    """
    assert est.shape == ref.shape and est.dim() == 2, \
        f"SI-SDR expects [B,N] with same shapes; got {est.shape} vs {ref.shape}"
    ref_energy = torch.sum(ref * ref, dim=-1, keepdim=True) + eps
    alpha = torch.sum(est * ref, dim=-1, keepdim=True) / ref_energy
    s_target = alpha * ref
    e_noise  = est - s_target
    num = torch.sum(s_target**2, dim=-1) + eps
    den = torch.sum(e_noise**2,  dim=-1) + eps
    return 10.0 * torch.log10(num / den)

def si_sdr_robust(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Stable SI-SDR: zero-mean both signals and clamp denominators.
    """
    assert est.shape == ref.shape and est.dim() == 2
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)
    ref_energy = (ref * ref).sum(dim=-1, keepdim=True).clamp_min(eps)
    alpha = (est * ref).sum(dim=-1, keepdim=True) / ref_energy
    proj = alpha * ref
    noise = est - proj
    num = (proj * proj).sum(dim=-1).clamp_min(eps)
    den = (noise * noise).sum(dim=-1).clamp_min(eps)
    return 10.0 * torch.log10(num / den)

def lmfca_loss(
    pred_mask_2ft: torch.Tensor,   # [B,2,F,T]
    true_mask_2ft: torch.Tensor,   # [B,2,F,T]
    noisy_ref_2ft: torch.Tensor,   # [B,2,F,T]
    clean_ref_ft: torch.Tensor,    # [B,F,T] complex (used for ISTFT to TD)
    stft_params: Dict[str, int],   # {n_fft, hop_length, win_length, center}
    T_true: Optional[torch.Tensor] = None,  # [B]
    alpha: float = 0.1,
    beta: float  = 0.0001,
) -> Dict[str, torch.Tensor]:

    # 1) Spectral losses on mask (Re/Im + magnitude)
    # Lspec: MSE on re/im
    Lspec = mse(pred_mask_2ft, true_mask_2ft)
    # Lmag: MSE on magnitude of masks
    Lmag = mse(mask_mag(pred_mask_2ft), mask_mag(true_mask_2ft))

    # 2) Time-domain SI-SDR via masked ref channel
    # apply mask on noisy ref in complex domain
    pred_mask_c = reim_to_complex(pred_mask_2ft)     # [B,F,T] complex
    noisy_ref_c = reim_to_complex(noisy_ref_2ft)     # [B,F,T] complex
    est_ref_c   = pred_mask_c * noisy_ref_c          # [B,F,T] complex

    # iSTFT to time domain
    from utils.time_frequency import istft_torch
    n_fft   = stft_params["n_fft"]
    hop     = stft_params["hop_length"]
    win_len = stft_params["win_length"]
    center  = bool(stft_params.get("center", False))

    # --- pick signal length consistent with center flag : https://docs.pytorch.org/docs/stable/generated/torch.istft.html ---
    Tframes = int(est_ref_c.shape[-1])
    if center:
        full_sig_len = hop * (Tframes - 1)
    else:
        full_sig_len = hop * (Tframes - 1) + win_len

    # ISTFT expects [C,F,T] complex; here treat batch as channels (C=B)
    est_td   = istft_torch(est_ref_c,   n_fft=n_fft, hop=hop, win_len=win_len, center=center, length=full_sig_len)  # [B,N]
    clean_td = istft_torch(clean_ref_ft, n_fft=n_fft, hop=hop, win_len=win_len, center=center, length=full_sig_len) # [B,N]

    # --- SI-SDR: ignore padded frames using T_true ---
    si_list = []
    B, Nmax = est_td.shape
    for b in range(B):
        t_true = int(T_true[b].item()) if T_true is not None else Tframes
        if t_true <= 0:
            continue
        if center:
            n_true = hop * (t_true - 1)
        else:
            n_true = hop * (t_true - 1) + win_len
        n_true = max(0, min(n_true, Nmax))
        if n_true > 0:
            si_list.append(si_sdr(est_td[b:b+1, :n_true], clean_td[b:b+1, :n_true]))

    LSISDR = -torch.cat(si_list, dim=0).mean() if si_list else torch.tensor(0.0, device=est_td.device)

    total = alpha * Lmag + (1 - alpha) * Lspec + beta * LSISDR
    return {"total": total, "Lmag": Lmag, "Lspec": Lspec, "LSISDR": LSISDR}
