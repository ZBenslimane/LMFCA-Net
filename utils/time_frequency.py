# utils/time_frequency.py
import torch
import numpy as np
import soundfile as sf
from typing import Optional, Dict, Tuple

# ---------- STFT / ISTFT ----------
def stft_torch(wav_cn: torch.Tensor, *, n_fft: int, hop: int, win_len: int,
               center: bool, device: Optional[str] = None) -> torch.Tensor:
    """
    wav_cn: [C, N] float32
    returns complex STFT: [C, F, T] (torch.complex64)
    """
    # Use symmetric (periodic=False) Hann to satisfy 50% overlap COLA
    window = torch.hann_window(win_len, periodic=True, device=wav_cn.device, dtype=wav_cn.dtype)
    S = torch.stft(wav_cn, n_fft=n_fft, hop_length=hop, win_length=win_len,
                   window=window, center=center, return_complex=True)
    return S

def istft_torch(S_cft: torch.Tensor, *, n_fft: int, hop: int, win_len: int,
                center: bool, length: Optional[int] = None) -> torch.Tensor:
    """
    S_cft: [C, F, T] complex
    returns time-domain [C, N] float32

    If length is None, infer a consistent length from (hop, win_len, T).
    """

    # Use symmetric (periodic=False) Hann to satisfy 50% overlap COLA
    window = torch.hann_window(win_len, periodic=True, device=S_cft.device, dtype=torch.float32)
    # If center=False and no length given, infer the exact time-domain length
    if not center and length is None:
        Tframes = int(S_cft.shape[-1]) #  number of STFT frames
        length = hop * (Tframes - 1) + win_len # Natural segment length (works for center=False and avoids OLA errors)

    y = torch.istft(S_cft, n_fft=n_fft, hop_length=hop, win_length=win_len, window=window, center=center, length=length)

    return y 


# ---------- Conversions ----------
def load_complex(path: str) -> torch.Tensor:
    """Load a numpy .npy complex64 array as torch.complex64."""
    z = np.load(path)  # saved as complex64
    return torch.from_numpy(z).to(torch.complex64)

def complex_to_2ft(z: torch.Tensor) -> torch.Tensor:
    """
    [..., F, T] complex -> [..., 2, F, T] float32 (split into real/imag).
    """
    return torch.view_as_real(z).movedim(-1, -3).contiguous().to(torch.float32)

# ---------- Shapes for LMFCA ----------
def complex_to_reim(z: torch.Tensor) -> torch.Tensor:
    """
    z: [..., F, T] complex -> [..., 2, F, T] float32
    """
    return torch.view_as_real(z).movedim(-1, -3).float()

def reim_to_complex(x_2ft: torch.Tensor) -> torch.Tensor:
    """
    x_2ft: [..., 2, F, T] float -> [..., F, T] complex
    """
    return torch.view_as_complex(x_2ft.movedim(-3, -1).contiguous())

# ---------- for LMFCA Training ----------
def build_stft_params(cfg: dict) -> dict:
    s = cfg.get("stft", {})
    # accept either n_fft/hop_length or fft_len/fft_hop naming
    n_fft = s.get("n_fft", s.get("fft_len"))
    hop   = s.get("hop_length", s.get("fft_hop"))
    win   = s.get("win_length", n_fft)
    center = bool(s.get("center", False))
    if n_fft is None or hop is None:
        raise ValueError("Config.stft must define n_fft/fft_len and hop_length/fft_hop.")
    return {"n_fft": int(n_fft), "hop_length": int(hop), "win_length": int(win), "center": center}
