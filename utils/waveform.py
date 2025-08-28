# utils/waveform.py
import torch
import numpy as np
import soundfile as sf
from typing import Optional, Dict, Tuple

# ---------- I/O ----------
def load_wav_torch(path: str, device: str = "cpu") -> torch.Tensor:
    """
    Load multichannel WAV -> torch.float32 [C, N] on device.
    """
    y, sr = sf.read(path, always_2d=True, dtype="float32")  # [N, C]
    return torch.from_numpy(y.T).to(device)                 # [C, N]

def apply_permutation_torch(x: torch.Tensor, perm: Optional[list]) -> torch.Tensor:
    """
    x: [C, N], perm: list/1D tensor of indices (or None).
    """
    if perm is None:
        return x
    idx = torch.as_tensor(perm, dtype=torch.long, device=x.device)
    return x.index_select(dim=0, index=idx)

def normalize_joint_torch(signals: Dict[str, Optional[torch.Tensor]],
                          peak: float = 0.99, eps: float = 1e-8) -> Dict[str, Optional[torch.Tensor]]:
    """
    One global gain (same scalar) across signals to avoid clipping.
    signals: mapping {name: [C,N] float32}. Returns new dict.
    """
    max_abs = 0.0
    for v in signals.values():
        if v is None: 
            continue
        if v.numel() == 0: 
            continue
        max_abs = max(max_abs, float(v.abs().max().item() + eps))
    if max_abs == 0.0:
        return signals
    gain = peak / max_abs
    return {k: (v * gain if v is not None else None) for k, v in signals.items()}