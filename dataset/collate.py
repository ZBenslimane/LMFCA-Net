# data/collate.py
import math, torch
from typing import List, Dict, Any

def pad_to_multiple(x: torch.Tensor, multiple: int, dim: int=-1, value: float=0.0) -> torch.Tensor:
    t = x.shape[dim]
    pad = (multiple - (t % multiple)) % multiple
    if pad == 0:
        return x
    pad_shape = list(x.shape)
    pad_shape[dim] = pad
    pad_tensor = torch.zeros(pad_shape, dtype=x.dtype, device=x.device) + value
    return torch.cat([x, pad_tensor], dim=dim)

def collate_pad8(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Find max T, then ceil to multiple of 8
    T_list = [b["x_in"].shape[-1] for b in batch]
    T_max  = max(T_list)
    T_pad  = int(math.ceil(T_max / 8.0) * 8)

    xs, ys, xr, cr, utts, Ts = [], [], [], [], [], []
    for b in batch:
        xs.append(pad_to_multiple(b["x_in"],      8, dim=-1))  # [2C,F,Tp]
        ys.append(pad_to_multiple(b["cirm_2ft"],  8, dim=-1))  # [2,F,Tp]
        xr.append(pad_to_multiple(b["noisy_ref"], 8, dim=-1))  # [2,F,Tp]
        cr.append(pad_to_multiple(b["clean_ref"], 8, dim=-1))  # [F,Tp] complex
        utts.append(b["utt_id"])
        Ts.append(b["x_in"].shape[-1])  # original length

    X  = torch.stack(xs, dim=0)   # [B, 2C, F, Tpad]
    Y  = torch.stack(ys, dim=0)   # [B, 2,  F, Tpad]
    XR = torch.stack(xr, dim=0)   # [B, 2,  F, Tpad]
    CR = torch.stack(cr, dim=0)   # [B, F,  Tpad] complex

    return {
        "x_in": X,
        "cirm_2ft": Y,
        "noisy_ref": XR,
        "clean_ref": CR,
        "utt_id": utts,
        "T_orig": torch.tensor(Ts, dtype=torch.long),
    }
