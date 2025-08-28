# data/dataset_crops.py
import json, numpy as np, torch, random
from torch.utils.data import Dataset
from typing import Dict, Any, List, Tuple, Optional, Union
import yaml

from utils.config import _load_config, _get
from utils.time_frequency import load_complex, complex_to_2ft

# ---------- Dataset ----------
class LmfcaNetDataset(Dataset):
    """
    Multi-crop dataset: each index is a (utterance_idx, start_frame) crop.
    It returns fixed-length STFT segments of length `segment_frames` (from config).

    JSONL per-line fields (already created by your file-list script):
      - "mixture_stft": path to [C,F,T] complex
      - "clean_stft":   path to [C,F,T] complex
      - "cirm":         path to [F,T]   complex (or re/im split; here we use complex)
      - "ref_ch":       int (flattened reference channel)
      - "utt_id":       str (optional but present in your list)

    __getitem__ returns a dict:
      x_in:      [2C, F, Tseg] float32  (Re/Im of all channels concatenated)
      cirm_2ft:  [2,  F, Tseg] float32  (target mask Re/Im)
      noisy_ref: [2,  F, Tseg] float32  (mixture ref Re/Im)
      clean_ref: [F,  Tseg]   complex64 (clean ref STFT for iSTFT/SI-SDR)
      T_true:    int                        (non-padded frames in this crop)
      utt_id:    str
      start:     int                        (start frame of this crop in the utterance)
    """
    def __init__(
        self,
        jsonl_path: str,
        config: Union[str, dict],
        split: str = "train",   # "train" or "valid"
        seed: int = 0,
    ):
        self.items = [json.loads(l) for l in open(jsonl_path, "r", encoding="utf-8")]
        self.cfg = _load_config(config)
        self.split = split
        self.rng = random.Random(seed)

        # ---- dataset hyperparams from config (with sensible defaults) ----
        self.Tseg   = int(_get(self.cfg, ["dataset", "segment_frames"], 184))          # must be divisible by 8
        if split == "train":
            stride_default = self.Tseg // 2  # 50% overlap by default
            self.stride = int(_get(self.cfg, ["dataset", "stride_frames_train"], stride_default))
        else:
            self.stride = int(_get(self.cfg, ["dataset", "stride_frames_valid"], self.Tseg))
        self.min_frames_for_multi = int(_get(self.cfg, ["dataset", "min_frames_for_multi"], 128))
        self.max_crops_per_utt    = _get(self.cfg, ["dataset", "max_crops_per_utt"], None)
        self.use_memmap_shapes    = bool(_get(self.cfg, ["dataset", "use_memmap_shapes"], True))

        # ---- build (utt_idx, start_frame) index for all crops ----
        self.index: List[Tuple[int, int]] = [] # (utt_idx, start_frame)

        for i, it in enumerate(self.items):
            # Inspect frames T from cirm shape (F,T) without loading the whole array
            if self.use_memmap_shapes:
                z = np.load(it["cirm"], mmap_mode="r")
                T = int(z.shape[1])
                del z
            else:
                T = int(np.load(it["cirm"]).shape[1])

            if T >= self.Tseg and T >= self.min_frames_for_multi:
                # stride windows across utterance
                n = 1 + max(0, (T - self.Tseg) // self.stride)
                starts = [s * self.stride for s in range(n)]
                # add a tail window if remainder exists (to cover the very end)
                if T - (starts[-1] + self.Tseg) > 0:
                    starts.append(T - self.Tseg)
                # optional cap
                if self.max_crops_per_utt is not None and len(starts) > int(self.max_crops_per_utt):
                    if self.split == "train":
                        self.rng.shuffle(starts)                # random subset for training
                        starts = starts[: int(self.max_crops_per_utt)]
                    else:
                        starts = sorted(starts[: int(self.max_crops_per_utt)])  # deterministic for valid
            else:
                # too short for multi; single crop at 0 (will be padded)
                starts = [0]

            for s in starts:
                self.index.append((i, int(s)))

        # shuffle crop order for training
        if self.split == "train":
            self.rng.shuffle(self.index)

    def __len__(self) -> int:
        return len(self.index)

    def reseed(self, seed: Optional[int] = None):
        """
        Call once per epoch (optional) to reshuffle the crop order for training.
        """
        if seed is None:
            seed = self.rng.randrange(1 << 30)
        self.rng.seed(seed)
        if self.split == "train":
            self.rng.shuffle(self.index)

    def _crop_or_pad_T(self, Z: torch.Tensor, start: int) -> torch.Tensor:
        """
        Z: [..., T] (complex or float), Crop [start:start+Tseg]; if not enough frames, right-pad with zeros to Tseg.
        """
        T = Z.shape[-1]
        if start + self.Tseg <= T:
            return Z[..., start:start + self.Tseg]
        Zc = Z[..., start:]
        pad = self.Tseg - Zc.shape[-1]
        pad_shape = list(Zc.shape)
        pad_shape[-1] = pad
        return torch.cat([Zc, torch.zeros(pad_shape, dtype=Zc.dtype, device=Zc.device)], dim=-1)

    def __getitem__(self, k: int) -> Dict[str, Any]:
        utt_idx, start = self.index[k]
        it = self.items[utt_idx]

        # load arrays (complex STFTs)
        mix   = load_complex(it["mixture_stft"])   # [C,F,T]
        clean = load_complex(it["clean_stft"])     # [C,F,T]
        cirm  = load_complex(it["cirm"])           # [F,T]
        ref   = int(it["ref_ch"])

        C, F, T = mix.shape
        # how many real (non-padded) frames will this crop contain?
        T_true = min(self.Tseg, max(0, T - start))

        # crop/pad each to exactly Tseg frames
        mix   = self._crop_or_pad_T(mix,   start)  # [C,F,Tseg]
        clean = self._crop_or_pad_T(clean, start)  # [C,F,Tseg]
        cirm  = self._crop_or_pad_T(cirm,  start)  # [F,Tseg]

        # build tensors for model & loss
        mixture_2ft   = complex_to_2ft(mix).reshape(C * 2, F, self.Tseg).contiguous()  # [2C,F,Tseg]
        cirm_2ft     = complex_to_2ft(cirm)                              # [2,F,Tseg]
        mixture_ref_2ft = complex_to_2ft(mix[ref])                          # [2,F,Tseg]
        clean_ref = clean[ref]                                        # [F,Tseg] complex64

        return {
            "mixture_2ft": mixture_2ft,
            "cirm_2ft"   : cirm_2ft,
            "mixture_ref_2ft": mixture_ref_2ft,
            "clean_ref": clean_ref,
            "utt_id": it.get("utt_id", str(utt_idx)),
            "T_true": torch.tensor(T_true, dtype=torch.long),
            "start":  torch.tensor(start,  dtype=torch.long),
        }

# ---------- Collate (everything is already fixed-length; just stack) ----------
def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "mixture_2ft":      torch.stack([b["mixture_2ft"]      for b in batch], dim=0),  # [B,2C,F,Tseg]
        "cirm_2ft":      torch.stack([b["cirm_2ft"]  for b in batch], dim=0),  # [B,2,F,Tseg]
        "mixture_ref_2ft": torch.stack([b["mixture_ref_2ft"] for b in batch], dim=0),  # [B,2,F,Tseg]
        "clean_ref": torch.stack([b["clean_ref"] for b in batch], dim=0),  # [B,F,Tseg] complex
        "utt_id":    [b["utt_id"] for b in batch],
        "T_true":    torch.stack([b["T_true"]    for b in batch], dim=0),  # [B]
        "start":     torch.stack([b["start"]     for b in batch], dim=0),  # [B]
    }
