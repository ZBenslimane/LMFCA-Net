#!/usr/bin/env python3
# tools/preprocess.py

import os
import argparse
import yaml
import numpy as np
import torch
from tqdm import tqdm

from utils.waveform import (
    load_wav_torch,
    apply_permutation_torch,
    normalize_joint_torch,
)
from utils.time_frequency import stft_torch

def process_sample(sample_dir: str, cfg: dict, normalize: bool):
    """Load mixture/target/noise, optional joint normalization, STFT -> save [C,F,T] complex64."""
    mix_p   = os.path.join(sample_dir, "mixture.wav")
    tgt_p   = os.path.join(sample_dir, "target.wav")
    noise_p = os.path.join(sample_dir, "noise.wav")

    for p in (mix_p, tgt_p, noise_p):
        if not os.path.exists(p):
            print(f"[warn] missing {p}; skip {sample_dir}")
            return

    # ---- STFT params (paper defaults unless overridden) ----
    stft_cfg = cfg.get("stft", {})
    n_fft   = int(stft_cfg.get("n_fft", 510))
    hop     = int(stft_cfg.get("hop_length", 255))
    win_len = int(stft_cfg.get("win_length", 510))
    center  = bool(stft_cfg.get("center", False))
    F_expected = n_fft // 2 + 1  # 256 for n_fft=510

    # Optional channel reordering
    perms = cfg.get("PERMUTATIONS", None)

    # 1) Load WAVs -> torch float32 [C, N] (CPU)
    mix   = load_wav_torch(mix_p)
    tgt   = load_wav_torch(tgt_p)
    noise = load_wav_torch(noise_p)

    # 2) Optional permutation
    if perms is not None:
        mix   = apply_permutation_torch(mix, perms)
        tgt   = apply_permutation_torch(tgt, perms)
        noise = apply_permutation_torch(noise, perms)

    # 3) Optional joint peak normalization (config-controlled)
    if normalize:
        sigs = normalize_joint_torch({"mixture": mix, "target": tgt, "noise": noise}, peak=0.99)
        mix, tgt, noise = sigs["mixture"], sigs["target"], sigs["noise"]

    # 4) STFT -> [C, F, T] complex
    with torch.no_grad():
        S_mix   = stft_torch(mix,   n_fft=n_fft, hop=hop, win_len=win_len, center=center)
        S_tgt   = stft_torch(tgt,   n_fft=n_fft, hop=hop, win_len=win_len, center=center)
        S_noise = stft_torch(noise, n_fft=n_fft, hop=hop, win_len=win_len, center=center)

    # 5) Sanity check: F divisible by 8 (expect 256)
    assert S_mix.shape[1] == F_expected, f"Expected F={F_expected}, got {S_mix.shape[1]}"

    # 6) Save as complex64 numpy arrays (shape [C,F,T])
    np.save(os.path.join(sample_dir, "stft_mixture.npy"), S_mix.cpu().numpy().astype(np.complex64))
    np.save(os.path.join(sample_dir, "stft_speech.npy"),  S_tgt.cpu().numpy().astype(np.complex64))
    np.save(os.path.join(sample_dir, "stft_noise.npy"),   S_noise.cpu().numpy().astype(np.complex64))

def main():
    ap = argparse.ArgumentParser(description="Compute STFT features ([C,F,T]) for LMFCA-Net.")
    ap.add_argument("--data_dir", required=True, help="Root with train/ and valid/ subfolders")
    ap.add_argument("--config_file", required=True, help="YAML with STFT params (+ PERMUTATIONS, preprocess.normalize)")
    args = ap.parse_args()

    with open(args.config_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # Read normalization choice from YAML (default True)
    normalize = bool(cfg.get("preprocess", {}).get("normalize", True))

    for split in ("train", "valid"):
        split_dir = os.path.join(args.data_dir, split)
        if not os.path.isdir(split_dir):
            print(f"[warn] {split_dir} missing; skip")
            continue
        sample_dirs = [os.path.join(split_dir, d)
                       for d in os.listdir(split_dir)
                       if os.path.isdir(os.path.join(split_dir, d))]
        print(f"Processing {len(sample_dirs)} samples in {split} (normalize={normalize})...")
        for sd in tqdm(sample_dirs, desc=f"STFT {split}", unit="sample"):
            process_sample(sd, cfg, normalize)
    print("STFT computation completed.")

if __name__ == "__main__":
    main()
