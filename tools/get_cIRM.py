import argparse, os, json, numpy as np
from typing import Tuple, Dict, List
import torch
import yaml
from tqdm import tqdm

def compute_complex_IRM(target_ref: torch.Tensor, mix_ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
        Compute the Complex Ideal Ratio Mask (cIRM) for a single reference channel.

    Args:
        target_ref: Complex STFT tensor of shape [F, T] (or [..., F, T]),
                   containing the clean reference channel.
        mix_ref:   Complex STFT tensor with the same shape as `clean_ref`,
                   containing the mixture reference channel.
        eps:       Small constant added to the denominator for numerical
                   stability to avoid division by zero.
    Returns:
        torch.FloatTensor of shape [2, F, T] (or [..., 2, F, T]):
            - index 0: real part of the cIRM
            - index 1: imaginary part of the cIRM
    Notes:
        - This function stays in torch. If you need a NumPy array for saving, call `.cpu().numpy()` on the returned tensor.
        - Inputs must be complex dtype (e.g., torch.cfloat/torch.cdouble).
    """
    cr, ci = target_ref.real, target_ref.imag
    xr, xi = mix_ref.real,  mix_ref.imag
    denom = (xr*xr + xi*xi).clamp_min(eps)
    re = (cr*xr + ci*xi) / denom
    im = (ci*xr - cr*xi) / denom
    return torch.stack([re, im], dim=0).to(torch.float32)


def get_cIRM(sample_dir, ref_mic_ind):
    """Computes and saves complex Ideal Ratio Masks for a single sample.

    Args:
        sample_dir (str): Path to the sample directory containing STFT `.npy` files.
        ref_mic_ind (int): Index of the reference microphone to extract the IRM.
    """
    # Define expected file paths
    stft_mixture_path  = os.path.join(sample_dir, "stft_mixture.npy")
    stft_target_path  = os.path.join(sample_dir, "stft_speech.npy")

    # Check if all files exist
    if not (os.path.exists(stft_target_path) and os.path.exists(stft_mixture_path)):
        print(f"Warning: Missing STFT files in {sample_dir}, skipping...")
        return

    # Load STFTs
    stft_mixture  = torch.tensor(np.load(stft_mixture_path), dtype=torch.cfloat) # torch.Size([ch=4, Freq, T])
    stft_target  = torch.tensor(np.load(stft_target_path), dtype=torch.cfloat)

    # Extract the reference microphone index
    try:
        stft_target_ref = stft_target[ref_mic_ind, :, :]
        stft_mixture_ref = stft_mixture[ref_mic_ind, :, :]
    except IndexError:
        print(f"Error: ref_mic_ind {ref_mic_ind} is out of bounds for shape {stft_target.shape}")
        return

    # Compute Complex Ideal Ratio Mask (cIRM)
    cirm = compute_complex_IRM(stft_target_ref, stft_mixture_ref) # compute cIRM as [2,F,T] float32

    # convert to complex for saving
    cirm_complex = torch.view_as_complex(cirm.permute(1,2,0).contiguous())  # [F,T,2] -> complex [F,T]
    cirm_complex_np = cirm_complex.cpu().numpy().astype(np.complex64)

    # Save the cIRM
    np.save(os.path.join(sample_dir, "cirm.npy"), cirm_complex_np)


def process_all_samples(data_dir, config_file):
    """Iterates over all samples in the dataset and computes cIRM masks.

    Args:
        data_dir (str): Root directory containing train/ and val/ sample directories.
        config_file (str): Path to the YAML configuration file.
    """
    # Load configuration
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    ref_mic_ind = config.get("ref_mic_ind", 0)  # Default to 0 if not found

    for split in ["train", "valid"]:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            print(f"Warning: {split_dir} not found. Skipping.")
            continue

        # Get all sample directories
        sample_folders = [os.path.join(split_dir, s) for s in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, s))]
        
        print(f"Processing {len(sample_folders)} samples in {split}... (Using ref_mic_ind={ref_mic_ind})")

        # Compute cIRM for each sample
        for sample_dir in tqdm(sample_folders, desc=f"Computing complex cIRM for {split}", unit="sample"):
            get_cIRM(sample_dir, ref_mic_ind)

    print("\ncIRM computation completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Complex Ideal Ratio Masks from STFT files using a reference microphone index.")
    parser.add_argument("--data_dir", required=True, help="Path to dataset containing train/ and valid/ subdirectories.")
    parser.add_argument("--config_file", required=True, help="Path to the YAML config file.")
    args = parser.parse_args()

    process_all_samples(args.data_dir, args.config_file)
