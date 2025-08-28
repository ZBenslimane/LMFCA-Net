#!/usr/bin/env python3
import os, json, argparse, yaml, random

def collect_samples(split_dir):
    """Return a list of (name, dir, mix, clean, cirm) for one split."""
    samples = []
    if not os.path.isdir(split_dir):
        return samples
    for name in sorted(os.listdir(split_dir)):
        d = os.path.join(split_dir, name)
        if not os.path.isdir(d):
            continue
        mix   = os.path.join(d, "stft_mixture.npy")
        clean = os.path.join(d, "stft_speech.npy")
        cirm  = os.path.join(d, "cirm.npy")
        if os.path.exists(mix) and os.path.exists(clean) and os.path.exists(cirm):
            samples.append((name, d, mix, clean, cirm))
    return samples

def write_jsonl(samples, out_path, ref_ch):

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for name, d, mix, clean, cirm in samples:
            rec = {
                "utt_id": name,
                "dir": d,
                "mixture_stft": mix,
                "clean_stft": clean,
                "cirm": cirm,
                "ref_ch": int(ref_ch),  # flattened channel idx after stacking nodes×mics -> C
            }
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote {len(samples)} lines -> {out_path}")

def main():
    ap = argparse.ArgumentParser(description="Create train/valid JSONL file lists for LMFCA-Net.")
    ap.add_argument("--data_dir",   required=True, help="Root with train/ and valid/ subfolders")
    ap.add_argument("--output_dir", required=True, help="Where to write train.jsonl / valid.jsonl")
    ap.add_argument("--config_file",     required=True, help="YAML to read ref mic index")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- read YAML FIRST and resolve ref_ch ----
    with open(args.config_file, "r") as f:
        cfg = yaml.safe_load(f)

    # prefer top-level, else filter.ref_mic_ind, else 0
    ref_ch = int(cfg.get("ref_mic_ind"))

    # ---- collect & write ----
    train_samples = collect_samples(os.path.join(args.data_dir, "train"))
    valid_samples = collect_samples(os.path.join(args.data_dir, "valid"))

    write_jsonl(
        train_samples,
        os.path.join(args.output_dir, "train.jsonl"),
        ref_ch=ref_ch,
    )
    write_jsonl(
        valid_samples,
        os.path.join(args.output_dir, "valid.jsonl"),
        ref_ch=ref_ch,
    )

if __name__ == "__main__":
    main()
