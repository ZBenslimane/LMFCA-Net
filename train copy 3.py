"""Training pipeline"""
import os
import torch
import time
import argparse
import yaml  
import numpy as np
import wandb

from utils.config import _load_config
from utils.callbacks import SaveAndStop
from utils.time_frequency import build_stft_params
from losses.losses import lmfca_loss
from torch.utils.data import DataLoader, Subset
from dataset.dataset import LmfcaNetDataset, collate
from model.lmfca_net import lmfcaNet

from torch.optim.lr_scheduler import ReduceLROnPlateau

def to_device_batch(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out

# -------------------------
# 1. Parse Arguments
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Training Pipeline for CFNN")
    parser.add_argument("--files_to_load", type=str, required=True, help="Path to file lists directory")
    parser.add_argument("--config_yaml", type=str, required=True, help="Path to YAML config file")
    return parser.parse_args()

def compute_total_loss(loss_dict: dict) -> torch.Tensor:
    # Use explicit "total" if your loss returns it; otherwise sum components
    return loss_dict.get("total", sum(loss_dict.values()))

def train_one_batch(model, batch, optimizer, stft_params, alpha, beta, device) -> float:
    model.train()
    batch = to_device_batch(batch, device)
    pred_mask = model(batch["mixture_2ft"]) # [B,2,F,T]
    losses = lmfca_loss(
        pred_mask, batch["cirm_2ft"], batch["mixture_ref_2ft"], batch["clean_ref"],
        T_true=batch["T_true"], stft_params=stft_params, alpha=alpha, beta=beta
    )
    loss = compute_total_loss(losses)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())

@torch.no_grad()
def eval_one_batch(model, batch,stft_params, alpha, beta, device) -> float:
    model.eval()
    batch = to_device_batch(batch, device)
    pred_mask = model(batch["mixture_2ft"])
    losses = lmfca_loss(
        pred_mask, batch["cirm_2ft"], batch["mixture_ref_2ft"], batch["clean_ref"],
        T_true=batch["T_true"], stft_params=stft_params, alpha=alpha, beta=beta
    )
    return float(compute_total_loss(losses).detach().cpu())


# -------------------------
# 2. Training Function
# -------------------------
def train(args):
    """Main training pipeline"""
    cfg = _load_config(args.config_yaml)  # Load YAML config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.get("seed", 0))
    np.random.seed(cfg.get("seed", 0))

    # Parameters from YAML
    n_files = cfg["n_files"]
    n_epochs = int(cfg.get("n_epochs", 50))
    saver    = SaveAndStop(patience=int(cfg.get("early_stop_patience", n_epochs)), mode='min')
    save_dir = cfg.get("model_save_path", "./checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    # Initialize WandB
    # wandb.init(
    #     project=cfg["wandb"]["project"],
    #     config=cfg,  # Log full YAML config
    # )

    # Load train/valid file lists from the provided path
    train_list = os.path.join(args.files_to_load, "train.jsonl")
    valid_list = os.path.join(args.files_to_load, "valid.jsonl")

    # Datasets & DataLoader
    train_dataset = LmfcaNetDataset(train_list,cfg, split="train", seed=cfg.get("seed", 0))
    valid_dataset = LmfcaNetDataset(valid_list,cfg, split="valid", seed=cfg.get("seed", 0))

    # Optional: limit training size using Subset (instead of slicing strings)
    n_files = cfg.get("n_files", None)
    if isinstance(n_files, int) and n_files > 0:
        train_dataset = Subset(train_dataset, range(min(n_files, len(train_dataset))))
        valid_dataset = Subset(valid_dataset, range(min(n_files, len(valid_dataset))))

    # after: config = load_config(args.config_yaml)
    batch_size   = int(cfg.get("batch_size", 8))
    num_workers  = int(cfg.get("num_workers", 4))


    train_loader = DataLoader( train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers, collate_fn=collate, pin_memory=True)
    valid_loader = DataLoader( valid_dataset, batch_size=batch_size, shuffle=False, num_workers=max(0, num_workers // 2), collate_fn=collate, pin_memory=True)

    # 2) Peek one batch to infer in_ch and verify shapes
    batch0 = next(iter(train_loader))
    # mixture_2ft shape = [B, 2C, F, T]
    in_ch = int(batch0["mixture_2ft"].shape[1])
    Freq     = int(batch0["mixture_2ft"].shape[2])
    Tseg  = int(batch0["mixture_2ft"].shape[3])

    print(f"[Sanity] in_ch={in_ch} (2*C), F={Freq}, Tseg={Tseg}. "
          f"Batch sizes: mixture_2ft={tuple(batch0['mixture_2ft'].shape)}, "
          f"cirm_2ft={tuple(batch0['cirm_2ft'].shape)}")
    

    # 3) Build model
    model = lmfcaNet(in_ch=in_ch, out_ch=2).to(device)
    model.eval()  # just for the sanity pass

    # --- 7) Optimizer & Scheduler ---
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",      # lower val loss is better
        factor=0.5,      # halve LR
        patience=5,      # after 5 epochs with no improvement
        verbose=True
    )

    # 4) Build STFT params for the loss
    stft_params = build_stft_params(cfg)
    alpha = float(cfg.get("alpha", 0.1))
    beta  = float(cfg.get("beta", 0.001))

    n_fft   = stft_params["n_fft"]
    hop     = stft_params["hop_length"]
    win_len = stft_params["win_length"]

    assert 0 < hop <= win_len, f"Invalid STFT: hop={hop}, win_len={win_len}"
    assert (n_fft // 2 + 1) == batch0["mixture_2ft"].shape[2], \
        f"F from data ({batch0['mixture_2ft'].shape[2]}) != n_fft//2+1 ({n_fft//2+1})"
    
    # 5) Run one forward + loss (no grad for this sanity check)
    # with torch.no_grad():
    #     b = to_device_batch(batch0, device)
    #     pred_mask = model(b["mixture_2ft"])  # [B,2,F,T]

    #     # Compute the per-component losses (returns dict)
    #     losses = lmfca_loss(
    #         pred_mask, b["cirm_2ft"], b["mixture_ref_2ft"], b["clean_ref"], T_true=b["T_true"],
    #         stft_params=stft_params, alpha=alpha, beta=beta
    #     )
    # loss_print = {k: float(v.detach().cpu()) for k, v in losses.items()}
    # print("[Sanity] Loss components:", loss_print)
    # print("Step 1 OK ✅ — data ↔ model ↔ loss are wired correctly.")


    # --- 8) Full training loop ---
    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        train_loss = 0
        val_loss = 0

        # ---- Train epoch ----
        model.train()
        for batch in train_loader:
            train_loss += train_one_batch(
                model, batch, optimizer,
                stft_params=stft_params, alpha=alpha, beta=beta,
                device=device
            )

        # ---- Validate epoch ----
        model.eval()
        with torch.no_grad():   # Gradient will not be computed
            for batch in valid_loader:
                val_loss += eval_one_batch(
                    model, batch, stft_params=stft_params, alpha=alpha, beta=beta,
                    device = device
                )
                
        avg_train_loss = train_loss / max(1, len(train_loader))
        avg_val_loss = val_loss / max(1, len(valid_loader))

        # --- Step the scheduler with the validation metric ---
        if scheduler is not None:
            scheduler.step(avg_val_loss)

        # save / early stop
        if saver.save_model_query(avg_val_loss):
            ckpt_path = os.path.join(save_dir, "lmfca_best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": cfg,
                "metrics": {"train": avg_train_loss, "val": avg_val_loss},
            }, ckpt_path)
            print(f"✓ saved best -> {ckpt_path}")

        if saver.early_stop_query():
            print("early stop triggered")
            break

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d}/{n_epochs} | "
            f"train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} | "
            f"lr={current_lr:.2e} | {time.time()-t0:.1f}s"
        )

# -------------------------
# 4. Run Training
# -------------------------
if __name__ == "__main__":
    args = parse_args()
    train(args)