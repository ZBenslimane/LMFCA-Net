"""Training pipeline"""
import os
import torch
import time
import argparse
import yaml  
import numpy as np
import wandb

from utils.config import load_config
from utils.callbacks import SaveAndStop
from utils.time_frequency import build_stft_params
from torch.utils.data import DataLoader, Subset
from dataset.dataset import LmfcaNetDataset, collate
from model import lmfca_net

# -------------------------
# 1. Parse Arguments
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Training Pipeline for CFNN")
    parser.add_argument("--files_to_load", type=str, required=True, help="Path to file lists directory")
    parser.add_argument("--config_yaml", type=str, required=True, help="Path to YAML config file")
    return parser.parse_args()

# -------------------------
# 2. Training Function
# -------------------------
def train(args):
    """Main training pipeline"""
    cfg = load_config(args.config_yaml)  # Load YAML config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.get("seed", 0))
    np.random.seed(cfg.get("seed", 0))

    # Initialize WandB
    wandb.init(
        project=cfg["wandb"]["project"],
        config=cfg,  # Log full YAML config
    )

    # Parameters from YAML
    n_files = cfg["n_files"]
    n_epochs = cfg["n_epochs"]
    save_path = cfg["model_save_path"]
    output_frames = cfg["model_config"]["output_frames"]

    # Load train/valid file lists from the provided path
    train_list = os.path.join(args.files_to_load, "train.jsonl")
    valid_list = os.path.join(args.files_to_load, "valid.jsonl")

    # Datasets & DataLoader
    train_dataset = LmfcaNetDataset(train_list,cfg, split="train", seed=cfg.get("seed", 0))
    valid_dataset = LmfcaNetDataset(valid_list,cfg, split="valid", seed=cfg.get("seed", 0))

    # Optional: limit training size using Subset (instead of slicing strings)
    n_files = cfg.get("n_files", None)
    if isinstance(n_files, int) and n_files > 0:
        train_ds = Subset(train_ds, range(min(n_files, len(train_ds))))

    train_loader = DataLoader( train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers, collate_fn=collate, pin_memory=True)
    valid_loader = DataLoader( valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=max(0, args.num_workers // 2), collate_fn=collate, pin_memory=True)

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
    model = lmfca_net(in_ch=in_ch, out_ch=2).to(device)

    # 4) Build STFT params for the loss
    stft_params = build_stft_params(cfg)
    alpha = float(cfg.get("alpha", 0.3))
    beta  = float(cfg.get("beta", 0.2))

    # Start afresh if no pretraining
    train_losses, val_losses = {}, {}
    first_epoch, last_epoch = 0, n_epochs

    train_callback = SaveAndStop(patience=n_epochs, mode='min')
    os.makedirs(save_path, exist_ok=True)

    print('start training...')
    for i_epoch in range(first_epoch, last_epoch):
        train_loss = 0.0
        val_loss = 0.0

        # (optional) reshuffle multi-crop windows each epoch
        if hasattr(train_loader.dataset, "reseed"):
            train_loader.dataset.reseed(i_epoch)

        # ---- train ----
        for  in enumerate(train_loader):
            train_loss += train_one_batch(model, 
                                          )
            
        # ---- validate ----
        with torch.no_grad():   # Gradient will not be computed; Not stored in the graph either.
            for  in enumerate(valid_loader):
                val_loss += eval_one_batch(model,
                                           )
                
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(valid_loader)

        train_losses[i_epoch] = avg_train_loss
        val_losses[i_epoch] = avg_val_loss

        # Log to WandB
        wandb.log({
            "epoch": i_epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "learning_rate": optimizer.param_groups[0]["lr"]
        })

        print(f"Epoch {i_epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # Save model and optimizer
        torch.save({'train_loss': train_losses, 'val_loss': val_losses},
                   save_path + '{}_losses.pt'.format(rnd_string))    # Save the losses even if no improvement
        
        if train_callback.save_model_query(val_losses[i_epoch]):
            print('Saving ' + rnd_string)
            model_file = os.path.join(save_path, f"{rnd_string}_model.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_losses,
                'val_loss': val_losses,
            }, model_file)

                    # ---- Export to ONNX using a CPU CLONE (keeps optimizer state on GPU) ----
            import copy
            model_cpu = copy.deepcopy(model).to("cpu").eval()

            # Infer shapes correctly
            # in_ch = input channels (e.g., 2*C); F = n_fft//2 + 1; T = segment_frames
            in_ch = next(iter(train_loader))[0].shape[1]  # [B, in_ch, F, T]
            n_fft = cfg["stft"]["n_fft"]
            F = n_fft // 2 + 1
            T = cfg["dataset"]["segment_frames"]
            B = cfg.get("batch_size", 1)

            dummy_input = torch.randn(B, in_ch, F, T, dtype=torch.float32)

            onnx_path = os.path.join(save_path, f"{rnd_string}.onnx")
            torch.onnx.export(
                model_cpu,
                dummy_input,
                onnx_path,
                opset_version=12,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["mask_2ft"],
                dynamic_axes={"input": {0: "batch", 3: "time"}, "mask_2ft": {0: "batch", 3: "time"}},
            )
            
        if train_callback.early_stop_query():
            break

    print("Model Done Trianing :D")
    
# -------------------------
# 4. Run Training
# -------------------------
if __name__ == "__main__":
    args = parse_args()
    train(args)