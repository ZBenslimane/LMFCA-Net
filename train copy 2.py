# in train.py
import os, torch, numpy as np
from utils.config import _load_config
from utils.time_frequency import build_stft_params
from dataset.dataset import LmfcaNetDataset, collate
from model.lmfca_net import lmfcaNet
from torch.utils.data import DataLoader, Subset
from train_utils import to_device_batch, train_one_batch, eval_one_batch
from utils.callbacks import SaveAndStop

def train(args):
    cfg = _load_config(args.config_yaml)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.get("seed", 0)); np.random.seed(cfg.get("seed", 0))

    # data
    train_list = os.path.join(args.files_to_load, "train.jsonl")
    valid_list = os.path.join(args.files_to_load, "valid.jsonl")
    train_ds = LmfcaNetDataset(train_list, cfg, split="train", seed=cfg.get("seed", 0))
    valid_ds = LmfcaNetDataset(valid_list, cfg, split="valid", seed=cfg.get("seed", 0))

    n_files = cfg.get("n_files", None)
    if isinstance(n_files, int) and n_files > 0:
        train_ds = Subset(train_ds, range(min(n_files, len(train_ds))))

    bs  = int(cfg.get("batch_size", 8))
    nw  = int(cfg.get("num_workers", 4))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, collate_fn=collate, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=bs, shuffle=False, num_workers=max(0, nw//2), collate_fn=collate, pin_memory=True)

    # peek to infer in_ch
    b0 = next(iter(train_loader))
    in_ch = int(b0["mixture_2ft"].shape[1])

    # model / opt / sched
    model = lmfcaNet(in_ch=in_ch, out_ch=2).to(device)
    lr = float(cfg.get("lr", 2e-4))
    wd = float(cfg.get("weight_decay", 0.0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # optional scheduler
    sched_cfg = cfg.get("scheduler", {"name":"none"})
    if sched_cfg.get("name","none").lower() == "none":
        scheduler = None
    else:
        # example: step scheduler
        step = int(sched_cfg.get("step", 10))
        gamma = float(sched_cfg.get("gamma", 0.5))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=gamma)

    # loss params
    stft_params = build_stft_params(cfg)
    alpha = float(cfg.get("alpha", 0.1))
    beta  = float(cfg.get("beta", 0.001))
    grad_clip = float(cfg.get("grad_clip", 0.0))  # 0 or omit = no clip

    # training loop
    epochs   = int(cfg.get("n_epochs", 50))
    saver    = SaveAndStop(patience=int(cfg.get("early_stop_patience", epochs)), mode='min')
    save_dir = cfg.get("model_save_path", "./checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    tag = cfg.get("run_name", "lmfca_fp32")

    for ep in range(epochs):
        # reshuffle crops if dataset exposes reseed()
        if hasattr(train_loader.dataset, "reseed"):
            train_loader.dataset.reseed(ep)

        # ---- train ----
        tr_tot = tr_Lspec = tr_Lmag = tr_Lsisdr = 0.0
        for batch in train_loader:
            batch = to_device_batch(batch, device)
            loss_dict = train_one_batch(
                model, batch, optimizer,
                stft_params=stft_params,
                alpha=alpha, beta=beta,
                grad_clip=grad_clip if grad_clip > 0 else None,
            )
            tr_tot   += float(loss_dict["total"])
            tr_Lspec += float(loss_dict["Lspec"])
            tr_Lmag  += float(loss_dict["Lmag"])
            tr_Lsisdr+= float(loss_dict["LSISDR"])

        ntr = len(train_loader)
        tr_log = { "train_total": tr_tot/ntr, "train_Lspec": tr_Lspec/ntr, "train_Lmag": tr_Lmag/ntr, "train_LSISDR": tr_Lsisdr/ntr }

        # ---- valid ----
        va_tot = va_Lspec = va_Lmag = va_Lsisdr = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                batch = to_device_batch(batch, device)
                loss_dict = eval_one_batch(model, batch, stft_params, alpha, beta)
                va_tot   += float(loss_dict["total"])
                va_Lspec += float(loss_dict["Lspec"])
                va_Lmag  += float(loss_dict["Lmag"])
                va_Lsisdr+= float(loss_dict["LSISDR"])
        nva = len(valid_loader)
        va_log = { "val_total": va_tot/nva, "val_Lspec": va_Lspec/nva, "val_Lmag": va_Lmag/nva, "val_LSISDR": va_Lsisdr/nva }

        # step scheduler (after epoch)
        if scheduler is not None:
            scheduler.step()

        # logging
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {ep:03d} | lr={lr_now:.2e} | "
              f"train: {tr_log['train_total']:.4f} (spec {tr_log['train_Lspec']:.4f}, mag {tr_log['train_Lmag']:.4f}, si {tr_log['train_LSISDR']:.4f}) | "
              f"val: {va_log['val_total']:.4f} (spec {va_log['val_Lspec']:.4f}, mag {va_log['val_Lmag']:.4f}, si {va_log['val_LSISDR']:.4f})")

        # save / early stop
        if saver.save_model_query(va_log["val_total"]):
            ckpt_path = os.path.join(save_dir, f"{tag}_best.pt")
            torch.save({
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": cfg,
                "metrics": {"train": tr_log, "val": va_log},
            }, ckpt_path)
            print(f"✓ saved best -> {ckpt_path}")

        if saver.early_stop_query():
            print("early stop triggered")
            break

    # final save
    final_path = os.path.join(save_dir, f"{tag}_final.pt")
    torch.save({"epoch": ep, "model_state_dict": model.state_dict(), "config": cfg}, final_path)
    print(f"done. final model -> {final_path}")
