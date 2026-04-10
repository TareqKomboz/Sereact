import os
import time
import torch
from torch.utils.data import DataLoader
from dataset import BBox3DDataset
from visualize import plot_comparison, plot_training_curves
from utils import setup_run_dir, EarlyStopping
from config import Config
import logging

import torch.optim as optim
from train import train_epoch, test_epoch
from model import BBox3DModel


def get_device():
    if torch.cuda.is_available():  return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def setup_logging(run_dir):
    log_path = os.path.join(run_dir, "logs", "train.log")
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[logging.FileHandler(log_path),
                                  logging.StreamHandler()])


def get_loaders(path=Config.DATA_ROOT, bs=Config.BATCH_SIZE):
    train_ds = BBox3DDataset(path, split="train")
    val_ds   = BBox3DDataset(path, split="val")
    test_ds  = BBox3DDataset(path, split="test")
    return (DataLoader(train_ds, batch_size=bs, shuffle=True),
            DataLoader(val_ds,   batch_size=bs),
            DataLoader(test_ds,  batch_size=bs))


def run_training(model, loaders, opt, sched, early_stop, device, run_dir):
    """
    Train the model for up to Config.EPOCHS epochs with early stopping.

    Logs per-epoch metrics + wall-clock time to the console and train.log.
    Returns a history dict suitable for plot_training_curves().
    """
    train_loader, val_loader = loaders[0], loaders[1]
    history = {
        'epoch': [], 'mode': [],
        'train_total': [], 'train_center': [], 'train_l1': [],
        'val_total':   [], 'val_center':   [], 'val_l1':   [],
        'epoch_time_s': [],
    }

    for e in range(Config.EPOCHS):
        t0 = time.time()
        t_loss, t_center, t_l1 = train_epoch(model, train_loader, opt, device, epoch=e)
        v_loss, v_center, v_l1 = test_epoch(model, val_loader, device)
        epoch_s = time.time() - t0

        mode = "L1-only" if e < Config.L1_WARMUP_EPOCHS else "Hybrid"
        logging.info(
            f"Epoch {e:3d} [{mode:7s}] | "
            f"Train: total={t_loss:.4f}  ctr={t_center:.4f}  l1={t_l1:.4f} | "
            f"Val:   total={v_loss:.4f}  ctr={v_center:.4f}  l1={v_l1:.4f} | "
            f"{epoch_s:.1f}s"
        )

        history['epoch'].append(e);         history['mode'].append(mode)
        history['train_total'].append(t_loss);    history['val_total'].append(v_loss)
        history['train_center'].append(t_center); history['val_center'].append(v_center)
        history['train_l1'].append(t_l1);         history['val_l1'].append(v_l1)
        history['epoch_time_s'].append(epoch_s)

        sched.step(v_loss)
        if early_stop(v_loss):
            torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pth"))
        if early_stop.early_stop:
            logging.info(f"Early stopping at epoch {e}.")
            break

    return history


def run_final_eval(model, loader, device, run_dir):
    best_path = os.path.join(run_dir, "best_model.pth")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, weights_only=True))
    t_loss, t_center, t_l1 = test_epoch(model, loader, device)
    logging.info(f"Final Test Result -> Center: {t_center:.4f} | L1: {t_l1:.4f}")
    with open(os.path.join(run_dir, "logs", "test.log"), "w") as f:
        f.write(f"Test Result -> Center: {t_center:.4f} | L1: {t_l1:.4f}\n")
    s = next(iter(loader))
    p = model(s['pc'].to(device), s['obj_pc'].to(device),
              s['mask'].to(device), s['rgb'].to(device))[0].detach().cpu().numpy()
    plot_comparison(s['pc'][0].numpy(), s['bbox'][0].numpy(), p,
                    rgb=s['rgb'][0].numpy(),
                    title=f"Final Test | Center={t_center:.4f} | L1={t_l1:.4f}",
                    save_path=os.path.join(run_dir, "visualizations", "test_prediction.png"))


def main():
    t_start = time.time()
    device, run_dir = get_device(), setup_run_dir()
    setup_logging(run_dir)

    # ── Header info ───────────────────────────────────────────────────────────
    model = BBox3DModel().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Run: {run_dir}")
    logging.info(f"Device: {device}  |  Trainable params: {n_params:,}")
    logging.info(
        f"Config: LR={Config.LEARNING_RATE}  WD={Config.WEIGHT_DECAY}  "
        f"BS={Config.BATCH_SIZE}  AUGMENT={Config.AUGMENT}  "
        f"Epochs={Config.EPOCHS}  Warmup={Config.L1_WARMUP_EPOCHS}"
    )
    logging.info("-" * 90)

    loaders = get_loaders()
    opt   = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                        lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    sched = optim.lr_scheduler.ReduceLROnPlateau(
                opt, 'min',
                patience=Config.SCHEDULER_PATIENCE,
                factor=Config.SCHEDULER_FACTOR)

    history = run_training(
        model, loaders, opt, sched,
        EarlyStopping(patience=Config.EARLY_STOPPING_PATIENCE),
        device, run_dir)

    run_final_eval(model, loaders[2], device, run_dir)

    # ── Timing summary ────────────────────────────────────────────────────────
    total_s  = time.time() - t_start
    n_epochs = len(history['epoch'])
    mean_s   = sum(history['epoch_time_s']) / max(n_epochs, 1)
    h = int(total_s // 3600)
    m = int((total_s % 3600) // 60)
    s = int(total_s % 60)
    logging.info("-" * 90)
    logging.info(
        f"Training complete — {n_epochs} epochs in "
        f"{h:02d}h {m:02d}m {s:02d}s  (mean {mean_s:.1f}s/epoch)"
    )

    # ── Loss curves ───────────────────────────────────────────────────────────
    run_name   = os.path.basename(run_dir)
    curves_path = os.path.join(run_dir, "visualizations", "training_curves.png")
    plot_training_curves(
        history,
        save_path=curves_path,
        title=(f"Training Curves — {run_name}\n"
               f"LR={Config.LEARNING_RATE}  WD={Config.WEIGHT_DECAY}  "
               f"BS={Config.BATCH_SIZE}  AUGMENT={Config.AUGMENT}  "
               f"Epochs={n_epochs}  Device={device}  "
               f"Duration={h:02d}h{m:02d}m{s:02d}s"))

    logging.info(f"All results saved to {run_dir}")


if __name__ == '__main__':
    main()
