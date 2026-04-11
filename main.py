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


def run_training(model, loaders, opt, scheduler, early_stop, device, run_dir):
    """
    Train the model with OneCycleLR scheduler (Aggressive).
    """
    train_loader, val_loader = loaders[0], loaders[1]
    metrics = ['total', 'center', 'size', 'orient', 'conf']
    history = {f'train_{m}': [] for m in metrics}
    history.update({f'val_{m}': [] for m in metrics})
    history.update({'epoch': []})

    for e in range(Config.EPOCHS):
        # We pass the scheduler to train_epoch for per-batch stepping
        t_hist = train_epoch(model, train_loader, opt, device, scheduler=scheduler)
        v_hist = test_epoch(model, val_loader, device)

        # Get current LR from optimizer for logging
        current_lr = opt.param_groups[0]['lr']

        logging.info(
            f"Epoch {e:3d} | LR: {current_lr:.6f} | "
            f"Tr: tot={t_hist['total']:.3f} ctr={t_hist['center']:.3f} sz={t_hist['size']:.3f} "
            f"or={t_hist['orient']:.3f} cf={t_hist['conf']:.3f} | "
            f"Val: tot={v_hist['total']:.3f} ctr={v_hist['center']:.3f} sz={v_hist['size']:.3f} "
            f"or={v_hist['orient']:.3f} cf={v_hist['conf']:.3f}"
        )

        history['epoch'].append(e)
        for m in metrics:
            history[f'train_{m}'].append(t_hist[m])
            history[f'val_{m}'].append(v_hist[m])

        # For OneCycleLR, we don't call scheduler.step(val_loss) here as it's stepped per batch.
        # But we still check EarlyStopping.
        if early_stop(v_hist['total']):
            torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pth"))
        if early_stop.early_stop:
            logging.info(f"Early stopping at epoch {e}.")
            break

    return history


def run_final_eval(model, loader, device, run_dir):
    best_path = os.path.join(run_dir, "best_model.pth")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, weights_only=True))
    
    hist = test_epoch(model, loader, device)
    res_str = f"Ctr: {hist['center']:.4f} | Sz: {hist['size']:.4f} | Or: {hist['orient']:.4f} | Cf: {hist['conf']:.4f}"
    logging.info(f"Final Test Result -> {res_str}")
    
    with open(os.path.join(run_dir, "logs", "test.log"), "w") as f:
        f.write(f"Test Result -> {res_str}\n")
    
    s_batch = next(iter(loader))
    with torch.no_grad():
        p_c, p_s, p_R, p_conf, p_s_log = model(s_batch['pc'].to(device), s_batch['obj_pc'].to(device),
                                                s_batch['mask'].to(device), s_batch['rgb'].to(device))
        # Reconstruct corners for visualization mapping
        p_corners = model.reconstruct_corners(p_c, p_s, p_R)
    
    # Save a comparison plot
    plot_comparison(s_batch['pc'][0].numpy(), s_batch['bbox'][0].numpy(), p_corners[0].cpu().numpy(),
                    rgb=s_batch['rgb'][0].numpy(), conf=p_conf[0].cpu().numpy(),
                    pred_s=p_s[0].cpu().numpy(), pred_R=p_R[0].cpu().numpy(),
                    title=f"Final Test | {res_str}",
                    save_path=os.path.join(run_dir, "visualizations", "test_prediction.png"))


def main():
    t_start = time.time()
    device, run_dir = get_device(), setup_run_dir()
    setup_logging(run_dir)

    model = BBox3DModel().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Run: {run_dir} | Device: {device} | Params: {n_params:,}")
    logging.info(
        f"Config: PeakLR={Config.LEARNING_RATE}  WD={Config.WEIGHT_DECAY}  "
        f"BS={Config.BATCH_SIZE}  Epochs={Config.EPOCHS}"
    )
    logging.info("-" * 110)

    loaders = get_loaders()
    opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    
    # OneCycleLR setup
    steps_per_epoch = len(loaders[0])
    scheduler = optim.lr_scheduler.OneCycleLR(
        opt, max_lr=Config.LEARNING_RATE,
        steps_per_epoch=steps_per_epoch,
        epochs=Config.EPOCHS,
        pct_start=0.3,
        div_factor=25,
        final_div_factor=1000
    )

    history = run_training(model, loaders, opt, scheduler,
                           EarlyStopping(patience=Config.EARLY_STOPPING_PATIENCE),
                           device, run_dir)

    run_final_eval(model, loaders[2], device, run_dir)

    # Final summary and curve plotting
    total_s  = time.time() - t_start
    h, m, s = int(total_s // 3600), int((total_s % 3600) // 60), int(total_s % 60)
    logging.info("-" * 110)
    logging.info(f"Training complete — {len(history['epoch'])} epochs in {h:02d}h {m:02d}m {s:02d}s")

    curves_path = os.path.join(run_dir, "visualizations", "training_curves.png")
    plot_training_curves(history, save_path=curves_path, title=f"Run: {os.path.basename(run_dir)}")
    logging.info(f"All results saved to {run_dir}")


if __name__ == '__main__':
    main()
