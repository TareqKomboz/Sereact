import os
import torch
from torch.utils.data import DataLoader
from dataset import BBox3DDataset
from visualize import plot_result, plot_comparison
from utils import setup_run_dir, EarlyStopping
from config import Config
import logging

import torch.nn as nn, torch.optim as optim
from train import train_epoch, test_epoch
from model import BBox3DModel

def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def setup_logging(run_dir):
    log_path = os.path.join(run_dir, "logs", "train.log")
    logging.basicConfig(level=logging.INFO, format="%(message)s", 
                        handlers=[logging.FileHandler(log_path), logging.StreamHandler()])

def get_loaders(path=Config.DATA_ROOT, bs=Config.BATCH_SIZE):
    train_ds = BBox3DDataset(path, split="train")
    val_ds = BBox3DDataset(path, split="val")
    test_ds = BBox3DDataset(path, split="test")
    return (DataLoader(train_ds, batch_size=bs, shuffle=True),
            DataLoader(val_ds, batch_size=bs),
            DataLoader(test_ds, batch_size=bs))

def run_training(model, loaders, opt, sched, early_stop, device, run_dir):
    train_loader, val_loader = loaders[0], loaders[1]
    for e in range(Config.EPOCHS):
        t_loss, t_diou, t_l1 = train_epoch(model, train_loader, opt, device, epoch=e)
        v_loss, v_diou, v_l1 = test_epoch(model, val_loader, device)
        mode = "L1-only" if e < Config.L1_WARMUP_EPOCHS else "Hybrid"
        logging.info(f"Epoch {e:3d} [{mode}] | Train L1: {t_l1:.4f} | Val DIoU: {v_diou:.4f} Val L1: {v_l1:.4f}")
        sched.step(v_loss)
        if early_stop(v_loss):
            torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pth"))
        if early_stop.early_stop: break


def run_final_eval(model, loader, device, run_dir):
    best_path = os.path.join(run_dir, "best_model.pth")
    if os.path.exists(best_path): model.load_state_dict(torch.load(best_path, weights_only=True))
    t_loss, t_diou, t_l1 = test_epoch(model, loader, device)
    logging.info(f"Final Test Result -> DIoU: {t_diou:.4f} | L1: {t_l1:.4f}")
    with open(os.path.join(run_dir, "logs", "test.log"), "w") as f:
        f.write(f"Test Result -> DIoU: {t_diou:.4f} | L1: {t_l1:.4f}\n")
    s = next(iter(loader))
    p = model(s['pc'].to(device), s['obj_pc'].to(device),
              s['mask'].to(device), s['rgb'].to(device))[0].detach().cpu().numpy()
    plot_comparison(s['pc'][0].numpy(), s['bbox'][0].numpy(), p,
                    rgb=s['rgb'][0].numpy(),
                    title=f"Final Test | DIoU={t_diou:.4f} | L1={t_l1:.4f}",
                    save_path=os.path.join(run_dir, "visualizations", "test_prediction.png"))

def main():
    device, run_dir = get_device(), setup_run_dir()
    setup_logging(run_dir)
    loaders = get_loaders()
    model = BBox3DModel().to(device)
    opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=Config.SCHEDULER_PATIENCE, factor=Config.SCHEDULER_FACTOR)
    run_training(model, loaders, opt, sched, EarlyStopping(patience=Config.EARLY_STOPPING_PATIENCE), device, run_dir)
    run_final_eval(model, loaders[2], device, run_dir)
    logging.info(f"Final results saved to {run_dir}")

if __name__ == '__main__':
    main()
