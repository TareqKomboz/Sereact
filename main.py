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
from export_onnx import export_to_onnx

PYTORCH_ENABLE_MPS_FALLBACK=1

try:
    import wandb
except ImportError:
    wandb = None


def _config_to_dict():
    cfg = {}
    for key in dir(Config):
        if key.startswith("_"):
            continue
        value = getattr(Config, key)
        if callable(value):
            continue
        if isinstance(value, (str, int, float, bool, list, tuple, dict)) or value is None:
            cfg[key] = value
        else:
            cfg[key] = str(value)
    return cfg


def get_device():
    if torch.cuda.is_available():  return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def setup_logging(run_dir):
    log_path = os.path.join(run_dir, "logs", "train.log")
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[logging.FileHandler(log_path),
                                  logging.StreamHandler()])


def setup_wandb(run_dir, device, n_params):
    if not Config.WANDB_ENABLED:
        return None

    if wandb is None:
        logging.warning("W&B is enabled in config but 'wandb' is not installed. Skipping W&B tracking.")
        return None

    if Config.WANDB_MODE == "disabled":
        return None

    run = wandb.init(
        project=Config.WANDB_PROJECT,
        entity=Config.WANDB_ENTITY,
        name=os.path.basename(run_dir),
        dir=run_dir,
        mode=Config.WANDB_MODE,
        config=_config_to_dict(),
        tags=Config.WANDB_TAGS
    )

    run.config.update({
        "device": str(device),
        "run_dir": run_dir,
        "n_trainable_params": n_params
    }, allow_val_change=True)

    wandb.define_metric("epoch")
    for metric in ["total", "center", "size", "orient", "iou", "rmse"]:
        wandb.define_metric(f"train/{metric}", step_metric="epoch")
        wandb.define_metric(f"val/{metric}", step_metric="epoch")
    wandb.define_metric("lr", step_metric="epoch")
    return run


def get_loaders(path=Config.DATA_ROOT, bs=Config.BATCH_SIZE):
    train_ds = BBox3DDataset(path, split="train")
    val_ds   = BBox3DDataset(path, split="val")
    test_ds  = BBox3DDataset(path, split="test")
    return (DataLoader(train_ds, batch_size=bs, shuffle=True),
            DataLoader(val_ds,   batch_size=bs),
            DataLoader(test_ds,  batch_size=bs))


def run_training(model, loaders, opt, scheduler, early_stop, device, run_dir, wb_run=None):
    """
    Train the model with OneCycleLR scheduler (Aggressive).
    """
    train_loader, val_loader = loaders[0], loaders[1]
    metrics = ['total', 'center', 'size', 'orient', 'iou', 'rmse']
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
            f"orient={t_hist['orient']:.3f} iou={t_hist['iou']:.3f} rmse={t_hist['rmse']:.3f} | "
            f"Val: tot={v_hist['total']:.2f} ctr={v_hist['center']:.3f} sz={v_hist['size']:.3f} "
            f"orient={v_hist['orient']:.3f} iou={v_hist['iou']:.3f} rmse={v_hist['rmse']:.3f}"
        )

        history['epoch'].append(e)
        for m in metrics:
            history[f'train_{m}'].append(t_hist[m])
            history[f'val_{m}'].append(v_hist[m])

        if wb_run is not None:
            wb_payload = {"epoch": e, "lr": current_lr}
            for m in metrics:
                wb_payload[f"train/{m}"] = t_hist[m]
                wb_payload[f"val/{m}"] = v_hist[m]
            wandb.log(wb_payload)
            
        # Check EarlyStopping
        if early_stop(v_hist['total']):
            torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pth"))
            if wb_run is not None:
                wb_run.summary["best_epoch"] = e
                wb_run.summary["best_val_total"] = v_hist["total"]
                wb_run.summary["best_val_iou"] = v_hist["iou"]
                wb_run.summary["best_val_rmse"] = v_hist["rmse"]
        if early_stop.early_stop:
            logging.info(f"Early stopping at epoch {e}.")
            break

    return history


def run_final_eval(model, loader, device, run_dir, wb_run=None):
    best_path = os.path.join(run_dir, "best_model.pth")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, weights_only=True, map_location=device))
    
    hist = test_epoch(model, loader, device)
    res_str = f"IoU: {hist['iou']:.4f} | RMSE: {hist['rmse']:.4f}m | Ctr: {hist['center']:.4f} | Sz: {hist['size']:.4f} | Orient: {hist['orient']:.4f}"
    logging.info(f"Final Test Result -> {res_str}")
    
    with open(os.path.join(run_dir, "logs", "test.log"), "w") as f:
        f.write(f"Test Result -> {res_str}\n")
    
    s_batch = next(iter(loader))
    with torch.no_grad():
        p_c, p_s, p_orient, p_s_log = model(
            s_batch['pc'].to(device),
            s_batch['obj_pc'].to(device),
            s_batch['obj_indices'].to(device),
            s_batch['rgb'].to(device),
            s_batch['mask'].to(device)
        )
        # Reconstruct corners for visualization mapping
        p_corners = model.reconstruct_corners(p_c, p_s, p_orient)
    
    # Save a comparison plot
    vis_path = os.path.join(run_dir, "visualizations", "test_prediction.png")
    plot_comparison(s_batch['pc'][0].numpy(), s_batch['bbox'][0].numpy(), p_corners[0].cpu().numpy(),
                    rgb=s_batch['rgb'][0].numpy(), gt_mask=s_batch['mask'][0].cpu().numpy(),
                    valid_slots=s_batch['valid'][0].cpu().numpy(),
                    pred_s=p_s[0].cpu().numpy(), pred_orient=p_orient[0].cpu().numpy(),
                    title=f"Final Test | {res_str}",
                    save_path=vis_path)

    if wb_run is not None:
        wandb.log({f"test/{k}": v for k, v in hist.items()})
        if os.path.exists(vis_path):
            wandb.log({"visualizations/test_prediction": wandb.Image(vis_path)})
        wb_run.summary["final_test_iou"] = hist["iou"]
        wb_run.summary["final_test_rmse"] = hist["rmse"]
        wb_run.summary["final_test_total"] = hist["total"]


def export_best_model_to_onnx(run_dir, wb_run=None):
    if not Config.ONNX_EXPORT:
        return None

    best_path = os.path.join(run_dir, "best_model.pth")
    onnx_path = os.path.join(run_dir, "model.onnx")
    if not os.path.exists(best_path):
        logging.warning(f"Skipping ONNX export: checkpoint not found at {best_path}")
        return None

    try:
        export_to_onnx(best_path, onnx_path, device="cpu", opset_version=Config.ONNX_OPSET_VERSION)
        logging.info(f"ONNX export complete: {onnx_path}")
        if wb_run is not None and os.path.exists(onnx_path):
            wb_run.summary["onnx_path"] = onnx_path
        return onnx_path
    except Exception as exc:
        logging.exception(f"ONNX export failed: {exc}")
        return None


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
    wb_run = setup_wandb(run_dir, device, n_params)
    if wb_run is not None and Config.WANDB_WATCH_MODEL:
        wandb.watch(model, log="all", log_freq=Config.WANDB_WATCH_LOG_FREQ)

    try:
        loaders = get_loaders()
        
        # Optimizer for heads (Backbone is frozen via Config.FREEZE_BACKBONE)
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
                               device, run_dir, wb_run=wb_run)

        run_final_eval(model, loaders[2], device, run_dir, wb_run=wb_run)
        export_best_model_to_onnx(run_dir, wb_run=wb_run)

        # Final summary and curve plotting
        total_s  = time.time() - t_start
        h, m, s = int(total_s // 3600), int((total_s % 3600) // 60), int(total_s % 60)
        logging.info("-" * 110)
        logging.info(f"Training complete — {len(history['epoch'])} epochs in {h:02d}h {m:02d}m {s:02d}s")

        curves_path = os.path.join(run_dir, "visualizations", "training_curves.png")
        plot_training_curves(history, save_path=curves_path, title=f"Run: {os.path.basename(run_dir)}")
        logging.info(f"All results saved to {run_dir}")

        if wb_run is not None:
            if os.path.exists(curves_path):
                wandb.log({"visualizations/training_curves": wandb.Image(curves_path)})
            if Config.WANDB_LOG_ARTIFACTS:
                artifact = wandb.Artifact(f"{os.path.basename(run_dir)}-artifacts", type="experiment")
                artifact_files = [
                    os.path.join(run_dir, "best_model.pth"),
                    os.path.join(run_dir, "model.onnx"),
                    os.path.join(run_dir, "logs", "train.log"),
                    os.path.join(run_dir, "logs", "test.log"),
                    os.path.join(run_dir, "visualizations", "training_curves.png"),
                    os.path.join(run_dir, "visualizations", "test_prediction.png"),
                ]
                for file_path in artifact_files:
                    if os.path.exists(file_path):
                        artifact.add_file(file_path, name=os.path.relpath(file_path, run_dir))
                wb_run.log_artifact(artifact)
    finally:
        if wb_run is not None:
            wandb.finish()


if __name__ == '__main__':
    main()
