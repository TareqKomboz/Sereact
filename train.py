import torch
from metrics import hybrid_3d_loss
from config import Config

def train_epoch(model, dataloader, optimizer, device, scheduler=None):
    """
    Optimizes for multi-component geometric + objectness + mask loss.
    """
    model.train()
    metrics = ['total', 'center', 'size', 'orient', 'conf', 'mask', 'iou', 'rmse']
    history = {k: 0.0 for k in metrics}
    
    for batch in dataloader:
        optimizer.zero_grad()
        pc, obj_pc = batch['pc'].to(device), batch['obj_pc'].to(device)
        obj_indices = batch['obj_indices'].to(device)
        mask, rgb, bbox = batch['mask'].to(device), batch['rgb'].to(device), batch['bbox'].to(device)
        valid = batch['valid'].to(device)
        
        # New model returns (center, size, orient, conf, log_size, mask_pred)
        pred_c, pred_s, pred_orient, pred_conf, pred_s_log, pred_mask = model(pc, obj_pc, obj_indices, rgb)
        
        # Pass model to loss to enable IoU calculation
        losses = hybrid_3d_loss(pred_c, pred_s_log, pred_orient, pred_conf, pred_mask, bbox, mask, valid, model=model)
        
        losses[0].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=Config.CLIP_GRAD)
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        for i, k in enumerate(metrics):
            history[k] += losses[i].item()
            
    n = len(dataloader)
    return {k: v / n for k, v in history.items()}


def test_epoch(model, dataloader, device):
    """Evaluates the model on multi-task metrics."""
    model.eval()
    metrics = ['total', 'center', 'size', 'orient', 'conf', 'mask', 'iou', 'rmse']
    history = {k: 0.0 for k in metrics}
    
    with torch.no_grad():
        for b in dataloader:
            pc, obj_pc = b['pc'].to(device), b['obj_pc'].to(device)
            obj_indices = b['obj_indices'].to(device)
            mask, rgb, bbox = b['mask'].to(device), b['rgb'].to(device), b['bbox'].to(device)
            valid = b['valid'].to(device)
            
            p_c, p_s, p_orient, p_conf, p_s_log, p_mask = model(pc, obj_pc, obj_indices, rgb)
            losses = hybrid_3d_loss(p_c, p_s_log, p_orient, p_conf, p_mask, bbox, mask, valid, model=model)
            
            for i, k in enumerate(metrics):
                history[k] += losses[i].item()
    
    n = len(dataloader)
    return {k: v / n for k, v in history.items()}
