import torch
from metrics import hybrid_3d_loss

def train_epoch(model, dataloader, optimizer, device, scheduler=None):
    """
    Optimizes for multi-component geometric + objectness loss.
    Now supports Direct Supervision via raw Size and Rotation outputs.
    """
    model.train()
    metrics = ['total', 'center', 'size', 'orient', 'conf']
    history = {k: 0.0 for k in metrics}
    
    for batch in dataloader:
        optimizer.zero_grad()
        pc, obj_pc = batch['pc'].to(device), batch['obj_pc'].to(device)
        mask, rgb, bbox = batch['mask'].to(device), batch['rgb'].to(device), batch['bbox'].to(device)
        valid = batch['valid'].to(device)
        
        # New model returns (corners, conf, size, R)
        pred_corners, conf, pred_s, pred_R = model(pc, obj_pc, mask, rgb)
        
        # Pass raw size and R directly to the loss
        losses = hybrid_3d_loss(pred_corners, conf, bbox, valid, 
                                pred_s=pred_s, pred_R=pred_R)
        
        losses[0].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        for i, k in enumerate(metrics):
            history[k] += losses[i].item()
            
    n = len(dataloader)
    return {k: v / n for k, v in history.items()}


def test_epoch(model, dataloader, device):
    """Evaluates the model on pure component metrics."""
    model.eval()
    metrics = ['total', 'center', 'size', 'orient', 'conf']
    history = {k: 0.0 for k in metrics}
    
    with torch.no_grad():
        for b in dataloader:
            pc, obj_pc = b['pc'].to(device), b['obj_pc'].to(device)
            mask, rgb, bbox = b['mask'].to(device), b['rgb'].to(device), b['bbox'].to(device)
            valid = b['valid'].to(device)
            
            p_corn, p_conf, p_s, p_R = model(pc, obj_pc, mask, rgb)
            losses = hybrid_3d_loss(p_corn, p_conf, bbox, valid, 
                                    pred_s=p_s, pred_R=p_R)
            
            for i, k in enumerate(metrics):
                history[k] += losses[i].item()
    
    n = len(dataloader)
    return {k: v / n for k, v in history.items()}
