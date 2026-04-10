import torch
from metrics import hybrid_3d_loss
from config import Config

def train_epoch(model, dataloader, optimizer, device):
    """Optimizes for multi-component geometric + objectness loss."""
    model.train()
    metrics = ['total', 'center', 'size', 'orient', 'conf']
    history = {k: 0.0 for k in metrics}
    
    for batch in dataloader:
        optimizer.zero_grad()
        pc, obj_pc = batch['pc'].to(device), batch['obj_pc'].to(device)
        mask, rgb, bbox = batch['mask'].to(device), batch['rgb'].to(device), batch['bbox'].to(device)
        valid = batch['valid'].to(device)
        
        pred, conf = model(pc, obj_pc, mask, rgb)
        losses = hybrid_3d_loss(pred, conf, bbox, valid)
        
        losses[0].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=Config.CLIP_GRAD)
        optimizer.step()
        
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
            
            p, c = model(pc, obj_pc, mask, rgb)
            losses = hybrid_3d_loss(p, c, bbox, valid)
            
            for i, k in enumerate(metrics):
                history[k] += losses[i].item()
    
    n = len(dataloader)
    return {k: v / n for k, v in history.items()}
