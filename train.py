import torch
from metrics import hybrid_3d_loss
from config import Config

def train_epoch(model, dataloader, optimizer, device, epoch=0):
    """Optimizes for hybrid DIoU + L1 Loss, with pure L1 warmup."""
    model.train()
    total_loss, total_center, total_l1 = 0.0, 0.0, 0.0
    l1_only = epoch < Config.L1_WARMUP_EPOCHS
    for batch in dataloader:
        optimizer.zero_grad()
        pc, obj_pc = batch['pc'].to(device), batch['obj_pc'].to(device)
        mask, rgb, bbox = batch['mask'].to(device), batch['rgb'].to(device), batch['bbox'].to(device)
        valid = batch['valid'].to(device)
        pred = model(pc, obj_pc, mask, rgb)
        loss, center_loss, l1_loss = hybrid_3d_loss(pred, bbox, valid_mask=valid, l1_only=l1_only)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=Config.CLIP_GRAD)
        optimizer.step()
        total_loss   += loss.item()
        total_center += center_loss.item()
        total_l1     += l1_loss.item()
    return total_loss / len(dataloader), total_center / len(dataloader), total_l1 / len(dataloader)

def test_epoch(model, dataloader, device):
    """Evaluates for 3D DIoU Loss and mean L1 Error."""
    model.eval()
    total_loss, total_center, total_l1 = 0.0, 0.0, 0.0
    with torch.no_grad():
        for b in dataloader:
            pc, obj_pc = b['pc'].to(device), b['obj_pc'].to(device)
            mask, rgb, bbox = b['mask'].to(device), b['rgb'].to(device), b['bbox'].to(device)
            valid = b['valid'].to(device)
            p = model(pc, obj_pc, mask, rgb)
            loss, center_loss, l1_loss = hybrid_3d_loss(p, bbox, valid_mask=valid)
            total_loss   += loss.item()
            total_center += center_loss.item()
            total_l1     += l1_loss.item()
    return total_loss / len(dataloader), total_center / len(dataloader), total_l1 / len(dataloader)
