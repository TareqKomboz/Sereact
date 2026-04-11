import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import BBox3DDataset
from model import BBox3DModel
from metrics import hybrid_3d_loss, symmetry_aware_orient_loss
from config import Config

# Setup
device = torch.device("cpu")
ds = BBox3DDataset(Config.DATA_ROOT, split="train")
loader = DataLoader(ds, batch_size=2, shuffle=True)
batch = next(iter(loader))

print("="*60)
print("AUGMENTATION SYNC CHECK")
print("="*60)
print(f"Global PC shape: {batch['pc'].shape}")
print(f"Object PC shape: {batch['obj_pc'].shape}")
# If augmentation worked, obj_pc is no longer just zeros or unrotated blocks.
valid_obj_pts = batch['obj_pc'][batch['valid']].abs().sum()
print(f"Valid Object Points signal: {valid_obj_pts.item():.2f}")

model = BBox3DModel().to(device)
model.eval()

with torch.no_grad():
    # New model returns (center, size, R, conf, log_size)
    p_c, p_s, p_R, p_conf, p_s_log, p_mask = model(batch['pc'], batch['obj_pc'], batch['obj_indices'], batch['rgb'])

print("\n"+"="*60)
print("DIRECT SUPERVISION TEST")
print("="*60)
# hybrid_3d_loss(pred_c, pred_s_log, pred_R, pred_conf, pred_mask, true_corners, true_masks, valid_mask, model)
total, ctr, sz, ori, cnf, msk, iou, rmse = hybrid_3d_loss(
    p_c, p_s_log, p_R, p_conf, p_mask, batch['bbox'], batch['mask'], batch['valid'], model=model
)

print(f"Total:      {total:.4f}")
print(f"Center:     {ctr:.4f}")
print(f"Size:       {sz:.4f}")
print(f"Orient:     {ori:.4f}")
print(f"Conf BCE:   {cnf:.4f}")
print(f"Mask BCE:   {msk:.4f}")
print(f"IoU:        {iou:.4f}")
print(f"RMSE (m):   {rmse:.4f}")

# --- SYMMETRY SANITY CHECK ---
print("\n"+"="*60)
print("SYMMETRY SANITY CHECK (Numerical Safety)")
print("="*60)
R_gt = torch.eye(3).view(1, 1, 3, 3)
R_flip = torch.tensor([[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]]).view(1, 1, 3, 3)
loss_val = symmetry_aware_orient_loss(R_flip, R_gt)
print(f"Loss for 180° Z-flip (should be 0.0): {loss_val.item():.6f}")

# Check for NaNs (using empty slot)
empty_bbox = torch.zeros(1, 1, 8, 3)
empty_valid = torch.zeros(1, 1).bool()
empty_masks = torch.zeros(1, 1, Config.IMG_SIZE[0], Config.IMG_SIZE[1])
# This should NOT crash or return NaN due to our eps=1e-8 safety
_, ctr_v, sz_v, ori_v, _, _, _, _ = hybrid_3d_loss(
    p_c[:1, :1], p_s_log[:1, :1], p_R[:1, :1], p_conf[:1, :1], p_mask[:1, :1],
    empty_bbox, empty_masks, empty_valid, model=model
)
print(f"Empty Slot Size Loss (should NOT be NaN): {sz_v.item()}")
print(f"Empty Slot Orient Loss (should NOT be NaN): {ori_v.item()}")
