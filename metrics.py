"""
metrics.py — Pure component-based OBB loss with Direct Supervision & Sync Augmentation.
"""

import torch
import torch.nn.functional as F
from config import Config

# --- 180-degree rotations around X, Y, Z axes (Klein Four-Group) ---
_BOX_SYMMETRIES = torch.tensor([
    [[1.,  0.,  0.], [ 0.,  1.,  0.], [ 0.,  0.,  1.]],  # Identity
    [[1.,  0.,  0.], [ 0., -1.,  0.], [ 0.,  0., -1.]],  # 180° around X
    [[-1., 0.,  0.], [ 0.,  1.,  0.], [ 0.,  0., -1.]],  # 180° around Y
    [[-1., 0.,  0.], [ 0., -1.,  0.], [ 0.,  0.,  1.]],  # 180° around Z
], dtype=torch.float32)


def _corners_to_obb(corners: torch.Tensor):
    """
    Extract (center, size, R) from 8 corner coordinates with numerical safety.
    Indices: 0(-,-,-), 1(+,-,-), 3(-,+,-), 4(-,-,+)
    """
    center = corners.mean(dim=-2)
    e1 = corners[..., 1, :] - corners[..., 0, :]
    e2 = corners[..., 3, :] - corners[..., 0, :]
    e3 = corners[..., 4, :] - corners[..., 0, :]
    
    # Safe normalization: add epsilon to norm to prevent NaN on empty slots
    sz1 = e1.norm(dim=-1, keepdim=True)
    sz2 = e2.norm(dim=-1, keepdim=True)
    sz3 = e3.norm(dim=-1, keepdim=True)
    
    size = torch.cat([sz1, sz2, sz3], dim=-1)
    
    # R built from columns (the normalized axis directions)
    x = e1 / (sz1 + 1e-8)
    y = e2 / (sz2 + 1e-8)
    z = e3 / (sz3 + 1e-8)

    # Enforce Right-Handedness (fixes Chirality Deadlock from reflection augmentation)
    # A mirrored box is Left-Handed (det=-1), but our model is Right-Handed (SO(3)).
    # We flip the Z-axis if det < 0 to stay in SO(3); the symmetry-aware loss 
    # will handle the 180-deg ambiguous orientation.
    # Determinant of [x, y, z] is the triple product (x cross y) dot z
    x_cross_y = torch.cross(x, y, dim=-1)
    det = (x_cross_y * z).sum(dim=-1)
    z_mult = torch.where(det < 0, -1.0, 1.0).to(x.device).unsqueeze(-1)
    
    # Final right-handed R
    R = torch.stack([x, y, z * z_mult], dim=-1)
    
    return center, size, R


def symmetry_aware_orient_loss(R_pred: torch.Tensor, R_gt: torch.Tensor):
    """
    Computes minimum rotation distance between R_pred and symmetry-equivalent R_gt.
    Metric: 1 - 1/3 * Tr(R_pred^T @ (R_gt @ S))
    Range : [0.0 (perfect) to 0.66 (90° rotation from any symmetry)]
    """
    B, M = R_pred.shape[:2]
    device = R_pred.device
    syms = _BOX_SYMMETRIES.to(device)
    
    # R_gt @ S: (B, M, 4, 3, 3)
    R_gt_sym = torch.matmul(R_gt.unsqueeze(2), syms)
    
    # 2. Compute trace for each symmetry: (B, M, 4)
    # Tr(R_p^T @ R_gt_sym) = sum(R_p * R_gt_sym)
    R_p_exp = R_pred.unsqueeze(2).expand(-1, -1, 4, -1, -1)
    traces = (R_p_exp * R_gt_sym).sum(dim=(-1, -2))
    
    # 3. Distance = (3 - trace) / 3   (Range: 0 to 2) 
    # With 180-deg symmetry, max distance should be ~0.66 (90 degrees away)
    dist_per_sym = (3.0 - traces) / 3.0
    
    # 4. Return minimum distance across symmetries
    return dist_per_sym.min(dim=-1)[0]


def hybrid_3d_loss(pred_c: torch.Tensor, pred_s: torch.Tensor, pred_R: torch.Tensor, pred_conf: torch.Tensor,
                   true_corners: torch.Tensor, valid_mask: torch.Tensor):
    """
    Direct Supervision Loss. 
    Accepts decoupled components (Center, Size, Rotation) directly.
    """
    # 1. Confidence Loss
    conf_loss = F.binary_cross_entropy_with_logits(pred_conf, valid_mask.float())

    # 2. Geometry (Ground Truth)
    # MUST reconstruct from GT corners as they are our primary label source
    gt_c, gt_s, gt_R = _corners_to_obb(true_corners)

    # 3. Component Errors
    # Center Loss
    ctr_err = (pred_c - gt_c).abs().mean(dim=-1)
    center_loss = ctr_err[valid_mask].mean() if valid_mask.any() else ctr_err.mean()

    # Size Loss
    sz_err = (pred_s - gt_s).abs().mean(dim=-1)
    size_loss = sz_err[valid_mask].mean() if valid_mask.any() else sz_err.mean()

    # Orientation Loss (Symmetry-Aware)
    orient_err = symmetry_aware_orient_loss(pred_R, gt_R)
    orient_loss = orient_err[valid_mask].mean() if valid_mask.any() else orient_err.mean()

    total = (Config.CENTER_WEIGHT * center_loss +
             Config.SIZE_WEIGHT * size_loss +
             Config.ORIENT_WEIGHT * orient_loss +
             Config.CONF_WEIGHT * conf_loss)

    return total, center_loss, size_loss, orient_loss, conf_loss
