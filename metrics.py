"""
metrics.py — Pure component-based OBB loss with Direct Supervision & Sync Augmentation.
"""

import torch
import torch.nn.functional as F
from config import Config

# --- 180-degree orientations around X, Y, Z axes (Klein Four-Group) ---
_ORIENTATION_SYMMETRIES = torch.tensor([
    [[1.,  0.,  0.], [ 0.,  1.,  0.], [ 0.,  0.,  1.]],  # Identity
    [[1.,  0.,  0.], [ 0., -1.,  0.], [ 0.,  0., -1.]],  # 180° around X
    [[-1., 0.,  0.], [ 0.,  1.,  0.], [ 0.,  0., -1.]],  # 180° around Y
    [[-1., 0.,  0.], [ 0., -1.,  0.], [ 0.,  0.,  1.]],  # 180° around Z
], dtype=torch.float32)


def _corners_to_obb(corners: torch.Tensor):
    """
    Extract (center, size, orient) from 8 corner coordinates with numerical safety.
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
    
    # Final right-handed orientation matrix
    orient = torch.stack([x, y, z * z_mult], dim=-1)
    
    return center, size, orient


def symmetry_aware_orient_loss(pred_orient: torch.Tensor, gt_orient: torch.Tensor):
    """
    Computes minimum orientation distance between pred_orient and symmetry-equivalent gt_orient.
    Metric: 1 - 1/3 * Tr(pred_orient^T @ (gt_orient @ S))
    Range : [0.0 (perfect) to 0.66 (90° rotation from any symmetry)]
    """
    B, M = pred_orient.shape[:2]
    device = pred_orient.device
    syms = _ORIENTATION_SYMMETRIES.to(device)
    
    # gt_orient @ S: (B, M, 4, 3, 3)
    gt_orient_sym = torch.matmul(gt_orient.unsqueeze(2), syms)
    
    # 2. Compute trace for each symmetry: (B, M, 4)
    # Tr(pred_orient^T @ gt_orient_sym) = sum(pred_orient * gt_orient_sym)
    pred_orient_exp = pred_orient.unsqueeze(2).expand(-1, -1, 4, -1, -1)
    traces = (pred_orient_exp * gt_orient_sym).sum(dim=(-1, -2))
    
    # 3. Distance = (3 - trace) / 3   (Range: 0 to 2) 
    # With 180-deg symmetry, max distance should be ~0.66 (90 degrees away)
    dist_per_sym = (3.0 - traces) / 3.0
    
    # 4. Return minimum distance across symmetries
    return dist_per_sym.min(dim=-1)[0]


def calculate_3d_iou(pred_corners: torch.Tensor, gt_corners: torch.Tensor):
    """
    3D IoU Approximation: BEV IoU * Height IoU.
    Input: (B, M, 8, 3)
    Returns: (B, M)
    """
    # 1. Height overlap (Z-axis)
    # Bottom: index 0, Top: index 4
    z_min_p, z_max_p = pred_corners[..., 0:4, 2].min(dim=-1)[0], pred_corners[..., 4:8, 2].max(dim=-1)[0]
    z_min_g, z_max_g = gt_corners[..., 0:4, 2].min(dim=-1)[0], gt_corners[..., 4:8, 2].max(dim=-1)[0]
    
    inter_z = (torch.min(z_max_p, z_max_g) - torch.max(z_min_p, z_min_g)).clamp(min=0)
    union_z = (z_max_p - z_min_p) + (z_max_g - z_min_g) - inter_z
    iou_z   = inter_z / (union_z + 1e-8)

    # 2. BEV overlap (X-Y plane)
    # Simple Axis-Aligned BEV approximation for the challenge
    x_min_p, x_max_p = pred_corners[..., :, 0].min(dim=-1)[0], pred_corners[..., :, 0].max(dim=-1)[0]
    x_min_g, x_max_g = gt_corners[..., :, 0].min(dim=-1)[0], gt_corners[..., :, 0].max(dim=-1)[0]
    y_min_p, y_max_p = pred_corners[..., :, 1].min(dim=-1)[0], pred_corners[..., :, 1].max(dim=-1)[0]
    y_min_g, y_max_g = gt_corners[..., :, 1].min(dim=-1)[0], gt_corners[..., :, 1].max(dim=-1)[0]

    i_x = (torch.min(x_max_p, x_max_g) - torch.max(x_min_p, x_min_g)).clamp(min=0)
    i_y = (torch.min(y_max_p, y_max_g) - torch.max(y_min_p, y_min_g)).clamp(min=0)
    inter_bev = i_x * i_y
    
    area_p = (x_max_p - x_min_p) * (y_max_p - y_min_p)
    area_g = (x_max_g - x_min_g) * (y_max_g - y_min_g)
    union_bev = area_p + area_g - inter_bev
    iou_bev = inter_bev / (union_bev + 1e-8)

    return iou_bev * iou_z


def hybrid_3d_loss(pred_c: torch.Tensor, pred_s_log: torch.Tensor, pred_orient: torch.Tensor, pred_conf: torch.Tensor,
                   true_corners: torch.Tensor, valid_mask: torch.Tensor, model=None):
    """
    Direct Supervision Loss. 
    Returns: (total_loss, center_loss, size_loss, orient_loss, conf_loss, iou_3d, rmse)
    """
    # 1. Confidence Loss
    conf_loss = F.binary_cross_entropy_with_logits(pred_conf, valid_mask.float())

    # 2. Geometry (Ground Truth)
    gt_c, gt_s, gt_orient = _corners_to_obb(true_corners)

    # 3. Component Errors
    # Center Loss
    ctr_err = (pred_c - gt_c).abs().mean(dim=-1)
    center_loss = ctr_err[valid_mask].mean() if valid_mask.any() else ctr_err.mean()

    # Size Loss (Log-Space L1)
    gt_s_log = torch.log(gt_s + 1e-8)
    sz_err = (pred_s_log - gt_s_log).abs().mean(dim=-1)
    size_loss = sz_err[valid_mask].mean() if valid_mask.any() else sz_err.mean()

    # Orientation Loss (Symmetry-Aware)
    orient_err = symmetry_aware_orient_loss(pred_orient, gt_orient)
    orient_loss = orient_err[valid_mask].mean() if valid_mask.any() else orient_err.mean()

    # 4. Evaluation Metrics (IoU & RMSE)
    # Only meaningful if model is provided to reconstruct corners
    iou_3d, rmse = torch.tensor(0.0), torch.tensor(0.0)
    if model is not None:
        with torch.no_grad():
            pred_corners = model.reconstruct_corners(pred_c, torch.exp(pred_s_log), pred_orient)
            iou_val = calculate_3d_iou(pred_corners, true_corners)
            iou_3d  = iou_val[valid_mask].mean() if valid_mask.any() else iou_val.mean()
            
            # Corner RMSE (meters)
            sq_err = (pred_corners - true_corners)**2
            rmse = torch.sqrt(sq_err[valid_mask].mean()) if valid_mask.any() else torch.sqrt(sq_err.mean())

    total = (Config.CENTER_WEIGHT * center_loss +
             Config.SIZE_WEIGHT * size_loss +
             Config.ORIENT_WEIGHT * orient_loss +
             Config.CONF_WEIGHT * conf_loss)

    return total, center_loss, size_loss, orient_loss, conf_loss, iou_3d, rmse
