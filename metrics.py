"""
metrics.py — Pure component-based OBB loss with Direct Supervision & Sync Augmentation.
"""

import torch
import torch.nn.functional as F
from config import Config

# --- Allowed orientation symmetries ---
# Keep only yaw-180 symmetry for upright objects.
# Allowing X/Y flips makes orientation under-constrained and can collapse heading learning.
_ORIENTATION_SYMMETRIES = torch.tensor([
    [[1.,  0.,  0.], [ 0.,  1.,  0.], [ 0.,  0.,  1.]],  # Identity
    [[-1., 0.,  0.], [ 0., -1.,  0.], [ 0.,  0.,  1.]],  # 180° around Z
], dtype=torch.float32)


def _corners_to_obb(corners: torch.Tensor):
    """
    OBB extraction using ordered corners:
      0(-x,-y,-z), 1(+x,-y,-z), 3(-x,+y,-z), 4(-x,-y,+z)
    Returns orientation in the same format as model output (SO(3), columns [x,y,z]).
    """
    center = corners.mean(dim=-2)                                               # (..., 3)

    e1 = corners[..., 1, :] - corners[..., 0, :]                                # +x edge
    e2 = corners[..., 3, :] - corners[..., 0, :]                                # +y edge
    e3 = corners[..., 4, :] - corners[..., 0, :]                                # +z edge

    x = F.normalize(e1, dim=-1, eps=1e-8)
    y_seed = F.normalize(e2, dim=-1, eps=1e-8)

    z = torch.cross(x, y_seed, dim=-1)
    z_norm = z.norm(dim=-1, keepdim=True)
    z_fallback = F.normalize(e3, dim=-1, eps=1e-8)
    z = torch.where(z_norm > 1e-6, z / (z_norm + 1e-8), z_fallback)

    # Keep a right-handed basis and stable sign with respect to corner 4.
    y = F.normalize(torch.cross(z, x, dim=-1), dim=-1, eps=1e-8)
    align = (z * z_fallback).sum(dim=-1, keepdim=True)
    flip = torch.where(align < 0, -1.0, 1.0).to(corners.dtype)
    y = y * flip
    z = z * flip

    orient = torch.stack([x, y, z], dim=-1)                                     # (..., 3, 3)

    centered = corners - center.unsqueeze(-2)
    local = centered @ orient
    size = (local.max(dim=-2).values - local.min(dim=-2).values).clamp(min=1e-8)
    return center, size, orient


def symmetry_aware_orient_loss(pred_orient: torch.Tensor, gt_orient: torch.Tensor):
    """
    Geodesic SO(3) distance with yaw 180° symmetry.
    Returns per-box angle in radians.
    """
    device = pred_orient.device
    syms = _ORIENTATION_SYMMETRIES.to(device=device, dtype=pred_orient.dtype)

    # (B, M, S, 3, 3), S = number of allowed symmetries
    gt_orient_sym = torch.matmul(gt_orient.unsqueeze(2), syms)
    rel = torch.matmul(pred_orient.transpose(-1, -2).unsqueeze(2), gt_orient_sym)
    traces = rel.diagonal(offset=0, dim1=-2, dim2=-1).sum(dim=-1)               # (B, M, 4)
    cos_theta = ((traces - 1.0) / 2.0).clamp(min=-1.0 + 1e-6, max=1.0)
    angle = torch.acos(cos_theta)                                                # (B, M, 4)
    return angle.min(dim=-1)[0]                                                  # (B, M)


def _orientation_reliability_from_size(size: torch.Tensor):
    """
    Reliability weight in [0.1, 1.0] based on anisotropy.
    Near-cubic boxes have weakly-defined orientation and get down-weighted.
    """
    s_sorted = torch.sort(size, dim=-1, descending=True).values
    gap1 = (s_sorted[..., 0] - s_sorted[..., 1]).abs()
    gap2 = (s_sorted[..., 1] - s_sorted[..., 2]).abs()
    rel = ((gap1 + gap2) / (s_sorted[..., 0] + 1e-6)).clamp(min=0.1, max=1.0)
    return rel


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


def hybrid_3d_loss(pred_c: torch.Tensor, pred_s_log: torch.Tensor, pred_orient: torch.Tensor,
                   true_corners: torch.Tensor, valid_mask: torch.Tensor, model=None):
    """
    Direct supervision loss for 3D bounding box regression.
    """
    # 1. Geometry (Ground Truth)
    gt_c, gt_s, gt_orient = _corners_to_obb(true_corners)

    # 2. Component Errors
    # Center Loss
    ctr_err = (pred_c - gt_c).abs().mean(dim=-1)
    center_loss = ctr_err[valid_mask].mean() if valid_mask.any() else ctr_err.mean()

    # Size Loss (Log-Space L1)
    gt_s_log = torch.log(gt_s + 1e-8)
    sz_err = (pred_s_log - gt_s_log).abs().mean(dim=-1)
    size_loss = sz_err[valid_mask].mean() if valid_mask.any() else sz_err.mean()

    # Orientation Loss (Symmetry-Aware)
    orient_err = symmetry_aware_orient_loss(pred_orient, gt_orient)
    orient_w = _orientation_reliability_from_size(gt_s)
    orient_weighted = orient_err * orient_w
    if valid_mask.any():
        orient_loss = orient_weighted[valid_mask].sum() / (orient_w[valid_mask].sum() + 1e-8)
    else:
        orient_loss = orient_weighted.mean()

    # 3. Evaluation Metrics (IoU & RMSE)
    iou_3d, rmse = torch.tensor(0.0, device=pred_c.device), torch.tensor(0.0, device=pred_c.device)
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
             Config.ORIENT_WEIGHT * orient_loss)

    return total, center_loss, size_loss, orient_loss, iou_3d, rmse
