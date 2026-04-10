"""
metrics.py — Pure component-based OBB loss.
"""

import torch
import torch.nn.functional as F
from config import Config


def _corners_to_obb(corners: torch.Tensor):
    """Extract (center, size, R) from 8 corner coordinates."""
    center = corners.mean(dim=-2)
    e1 = corners[..., 1, :] - corners[..., 0, :]
    e2 = corners[..., 3, :] - corners[..., 0, :]
    e3 = corners[..., 4, :] - corners[..., 0, :]
    
    size = torch.stack(
        [e1.norm(dim=-1), e2.norm(dim=-1), e3.norm(dim=-1)], dim=-1)
    R = torch.stack(
        [F.normalize(e1, dim=-1),
         F.normalize(e2, dim=-1),
         F.normalize(e3, dim=-1)], dim=-1)
    return center, size, R


def obb_3d_iou(pred_corners: torch.Tensor, true_corners: torch.Tensor,
               valid_mask: torch.Tensor = None,
               n_samples: int = Config.OBB_IOU_SAMPLES) -> float:
    """Monte Carlo 3D OBB IoU evaluation."""
    if valid_mask is None:
        valid_mask = true_corners.abs().sum(dim=(2, 3)) > 1e-6
    if not valid_mask.any():
        return 0.0

    with torch.no_grad():
        pred_c, pred_s, pred_R = _corners_to_obb(pred_corners)
        gt_c,   gt_s,   gt_R   = _corners_to_obb(true_corners)

        B, M = pred_corners.shape[:2]
        dev  = pred_corners.device

        pts_l = (torch.rand(B, M, n_samples, 3, device=dev) - 0.5) * pred_s.unsqueeze(-2)
        pts_w = pts_l @ pred_R.transpose(-1, -2) + pred_c.unsqueeze(-2)
        pts_g = (pts_w - gt_c.unsqueeze(-2)) @ gt_R

        half_gt = (gt_s / 2).unsqueeze(-2)
        inside  = (pts_g.abs() <= half_gt).all(dim=-1)

        vol_pred  = pred_s.prod(dim=-1)
        vol_gt    = gt_s.prod(dim=-1)
        vol_inter = inside.float().mean(dim=-1) * vol_pred
        vol_union = vol_pred + vol_gt - vol_inter

        iou = vol_inter / (vol_union + 1e-8)
        return iou[valid_mask].mean().item()


def hybrid_3d_loss(pred_corners: torch.Tensor, conf_logits: torch.Tensor,
                   true_corners: torch.Tensor, valid_mask: torch.Tensor):
    """
    Pure component loss: Center + Size + Orientation + Confidence.
    Directly optimizes all parameters from epoch 0.
    
    Returns: (total, center_loss, size_loss, orient_loss, conf_loss)
    """
    # ── 1. Objectness Loss (BCE) ──
    # Always active, penalizes ghost boxes in all slots.
    conf_loss = F.binary_cross_entropy_with_logits(conf_logits, valid_mask.float())

    # ── 2. Geometric Components (Valid objects only) ──
    pred_c, pred_s, pred_R = _corners_to_obb(pred_corners)
    gt_c,   gt_s,   gt_R   = _corners_to_obb(true_corners)

    # Translation
    ctr_err = (pred_c - gt_c).abs().mean(dim=-1)
    center_loss = ctr_err[valid_mask].mean() if valid_mask.any() else ctr_err.mean()

    # Scale
    sz_err = (pred_s - gt_s).abs().mean(dim=-1)
    size_loss = sz_err[valid_mask].mean() if valid_mask.any() else sz_err.mean()

    # Orientation
    rot_err = (pred_R - gt_R).abs().mean(dim=(-1, -2))
    orient_loss = rot_err[valid_mask].mean() if valid_mask.any() else rot_err.mean()

    total = (Config.CENTER_WEIGHT * center_loss +
             Config.SIZE_WEIGHT * size_loss +
             Config.ORIENT_WEIGHT * orient_loss +
             Config.CONF_WEIGHT * conf_loss)

    return total, center_loss, size_loss, orient_loss, conf_loss
