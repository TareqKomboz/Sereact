"""
metrics.py — OBB-aware loss functions and evaluation metrics.

Training loss  : hybrid_3d_loss()
  Corner L1 (primary) + Center L1 (auxiliary after warmup).
  No axis-alignment assumption — works for arbitrarily oriented boxes.

Eval metric    : obb_3d_iou()
  Monte Carlo 3D OBB IoU via uniform sampling inside the predicted box.
  Correct for arbitrary orientations; replaces the old AABB approximation
  which over-estimated volume by up to 41 % for 45°-rotated boxes.
"""

import torch
import torch.nn.functional as F
from config import Config


# ── Shared helper: extract OBB params from 8 corners ─────────────────────────

def _corners_to_obb(corners: torch.Tensor):
    """
    Extract (center, size, R) from 8 corner coordinates.
    Corner ordering must match model._decode_obb(), verified on GT data (err < 1e-6).

      corners : (..., 8, 3)
    Returns:
      center  : (..., 3)
      size    : (..., 3)   — positive edge lengths (sx, sy, sz)
      R       : (..., 3, 3) — rotation matrix, columns = local axis directions
    """
    center = corners.mean(dim=-2)                                               # (..., 3)
    e1 = corners[..., 1, :] - corners[..., 0, :]                               # local X * sx
    e2 = corners[..., 3, :] - corners[..., 0, :]                               # local Y * sy
    e3 = corners[..., 4, :] - corners[..., 0, :]                               # local Z * sz
    size = torch.stack(
        [e1.norm(dim=-1), e2.norm(dim=-1), e3.norm(dim=-1)], dim=-1)           # (..., 3)
    R = torch.stack(
        [F.normalize(e1, dim=-1),
         F.normalize(e2, dim=-1),
         F.normalize(e3, dim=-1)], dim=-1)                                      # (..., 3, 3)
    return center, size, R


# ── Evaluation metric: Monte Carlo 3D OBB IoU ────────────────────────────────

def obb_3d_iou(pred_corners: torch.Tensor, true_corners: torch.Tensor,
               valid_mask: torch.Tensor = None,
               n_samples: int = Config.OBB_IOU_SAMPLES) -> float:
    """
    Monte Carlo 3D OBB IoU, averaged over valid slots.

    Algorithm (no axis-alignment assumed):
      1. Extract OBB params (center, size, R) from both corner sets.
      2. Sample n_samples points uniformly inside each predicted box.
      3. Transform samples into the GT box's local frame.
      4. Count the fraction that fall within the GT box's half-extents.
      5. IoU = intersection_vol / union_vol.

    pred_corners, true_corners : (B, MAX_OBJ, 8, 3)
    valid_mask                 : (B, MAX_OBJ) bool — None = derive from GT sum
    Returns                    : scalar mean IoU over valid object slots
    """
    if valid_mask is None:
        valid_mask = true_corners.abs().sum(dim=(2, 3)) > 1e-6
    if not valid_mask.any():
        return 0.0

    with torch.no_grad():
        pred_c, pred_s, pred_R = _corners_to_obb(pred_corners)                 # (B, M, *)
        gt_c,   gt_s,   gt_R   = _corners_to_obb(true_corners)

        B, M = pred_corners.shape[:2]
        dev  = pred_corners.device

        # 1. Sample uniformly in predicted box local frame, scaled by size
        pts_l = (torch.rand(B, M, n_samples, 3, device=dev) - 0.5) \
                * pred_s.unsqueeze(-2)                                          # (B, M, N, 3)

        # 2. Rotate + translate to world frame: pts_w = pts_l @ pred_R^T + pred_c
        pts_w = pts_l @ pred_R.transpose(-1, -2) \
                + pred_c.unsqueeze(-2)                                          # (B, M, N, 3)

        # 3. Transform to GT box local frame: pts_g = (pts_w - gt_c) @ gt_R
        #    (gt_R cols are world-frame directions → gt_R^T maps world→local,
        #     but row-vectors use @ gt_R, which equals the same thing)
        pts_g = (pts_w - gt_c.unsqueeze(-2)) @ gt_R                            # (B, M, N, 3)

        # 4. Point is inside GT box iff |coord_i| ≤ half-size_i for all i
        half_gt = (gt_s / 2).unsqueeze(-2)                                     # (B, M, 1, 3)
        inside  = (pts_g.abs() <= half_gt).all(dim=-1)                         # (B, M, N) bool

        # 5. Volume estimates
        vol_pred  = pred_s.prod(dim=-1)                                         # (B, M)
        vol_gt    = gt_s.prod(dim=-1)
        vol_inter = inside.float().mean(dim=-1) * vol_pred                     # (B, M)
        vol_union = vol_pred + vol_gt - vol_inter

        iou = vol_inter / (vol_union + 1e-8)                                    # (B, M)
        return iou[valid_mask].mean().item()


# ── Training loss: corner L1 + center L1 (OBB-correct) ───────────────────────

def hybrid_3d_loss(pred_corners: torch.Tensor, true_corners: torch.Tensor,
                   valid_mask: torch.Tensor = None,
                   l1_weight: float = Config.L1_WEIGHT,
                   center_weight: float = Config.CENTER_WEIGHT,
                   l1_only: bool = False):
    """
    OBB-correct training loss: Corner L1 (primary) + Center L1 (auxiliary).

    Why not AABB DIoU?
      The old aabb_3d_iou took min/max over corners to build an axis-aligned
      enclosure.  For a 45° rotated OBB, this enclosure is sqrt(2)× larger than
      the true box, making the IoU signal systematically wrong.

    This loss avoids any axis-alignment assumption:
      • Warmup  : pure corner L1 — direct supervision on all 8 corner positions.
      • Full    : corner L1 + center L1 — the center term gives an explicit,
                  strong gradient signal for box position that is independent of
                  orientation/size errors.

    Args:
      pred_corners, true_corners : (B, MAX_OBJ, 8, 3)
      valid_mask   : (B, MAX_OBJ) bool  — pre-augmentation slot validity mask.
                     Pass the dataset's 'valid' field to avoid mis-classifying
                     augmentation-shifted real objects as padding.
      l1_only      : if True, skip the center term (used during warmup epochs).

    Returns: (total_loss, center_loss, corner_l1_loss)
      The second return value is center_loss (replaces old DIoU loss).
      All training logs / early stopping use total_loss (first value).
    """
    if valid_mask is None:
        valid_mask = true_corners.abs().sum(dim=(2, 3)) > 1e-6

    # ── 1. Corner L1 — mean absolute error across all 8 corners and 3 dims ──
    l1_per_slot = (pred_corners - true_corners).abs().mean(dim=(2, 3))         # (B, M)
    l1_loss = (l1_per_slot[valid_mask].mean()
               if valid_mask.any() else l1_per_slot.mean())

    if l1_only:
        return l1_loss, torch.zeros_like(l1_loss), l1_loss

    # ── 2. Center L1 — explicit centroid supervision ─────────────────────────
    pred_center = pred_corners.mean(dim=-2)                                     # (B, M, 3)
    gt_center   = true_corners.mean(dim=-2)
    ctr_per_slot = (pred_center - gt_center).abs().mean(dim=-1)               # (B, M)
    center_loss = (ctr_per_slot[valid_mask].mean()
                   if valid_mask.any() else ctr_per_slot.mean())

    total = l1_weight * l1_loss + center_weight * center_loss
    return total, center_loss, l1_loss
