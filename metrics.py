import torch
from config import Config

def aabb_3d_iou(pred_bbox, true_bbox, differentiable=False):
    """
    Primary Interpretive Metric: Mean 3D IoU (Axis-Aligned).
    Returns a scalar [0, 1] representing box overlap.
    """
    p_min, p_max = pred_bbox.min(dim=2)[0], pred_bbox.max(dim=2)[0]
    t_min, t_max = true_bbox.min(dim=2)[0], true_bbox.max(dim=2)[0]
    inter_min, inter_max = torch.max(p_min, t_min), torch.min(p_max, t_max)
    inter_vol = torch.clamp(inter_max - inter_min, min=0).prod(dim=-1)
    union_vol = (p_max - p_min).prod(dim=-1) + (t_max - t_min).prod(dim=-1) - inter_vol
    valid_mask = true_bbox.abs().sum(dim=(2, 3)) > 1e-6
    iou = inter_vol / torch.clamp(union_vol, min=1e-6)
    if not differentiable:
        # Bug B fix: guard against empty valid_mask (all-padding batch) to avoid empty-tensor mean crash
        return iou[valid_mask].mean().item() if valid_mask.any() else 0.0
    return iou, valid_mask


def diou_3d_loss(pred_bbox, true_bbox):
    """
    Primary Differentiable Loss: 3D Distance-IoU.
    Optimizes for both volume overlap and center-point distance.
    """
    iou, mask = aabb_3d_iou(pred_bbox, true_bbox, differentiable=True)
    p_c, t_c = pred_bbox.mean(dim=2), true_bbox.mean(dim=2)
    dist_sq = torch.sum((p_c - t_c)**2, dim=-1)
    p_min, p_max = pred_bbox.min(dim=2)[0], pred_bbox.max(dim=2)[0]
    t_min, t_max = true_bbox.min(dim=2)[0], true_bbox.max(dim=2)[0]
    c_min, c_max = torch.min(p_min, t_min), torch.max(p_max, t_max)
    c_diag_sq = torch.sum((c_max - c_min)**2, dim=-1)
    diou = iou - (dist_sq / torch.clamp(c_diag_sq, min=1e-6))
    return (1.0 - diou)[mask].mean() if mask.any() else torch.zeros(1, device=pred_bbox.device).squeeze()


def hybrid_3d_loss(pred_bbox, true_bbox, valid_mask=None, l1_weight=Config.L1_WEIGHT, l1_only=False):
    """
    Hybrid Loss: DIoU + L1 Coordinate Loss.
    - l1_only=True:  pure L1 warmup — stable gradients at initialization.
    - l1_only=False: full hybrid loss with DIoU geometric alignment.
    - valid_mask: optional pre-computed (B, MAX_OBJECTS) bool tensor.
      Provide this from the dataset's pre-augmentation valid_slots so that
      objects whose corners were shifted near zero by translation/scale
      augmentation are never silently excluded from the loss (Bug C fix).
      Falls back to deriving from true_bbox when not provided (e.g. eval).
    """
    # Bug C fix: prefer externally supplied mask; derive from coords only as fallback
    if valid_mask is None:
        valid_mask = true_bbox.abs().sum(dim=(2, 3)) > 1e-6

    # 1. L1 component (on all 8 corners × 3 dims, valid slots only)
    l1_per_slot = torch.abs(pred_bbox - true_bbox).mean(dim=(2, 3))
    l1_loss = l1_per_slot[valid_mask].mean() if valid_mask.any() else l1_per_slot.mean()

    if l1_only:
        return l1_loss, torch.zeros_like(l1_loss), l1_loss

    # 2. DIoU component (valid slots only)
    iou, _ = aabb_3d_iou(pred_bbox, true_bbox, differentiable=True)
    p_c, t_c = pred_bbox.mean(dim=2), true_bbox.mean(dim=2)
    dist_sq = torch.sum((p_c - t_c)**2, dim=-1)
    p_min, p_max = pred_bbox.min(dim=2)[0], pred_bbox.max(dim=2)[0]
    t_min, t_max = true_bbox.min(dim=2)[0], true_bbox.max(dim=2)[0]
    c_min, c_max = torch.min(p_min, t_min), torch.max(p_max, t_max)
    c_diag_sq = torch.sum((c_max - c_min)**2, dim=-1)
    diou = iou - (dist_sq / torch.clamp(c_diag_sq, min=1e-6))
    diou_loss_per = (1.0 - diou)
    diou_loss = diou_loss_per[valid_mask].mean() if valid_mask.any() else diou_loss_per.mean()

    total_loss = diou_loss + l1_weight * l1_loss
    return total_loss, diou_loss, l1_loss
