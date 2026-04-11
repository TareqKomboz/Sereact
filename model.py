"""
model.py — BBox3DModel with OBB-constrained output head.

Three complementary feature streams feed a shared per-object decoder that
predicts 12 raw OBB parameters per object slot.  These are decoded into 8
geometrically valid corner coordinates before being returned.

Output head (12 params → 8 corners):
  center   [0:3]  — box centroid in world coordinates
  log_size [3:6]  — log of box dimensions; exp() gives strictly positive sizes
  rot6d    [6:12] — two unconstrained 3D vectors; converted to SO(3) via
                    Gram-Schmidt (Zhou et al., 2019, "On the Continuity of
                    Rotation Representations in Neural Networks").

Corner ordering matches the GT data (verified: max error 2.6e-7):
  bottom face (z=-): 0(-x,-y), 1(+x,-y), 2(+x,+y), 3(-x,+y)
  top    face (z=+): 4(-x,-y), 5(+x,-y), 6(+x,+y), 7(-x,+y)
  pillars: 0-4, 1-5, 2-6, 3-7

Forward returns: (B, MAX_OBJECTS, 8, 3) — same interface as before,
  but now every predicted box is guaranteed to be a valid OBB.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.models as models
from config import Config

# ── Fixed canonical corner offsets (unit OBB, ±1 in local frame) ─────────────
# Row k = sign vector for corner k.  Multiply by half-sizes, rotate, translate.
_CORNER_OFFSETS = torch.tensor(
    [[-1., -1., -1.], [+1., -1., -1.], [+1., +1., -1.], [-1., +1., -1.],
     [-1., -1., +1.], [+1., -1., +1.], [+1., +1., +1.], [-1., +1., +1.]],
    dtype=torch.float32)                                                         # (8, 3)


def rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    """
    Continuous 6D representation → valid SO(3) rotation matrix via Gram-Schmidt.
    Input : (..., 6)    — two arbitrary (and independent) 3-D vectors
    Output: (..., 3, 3) — columns form a right-handed orthonormal frame
    """
    a1, a2 = rot6d[..., :3], rot6d[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (a2 * b1).sum(dim=-1, keepdim=True) * b1, dim=-1)
    # Cross product along last dim (compatible with all PyTorch versions)
    b3 = torch.stack([b1[..., 1]*b2[..., 2] - b1[..., 2]*b2[..., 1],
                      b1[..., 2]*b2[..., 0] - b1[..., 0]*b2[..., 2],
                      b1[..., 0]*b2[..., 1] - b1[..., 1]*b2[..., 0]], dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)                                    # (..., 3, 3)


def _pointnet_encoder(hidden_dims, use_bn=True):
    """Shared-MLP per-point encoder.  Returns (nn.Sequential, output_channels)."""
    layers, in_d = [], 3
    for d in hidden_dims:
        layers.append(nn.Linear(in_d, d))
        if use_bn:
            layers.append(nn.BatchNorm1d(d))
        layers.append(nn.ReLU())
        in_d = d
    return nn.Sequential(*layers), in_d


def _mlp(in_dim, hidden_dims, out_dim, dropout=0.0):
    """Dense MLP with optional Dropout after each hidden layer."""
    layers, cur = [], in_dim
    for h in hidden_dims:
        layers += [nn.Linear(cur, h), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        cur = h
    layers.append(nn.Linear(cur, out_dim))
    return nn.Sequential(*layers)


class BBox3DModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Register corner offsets as a buffer (auto-moves with .to(device))
        self.register_buffer('corner_offsets', _CORNER_OFFSETS)                 # (8, 3)

        # ── Stream 1: Global scene PC (with BN — all points are real geometry) ──
        self.pc_enc, _out = _pointnet_encoder(Config.POINT_HIDDEN_DIMS, use_bn=True)
        self.pc_post = nn.Sequential(
            nn.Linear(_out, Config.POINT_POST_DIM), nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE)
        )

        # ── Stream 2: Per-object PC (NO BN — ~85% of slots are zero-padded) ──
        self.obj_pc_enc, _obj_out = _pointnet_encoder(Config.OBJ_PC_HIDDEN_DIMS, use_bn=False)
        self.obj_pc_post = nn.Linear(_obj_out, Config.OBJ_PC_OUTPUT_DIM)

        # ── Stream 3: Image — ResNet18 up to layer4, spatial map preserved ──
        backbone = models.resnet18(weights='DEFAULT')
        self.img_backbone = nn.Sequential(*list(backbone.children())[:-2])
        if Config.FREEZE_BACKBONE:
            for param in self.img_backbone.parameters():
                param.requires_grad = False

        # ── OBB output head ──────────────────────────────────────────────────
        # 12 raw params per slot: center(3) + log_size(3) + rot6d(6)
        # _decode_obb() converts these to 8 geometrically valid corners.
        self.decoder = _mlp(Config.FUSED_DIM, Config.DECODER_HIDDEN_DIMS,
                            out_dim=12, dropout=Config.DROPOUT_RATE)

        # ── Confidence output head ───────────────────────────────────────────
        # Predicts if a slot contains an object (conf > 0.5) or is background padding.
        self.conf_head = nn.Linear(Config.FUSED_DIM, 1)

    # ── OBB decoding ─────────────────────────────────────────────────────────
    def reconstruct_corners(self, center: torch.Tensor, size: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct 8 corner coordinates from OBB components.
        center: (B, M, 3), size: (B, M, 3), R: (B, M, 3, 3)
        """
        # local corners (B, M, 8, 3): offset signs × half-sizes
        half  = size / 2                                                        # (B, M, 3)
        local = self.corner_offsets * half.unsqueeze(-2)                        # (B, M, 8, 3)

        # World coordinates: local @ R^T + center
        world = local @ R.transpose(-1, -2) + center.unsqueeze(-2)             # (B, M, 8, 3)
        return world

    def _decode_obb(self, raw: torch.Tensor) -> torch.Tensor:
        """
        Convert raw decoder output to 8 geometrically valid OBB corners.
        Input : (B, MAX_OBJ, 12)
        Output: (B, MAX_OBJ, 8, 3)
        """
        center = raw[..., 0:3]                                                  # (B, M, 3)
        size   = torch.exp(raw[..., 3:6]).clamp(min=1e-3)                      # (B, M, 3) > 0
        R      = rot6d_to_matrix(raw[..., 6:12])                               # (B, M, 3, 3)
        
        return self.reconstruct_corners(center, size, R)

    def forward(self, pc, obj_pc, mask, rgb):
        B, N, _ = pc.shape
        M = Config.MAX_OBJECTS

        # ── Stream 1: global scene feature → (B, 512) ───────────────────────
        pc_feat = self.pc_enc(pc.view(-1, 3)).view(B, N, -1).max(dim=1)[0]
        pc_feat = self.pc_post(pc_feat)

        # ── Stream 2: per-object PC feature → (B, M, 256) ──────────────────
        N_p = obj_pc.shape[2]
        obj_enc = self.obj_pc_enc(obj_pc.view(-1, 3))
        obj_enc = obj_enc.view(B, M, N_p, -1).max(dim=2)[0]
        obj_pc_feat = self.obj_pc_post(obj_enc)

        # ── Stream 3: per-object image feature → (B, M, 512) ───────────────
        rgb_norm = TF.normalize(rgb.float() / 255.0,
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        feat_map = self.img_backbone(rgb_norm)                                  # (B, C, H', W')
        _, C, H_, W_ = feat_map.shape
        mask_s   = F.interpolate(mask.float(), size=(H_, W_),
                                 mode='bilinear', align_corners=False)
        mask_f   = mask_s.view(B, M, H_ * W_)
        feat_f   = feat_map.view(B, C, H_ * W_).permute(0, 2, 1)
        mask_sum = mask_f.sum(dim=2, keepdim=True).clamp(min=1e-6)
        img_feat = torch.bmm(mask_f, feat_f) / mask_sum                        # (B, M, 512)

        # ── Fuse → predict OBB params + confidence ───────────────────────────
        pc_exp = pc_feat.unsqueeze(1).expand(-1, M, -1)
        fused  = torch.cat([pc_exp, obj_pc_feat, img_feat], dim=-1)            # (B, M, 1280)
        fused  = F.dropout(fused, p=Config.DROPOUT_RATE, training=self.training)
        
        raw    = self.decoder(fused)                                            # (B, M, 12)
        conf   = self.conf_head(fused).squeeze(-1)                             # (B, M)
        
        # Extract components directly from raw output
        center   = raw[..., 0:3]
        log_size = raw[..., 3:6]
        size     = torch.exp(log_size).clamp(min=1e-3)
        R        = rot6d_to_matrix(raw[..., 6:12])
        
        return center, size, R, conf, log_size
