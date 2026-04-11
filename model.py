"""
model.py — BBox3DModel with OBB-constrained output head.

Three complementary feature streams feed a shared per-object decoder that
predicts 12 raw OBB parameters per object slot.  These are decoded into 8
geometrically valid corner coordinates before being returned.

Output head (12 params → 8 corners):
  center   [0:3]  — box centroid in world coordinates
  log_size [3:6]  — log of box dimensions; exp() gives strictly positive sizes
  orient6d [6:12] — two unconstrained 3D vectors; converted to SO(3) via
                    Gram-Schmidt (Zhou et al., 2019, "On the Continuity of
                    Orientation Representations in Neural Networks").

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


def orient6d_to_matrix(orient6d: torch.Tensor) -> torch.Tensor:
    """
    Continuous 6D representation → valid SO(3) rotation matrix via Gram-Schmidt.
    Input : (..., 6)    — two arbitrary (and independent) 3-D vectors
    Output: (..., 3, 3) — columns form a right-handed orthonormal frame
    """
    a1, a2 = orient6d[..., :3], orient6d[..., 3:6]
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
        backbone = models.resnet50(weights='DEFAULT')
        self.img_backbone = nn.Sequential(*list(backbone.children())[:-2])
        if Config.FREEZE_BACKBONE:
            for param in self.img_backbone.parameters():
                param.requires_grad = False

        # ── Specialized OBB output heads ─────────────────────────────────────
        # Shared feature extractor for all OBB components
        self.shared_decoder = _mlp(Config.FUSED_DIM, Config.DECODER_HIDDEN_DIMS,
                                   out_dim=512, dropout=Config.DROPOUT_RATE)
        
        # Dedicated heads for different geometric properties
        self.center_head    = nn.Linear(512, 3)  # Predicts RESIDUAL offset from local centroid
        self.size_head      = nn.Linear(512, 3)  # Predicts log_size
        self.orient_head    = _mlp(512, [512, 256], out_dim=6) # Deeper branch for SO(3)
        
    # ── OBB decoding ─────────────────────────────────────────────────────────
    def reconstruct_corners(self, center: torch.Tensor, size: torch.Tensor, orient: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct 8 corner coordinates from OBB components.
        center: (B, M, 3), size: (B, M, 3), orient: (B, M, 3, 3)
        """
        # local corners (B, M, 8, 3): offset signs × half-sizes
        half  = size / 2                                                        # (B, M, 3)
        local = self.corner_offsets * half.unsqueeze(-2)                        # (B, M, 8, 3)

        # World coordinates: local @ orient^T + center
        world = local @ orient.transpose(-1, -2) + center.unsqueeze(-2)             # (B, M, 8, 3)
        return world

    def forward(self, pc, obj_pc, obj_indices, rgb, mask):
        """
        Multimodal Forward Pass.
        pc         : (B, N, 3)
        obj_pc     : (B, M, N_p, 3)
        obj_indices: (B, M, N_p, 2) -- (x, y) coordinates in IMG_SIZE space
        rgb        : (B, 3, H, W)
        mask       : (B, M, H, W) -- Per-object mask used as spatial prior for image pooling
        """
        B, N, _ = pc.shape
        M = Config.MAX_OBJECTS
        N_p = obj_pc.shape[2]

        # ── Pre-calculate Local Centroids for Residual Regression ───────────
        local_centroid = obj_pc.mean(dim=2)                                     # (B, M, 3)

        # ── Stream 1: global scene feature → (B, 512) ───────────────────────
        pc_feat = self.pc_enc(pc.view(-1, 3)).view(B, N, -1).max(dim=1)[0]
        pc_feat = self.pc_post(pc_feat)

        # ── Stream 2: per-object PC feature → (B, M, 256) ──────────────────
        obj_enc = self.obj_pc_enc(obj_pc.view(-1, 3))
        obj_enc = obj_enc.view(B, M, N_p, -1).max(dim=2)[0]
        obj_pc_feat = self.obj_pc_post(obj_enc)

        # ── Stream 3: per-object image feature → (B, M, C) ──────────────────
        rgb_norm = TF.normalize(rgb.float() / 255.0,
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        feat_map = self.img_backbone(rgb_norm)                                  # (B, C, H', W')
        _, C, H_, W_ = feat_map.shape

        # Use instance masks to pool per-object visual features directly.
        # This makes mask an actual forward input and aligns vision features to object extents.
        mask_ds = F.interpolate(
            mask.float().view(B * M, 1, mask.shape[-2], mask.shape[-1]),
            size=(H_, W_),
            mode='nearest'
        ).view(B, M, 1, H_, W_)
        masked_feat = feat_map.unsqueeze(1) * mask_ds                           # (B, M, C, H', W')
        mask_area = mask_ds.sum(dim=(-1, -2)).clamp(min=1.0)                    # (B, M, 1)
        img_feat = masked_feat.sum(dim=(-1, -2)) / mask_area                    # (B, M, C)

        # ── Fuse → predict OBB params + confidence ───────────────────────────
        pc_exp = pc_feat.unsqueeze(1).expand(-1, M, -1)
        
        # Stochastic Modality Masking (Regularization)
        if self.training:
            m_drop = 0.10
            if torch.rand(1) < m_drop: pc_exp = torch.zeros_like(pc_exp)
            if torch.rand(1) < m_drop: obj_pc_feat = torch.zeros_like(obj_pc_feat)
            if torch.rand(1) < m_drop: img_feat = torch.zeros_like(img_feat)
            
        fused  = torch.cat([pc_exp, obj_pc_feat, img_feat], dim=-1)            # (B, M, F)
        fused  = F.dropout(fused, p=Config.DROPOUT_RATE, training=self.training)
        
        # ── Specialized Prediction Heads ─────────────────────────────────────
        z = self.shared_decoder(fused)                                          # (B, M, 512)
        
        # 1. Residual Center
        center_offset = self.center_head(z)
        center        = local_centroid + center_offset
        
        # 2. Scale (Size)
        log_size = self.size_head(z)
        size     = torch.exp(log_size).clamp(min=1e-3)
        
        # 3. Orientation
        orient6d = self.orient_head(z)
        orient   = orient6d_to_matrix(orient6d)
        
        return center, size, orient, log_size
