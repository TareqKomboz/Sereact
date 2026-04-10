"""
model.py — BBox3DModel with three complementary feature streams:

  1. Global PC feature     : PointNet over all scene points  → scene-level context
  2. Per-object PC feature : PointNet over each object's own masked 3D points → local geometry
  3. Per-object image feat : ResNet18 feature map pooled within each instance mask → appearance

Streams 1+2 come from the structured (organized) depth point cloud; stream 3 from RGB.
All three feed a shared per-object MLP decoder that outputs (MAX_OBJECTS, 8, 3) corners.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.models as models
from config import Config


def _pointnet_encoder(hidden_dims, use_bn=True):
    """
    Shared-MLP per-point encoder block (no global pooling — caller does max-pool).
    use_bn=False for the per-object encoder to avoid zero-padded-slot stat contamination.
    Returns (nn.Sequential, output_channels).
    """
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

        # ── Stream 1: Global scene point cloud (with BN — all points are real) ──
        self.pc_enc, _out = _pointnet_encoder(Config.POINT_HIDDEN_DIMS, use_bn=True)
        self.pc_post = nn.Sequential(
            nn.Linear(_out, Config.POINT_POST_DIM), nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE)
        )

        # ── Stream 2: Per-object point cloud (NO BN — zero-padded slots skew statistics) ──
        self.obj_pc_enc, _obj_out = _pointnet_encoder(Config.OBJ_PC_HIDDEN_DIMS, use_bn=False)
        self.obj_pc_post = nn.Linear(_obj_out, Config.OBJ_PC_OUTPUT_DIM)

        # ── Stream 3: Image — ResNet18 up to layer4, spatial map preserved ──
        #    Output: (B, 512, 7, 7) for 224×224 input
        backbone = models.resnet18(weights='DEFAULT')
        self.img_backbone = nn.Sequential(*list(backbone.children())[:-2])
        if Config.FREEZE_BACKBONE:
            for param in self.img_backbone.parameters():
                param.requires_grad = False

        # ── Per-object decoder (deeper 3-layer MLP) ──
        self.decoder = _mlp(Config.FUSED_DIM, Config.DECODER_HIDDEN_DIMS,
                            out_dim=8 * 3, dropout=Config.DROPOUT_RATE)

    def forward(self, pc, obj_pc, mask, rgb):
        B, N, _ = pc.shape
        M = Config.MAX_OBJECTS

        # ── 1. Global scene PC feature → (B, POINT_POST_DIM) ──
        pc_feat = self.pc_enc(pc.view(-1, 3)).view(B, N, -1).max(dim=1)[0]
        pc_feat = self.pc_post(pc_feat)                                         # (B, 512)

        # ── 2. Per-object PC feature → (B, MAX_OBJECTS, OBJ_PC_OUTPUT_DIM) ──
        # obj_pc: (B, MAX_OBJECTS, N_OBJ_POINTS, 3)
        N_p = obj_pc.shape[2]
        obj_enc = self.obj_pc_enc(obj_pc.view(-1, 3))                          # (B*M*N_p, F)
        obj_enc = obj_enc.view(B, M, N_p, -1).max(dim=2)[0]                   # (B, M, F)
        obj_pc_feat = self.obj_pc_post(obj_enc)                                 # (B, M, 256)

        # ── 3. Per-object image feature via masked avg-pool → (B, MAX_OBJECTS, C) ──
        rgb_norm = TF.normalize(rgb.float() / 255.0,
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        feat_map = self.img_backbone(rgb_norm)                                  # (B, C, H', W')
        _, C, H_, W_ = feat_map.shape
        mask_s   = F.interpolate(mask.float(), size=(H_, W_),
                                 mode='bilinear', align_corners=False)          # (B, M, H', W')
        mask_f   = mask_s.view(B, M, H_ * W_)                                 # (B, M, H'W')
        feat_f   = feat_map.view(B, C, H_ * W_).permute(0, 2, 1)             # (B, H'W', C)
        mask_sum = mask_f.sum(dim=2, keepdim=True).clamp(min=1e-6)            # (B, M, 1)
        img_feat = torch.bmm(mask_f, feat_f) / mask_sum                        # (B, M, 512)

        # ── 4. Fuse all three streams and decode per object ──
        pc_exp = pc_feat.unsqueeze(1).expand(-1, M, -1)                        # (B, M, 512)
        fused  = torch.cat([pc_exp, obj_pc_feat, img_feat], dim=-1)            # (B, M, FUSED_DIM)
        return self.decoder(fused).view(B, M, 8, 3)
