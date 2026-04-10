import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.models as models
from config import Config

class BBox3DModel(nn.Module):
    def __init__(self):
        super(BBox3DModel, self).__init__()

        # --- Point cloud encoder (shared MLP + max-pool) ---
        self.pc_enc = nn.Sequential(
            nn.Linear(3, Config.POINT_HIDDEN_DIMS[0]), nn.BatchNorm1d(Config.POINT_HIDDEN_DIMS[0]), nn.ReLU(), nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(Config.POINT_HIDDEN_DIMS[0], Config.POINT_HIDDEN_DIMS[1]), nn.BatchNorm1d(Config.POINT_HIDDEN_DIMS[1]), nn.ReLU(), nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(Config.POINT_HIDDEN_DIMS[1], Config.POINT_HIDDEN_DIMS[2]), nn.BatchNorm1d(Config.POINT_HIDDEN_DIMS[2]), nn.ReLU()
        )
        self.pc_post = nn.Sequential(
            nn.Linear(Config.POINT_HIDDEN_DIMS[2], Config.POINT_POST_DIM), nn.ReLU(), nn.Dropout(Config.DROPOUT_RATE)
        )

        # --- Image backbone: ResNet18 up to layer4, NO global average pool ---
        # Output shape for 224×224 input: (B, 512, 7, 7)
        backbone = models.resnet18(weights='DEFAULT')
        self.img_backbone = nn.Sequential(*list(backbone.children())[:-2])

        if Config.FREEZE_BACKBONE:
            for param in self.img_backbone.parameters():
                param.requires_grad = False

        # --- Per-object decoder: (B, MAX_OBJECTS, FUSED_DIM) → (B, MAX_OBJECTS, 8, 3) ---
        self.decoder = nn.Sequential(
            nn.Linear(Config.FUSED_DIM, Config.DECODER_HIDDEN_DIM), nn.ReLU(), nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(Config.DECODER_HIDDEN_DIM, 8 * 3)
        )

    def forward(self, pc, mask, rgb):
        B, N, _ = pc.shape

        # 1. Global point cloud feature: (B, POINT_POST_DIM)
        pc_flat = pc.view(-1, 3)
        pc_feat = self.pc_enc(pc_flat).view(B, N, -1).max(dim=1)[0]
        pc_feat = self.pc_post(pc_feat)                                        # (B, 512)

        # 2. Dense image feature map — keep spatial dims for per-object pooling
        rgb_norm = TF.normalize(rgb.float() / 255.0,
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        feat_map = self.img_backbone(rgb_norm)                                  # (B, 512, 7, 7)
        _, C, H_, W_ = feat_map.shape

        # 3. Masked average pooling: one image feature vector per object slot.
        #    Resize the instance masks to the feature-map resolution, then pool.
        #    Uses efficient batched matmul instead of large intermediate broadcast.
        mask_small = F.interpolate(mask.float(), size=(H_, W_),
                                   mode='bilinear', align_corners=False)        # (B, MAX_OBJECTS, H', W')
        mask_flat  = mask_small.view(B, Config.MAX_OBJECTS, H_ * W_)           # (B, MAX_OBJECTS, H'W')
        feat_flat  = feat_map.view(B, C, H_ * W_).permute(0, 2, 1)            # (B, H'W', C)
        mask_sum   = mask_flat.sum(dim=2, keepdim=True).clamp(min=1e-6)        # (B, MAX_OBJECTS, 1)
        img_feat_obj = torch.bmm(mask_flat, feat_flat) / mask_sum              # (B, MAX_OBJECTS, C)

        # 4. Fuse: broadcast global PC feature across all object slots
        pc_feat_exp = pc_feat.unsqueeze(1).expand(-1, Config.MAX_OBJECTS, -1)  # (B, MAX_OBJECTS, 512)
        fused = torch.cat([pc_feat_exp, img_feat_obj], dim=-1)                 # (B, MAX_OBJECTS, FUSED_DIM)

        # 5. Per-object decoding
        return self.decoder(fused).view(B, Config.MAX_OBJECTS, 8, 3)
