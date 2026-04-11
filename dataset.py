import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import torchvision.io as io
from config import Config


class BBox3DDataset(Dataset):
    def __init__(self, root_dir, num_points=Config.NUM_POINTS, split="train"):
        self.root_dir  = root_dir
        self.num_points = num_points
        self.split     = split
        self.samples   = []
        for folder in sorted(os.listdir(root_dir)):
            sample_path = os.path.join(root_dir, folder)
            if os.path.isdir(sample_path):
                self.samples.append(sample_path)
        idx1 = int(len(self.samples) * Config.TRAIN_RATIO)
        idx2 = int(len(self.samples) * Config.VAL_RATIO)
        if   split == "train": self.samples = self.samples[:idx1]
        elif split == "val":   self.samples = self.samples[idx1:idx2]
        else:                  self.samples = self.samples[idx2:]

    def __len__(self):
        return len(self.samples)

    def _augment(self, pc, mask, bbox, rgb, obj_pc):
        # 1. Z-axis rotation
        if np.random.random() > 0.5:
            theta = np.random.uniform(-np.pi / 4, np.pi / 4)
            R = torch.tensor([[np.cos(theta), -np.sin(theta), 0],
                               [np.sin(theta),  np.cos(theta), 0],
                               [0, 0, 1]], dtype=torch.float32)
            pc   = pc   @ R.T
            bbox = bbox @ R.T
            obj_pc = obj_pc @ R.T
        
        # 2. Random 3-D translation
        offset = torch.randn(3) * 0.1
        pc   += offset
        bbox += offset
        obj_pc += offset
        
        # 3. Horizontal flip (synchronized across all modalities)
        if np.random.random() > 0.5:
            pc[:, 0] *= -1
            bbox[:, :, 0] *= -1
            obj_pc[:, :, 0] *= -1
            # Swap corner indices in bbox to match the coordinate flip
            swap = [1, 0, 3, 2, 5, 4, 7, 6]
            bbox = bbox[:, swap]
            rgb, mask = torch.flip(rgb, [2]), torch.flip(mask, [2])
            
        # 4. Point cloud jitter
        pc += torch.randn_like(pc) * 0.005
        obj_pc += torch.randn_like(obj_pc) * 0.002 # Less jitter for local
        
        # 5. Colour inversion (10 % chance)
        if np.random.random() > 0.9:
            rgb = 255 - rgb
            
        # 6. Uniform scaling
        scale = np.random.uniform(0.9, 1.1)
        pc *= scale; bbox *= scale; obj_pc *= scale
        
        return pc, mask, bbox, rgb, obj_pc

    def __getitem__(self, idx):
        path = self.samples[idx]

        # ── 1. Load raw structured point cloud (3, H, W) ──────────────────────
        pc_raw = np.load(os.path.join(path, "pc.npy"))                         # (3, H, W)

        # ── 2. Load instance masks at original resolution ───────────────────────
        mask_raw = np.load(os.path.join(path, "mask.npy"))                     # (M, H, W) bool

        # ── 3. Extract per-object point clouds via structured depth + 2-D masks ─
        #       pc_raw[:, mask_raw[k]] selects the 3-D points inside object k's
        #       segmentation mask directly — no projection needed.
        obj_pc_list = []
        for k in range(mask_raw.shape[0]):
            pts = pc_raw[:, mask_raw[k]].T.astype(np.float32)                 # (N_k, 3)
            n_k = pts.shape[0]
            if n_k == 0:
                pts_fixed = np.zeros((Config.N_OBJ_POINTS, 3), dtype=np.float32)
            elif n_k >= Config.N_OBJ_POINTS:
                sel = np.random.choice(n_k, Config.N_OBJ_POINTS, replace=False)
                pts_fixed = pts[sel]
            else:
                rep = np.random.choice(n_k, Config.N_OBJ_POINTS - n_k, replace=True)
                pts_fixed = np.concatenate([pts, pts[rep]], axis=0)
            obj_pc_list.append(pts_fixed)
        obj_pc = torch.from_numpy(np.stack(obj_pc_list)).float()               # (M, N_OBJ_POINTS, 3)

        # ── 4. Global point cloud — subsample / pad to NUM_POINTS ───────────────
        pc_np = pc_raw.reshape(3, -1).T                                        # (H*W, 3)
        if pc_np.shape[0] > self.num_points:
            sel = np.random.choice(pc_np.shape[0], self.num_points, replace=False)
            pc_np = pc_np[sel]
        elif pc_np.shape[0] < self.num_points:
            rep = np.random.choice(pc_np.shape[0], self.num_points - pc_np.shape[0], replace=True)
            pc_np = np.concatenate([pc_np, pc_np[rep]], axis=0)
        pc = torch.from_numpy(pc_np).float()

        # ── 5. Resize instance masks to IMG_SIZE for image-branch pooling ───────
        mask = F.interpolate(torch.from_numpy(mask_raw).unsqueeze(0).float(),
                             size=Config.IMG_SIZE).squeeze(0)                   # (M, 224, 224)

        # ── 6. Ground-truth bounding boxes ─────────────────────────────────────
        bbox = torch.from_numpy(np.load(os.path.join(path, "bbox3d.npy")))    # (M, 8, 3)

        # Bug-C fix: capture valid slots BEFORE augmentation.
        # Translation / scale augmentation can push corners near zero,
        # silently mis-classifying real objects as padding in the loss.
        valid_slots = bbox.abs().sum(dim=(1, 2)) > 1e-6                        # (M,) bool

        # ── 7. RGB image ────────────────────────────────────────────────────────
        rgb = F.interpolate(
            io.read_image(os.path.join(path, "rgb.jpg")).unsqueeze(0).float(),
            size=Config.IMG_SIZE).squeeze(0)                                    # (3, 224, 224)

        # ── 8. Augmentation (train split only, when enabled) ────────────────────
        if self.split == "train" and Config.AUGMENT:
            pc, mask, bbox, rgb, obj_pc = self._augment(pc, mask, bbox, rgb, obj_pc)

        # ── 9. Pad all per-instance tensors to MAX_OBJECTS ──────────────────────
        M   = mask.shape[0]
        pad = max(0, Config.MAX_OBJECTS - M)
        mask        = F.pad(mask,       (0, 0, 0, 0, 0, pad))[:Config.MAX_OBJECTS]
        bbox        = F.pad(bbox,       (0, 0, 0, 0, 0, pad))[:Config.MAX_OBJECTS]
        valid_slots = F.pad(valid_slots.float(), (0, pad))[:Config.MAX_OBJECTS].bool()
        obj_pc      = F.pad(obj_pc,     (0, 0, 0, 0, 0, pad))[:Config.MAX_OBJECTS]

        return {"pc": pc, "obj_pc": obj_pc, "mask": mask,
                "bbox": bbox, "rgb": rgb.byte(), "valid": valid_slots}
