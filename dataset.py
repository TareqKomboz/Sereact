import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import torchvision.io as io
import torchvision.transforms.functional as TF
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

    def _augment(self, pc, mask, bbox, rgb, obj_pc, valid_slots, obj_indices):
        """
        Synchronized Multi-Modal Orthogonal Augmentation.
        Includes 90-deg Z-orientations and X/Y mirroring (flips).
        """
        # img_w, img_h for index flipping
        _, img_h, img_w = rgb.shape

        # 1. Discrete Z-axis Orientations (90, 180, 270 degrees) + "Shimmy" Jitter
        k = np.random.randint(0, 4)
        jitter = np.random.uniform(-Config.ORIENTATION_JITTER, Config.ORIENTATION_JITTER)
        angle = k * 90 + jitter # degrees CCW
        
        if abs(angle) > 1e-3:
            # Orientation Matrices for points
            theta = np.deg2rad(angle)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            orient_mtx = torch.tensor([[cos_t, -sin_t, 0],
                                        [sin_t,  cos_t, 0],
                                        [0, 0, 1]], dtype=torch.float32)
            pc     = pc @ orient_mtx.T
            bbox   = bbox @ orient_mtx.T
            obj_pc = obj_pc @ orient_mtx.T

            # Synchronized Image/Mask rotation
            rgb  = TF.rotate(rgb, angle)
            mask = TF.rotate(mask, angle)
            
            # Rotate obj_indices (x, y) around center (img_w/2, img_h/2)
            # x' = (x-cx)cos - (y-cy)sin + cx
            # y' = (x-cx)sin + (y-cy)cos + cy
            cx, cy = img_w / 2, img_h / 2
            ox, oy = obj_indices[..., 0] - cx, obj_indices[..., 1] - cy
            nx = ox * cos_t - oy * sin_t + cx
            ny = ox * sin_t + oy * cos_t + cy
            obj_indices[..., 0], obj_indices[..., 1] = nx, ny

        # 2. Horizontal Flip (X-Mirroring)
        if Config.AUGMENT_HFLIP and np.random.random() > 0.5:
            pc[:, 0] *= -1
            bbox[:, :, 0] *= -1
            obj_pc[:, :, 0] *= -1
            rgb, mask = torch.flip(rgb, [2]), torch.flip(mask, [2])
            obj_indices[..., 0] = (img_w - 1) - obj_indices[..., 0]

        # 3. Vertical Flip (Y-Mirroring)
        if Config.AUGMENT_VFLIP and np.random.random() > 0.5:
            pc[:, 1] *= -1
            bbox[:, :, 1] *= -1
            obj_pc[:, :, 1] *= -1
            rgb, mask = torch.flip(rgb, [1]), torch.flip(mask, [1])
            obj_indices[..., 1] = (img_h - 1) - obj_indices[..., 1]

        # 3. Local Geometry Scaling
        if np.random.random() > 0.5:
            scale = np.random.uniform(0.9, 1.1)
            for k in range(bbox.shape[0]):
                if not valid_slots[k]: continue
                center = bbox[k].mean(dim=0, keepdim=True)
                bbox[k] = (bbox[k] - center) * scale + center
                obj_pc[k] = (obj_pc[k] - center) * scale + center
                
        # 4. Synchronized Point Cloud Jitter
        pc += torch.randn_like(pc) * 0.005
        obj_pc += torch.randn_like(obj_pc) * 0.005

        # 5. Colour inversion
        if np.random.random() > 0.95:
            rgb = 255 - rgb

        # Keep projected indices valid for downstream feature sampling.
        obj_indices[..., 0] = obj_indices[..., 0].clamp(0, img_w - 1)
        obj_indices[..., 1] = obj_indices[..., 1].clamp(0, img_h - 1)
        
        return pc, mask, bbox, rgb, obj_pc, obj_indices

    def __getitem__(self, idx):
        path = self.samples[idx]

        # ── 1. Load raw structured point cloud (3, H, W) ──────────────────────
        pc_raw = np.load(os.path.join(path, "pc.npy"))                         # (3, H, W)

        # ── 2. Load instance masks at original resolution ───────────────────────
        mask_raw = np.load(os.path.join(path, "mask.npy"))                     # (M, H, W) bool

        # ── 3. Extract per-object point clouds via structured depth + 2-D masks ─
        #       pc_raw[:, mask_raw[k]] selects the 3-D points inside object k's
        #       segmentation mask directly — no projection needed.
        obj_pc_list, obj_indices_list, has_points = [], [], []
        H, W = pc_raw.shape[1], pc_raw.shape[2]
        for k in range(mask_raw.shape[0]):
            mask_k = mask_raw[k]
            y_coords, x_coords = np.where(mask_k)
            pts = pc_raw[:, mask_k].T.astype(np.float32)                      # (N_k, 3)
            indices = np.stack([x_coords, y_coords], axis=1).astype(np.float32) # (N_k, 2)
            n_k = pts.shape[0]
            has_points.append(n_k > 0)
            
            if n_k == 0:
                pts_fixed = np.zeros((Config.N_OBJ_POINTS, 3), dtype=np.float32)
                idx_fixed = np.zeros((Config.N_OBJ_POINTS, 2), dtype=np.float32)
            else:
                if n_k >= Config.N_OBJ_POINTS:
                    sel = np.random.choice(n_k, Config.N_OBJ_POINTS, replace=False)
                    pts_fixed = pts[sel]
                    idx_fixed = indices[sel]
                else:
                    rep = np.random.choice(n_k, Config.N_OBJ_POINTS - n_k, replace=True)
                    pts_fixed = np.concatenate([pts, pts[rep]], axis=0)
                    idx_fixed = np.concatenate([indices, indices[rep]], axis=0)
            obj_pc_list.append(pts_fixed)
            obj_indices_list.append(idx_fixed)

        obj_pc = torch.from_numpy(np.stack(obj_pc_list)).float()               # (M, N_OBJ_POINTS, 3)
        obj_indices = torch.from_numpy(np.stack(obj_indices_list)).float()     # (M, N_OBJ_POINTS, 2)
        has_points = torch.tensor(has_points, dtype=torch.bool)

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
        # Use 'nearest' to preserve binary labels (0/1) across resolutions.
        mask = F.interpolate(torch.from_numpy(mask_raw).unsqueeze(0).float(),
                             size=Config.IMG_SIZE, mode='nearest').squeeze(0)   # (M, 224, 224)

        # Scale obj_indices from raw res to Config.IMG_SIZE (224x224)
        # This allows direct sampling from the ResNet feature map after interpolation
        obj_indices[:, :, 0] *= (Config.IMG_SIZE[1] / W) # Scale X
        obj_indices[:, :, 1] *= (Config.IMG_SIZE[0] / H) # Scale Y

        # ── 6. Ground-truth bounding boxes ─────────────────────────────────────
        bbox = torch.from_numpy(np.load(os.path.join(path, "bbox3d.npy")))    # (M, 8, 3)

        # Bug-C fix: capture valid slots BEFORE augmentation.
        # An object is ONLY valid if it has both a non-empty BBox AND at least one Point.
        valid_slots = (bbox.abs().sum(dim=(1, 2)) > 1e-6) & has_points         # (M,) bool

        # ── 7. RGB image ────────────────────────────────────────────────────────
        rgb = F.interpolate(
            io.read_image(os.path.join(path, "rgb.jpg")).unsqueeze(0).float(),
            size=Config.IMG_SIZE).squeeze(0)                                    # (3, 224, 224)

        # ── 8. Augmentation (train split only, when enabled) ────────────────────
        if self.split == "train" and Config.AUGMENT:
            pc, mask, bbox, rgb, obj_pc, obj_indices = self._augment(pc, mask, bbox, rgb, obj_pc, valid_slots, obj_indices)

        # ── 9. Pad all per-instance tensors to MAX_OBJECTS ──────────────────────
        M   = mask.shape[0]
        pad = max(0, Config.MAX_OBJECTS - M)
        mask        = F.pad(mask,       (0, 0, 0, 0, 0, pad))[:Config.MAX_OBJECTS]
        bbox        = F.pad(bbox,       (0, 0, 0, 0, 0, pad))[:Config.MAX_OBJECTS]
        valid_slots = F.pad(valid_slots.float(), (0, pad))[:Config.MAX_OBJECTS].bool()
        obj_pc      = F.pad(obj_pc,     (0, 0, 0, 0, 0, pad))[:Config.MAX_OBJECTS]
        obj_indices = F.pad(obj_indices, (0, 0, 0, 0, 0, pad))[:Config.MAX_OBJECTS]

        return {"pc": pc, "obj_pc": obj_pc, "obj_indices": obj_indices, "mask": mask,
                "bbox": bbox, "rgb": rgb.byte(), "valid": valid_slots}
