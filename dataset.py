import os
import torch
from torch.utils.data import Dataset
import numpy as np
import torchvision.io as io
from config import Config

class BBox3DDataset(Dataset):
    def __init__(self, root_dir, num_points=2048, split="train"):
        self.root_dir = root_dir
        self.num_points = num_points
        self.split = split
        self.samples = []
        for folder in sorted(os.listdir(root_dir)):  # Bug #2: sorted for deterministic splits
            sample_path = os.path.join(root_dir, folder)
            if os.path.isdir(sample_path):
                self.samples.append(sample_path)
        idx1 = int(len(self.samples) * Config.TRAIN_RATIO)
        idx2 = int(len(self.samples) * Config.VAL_RATIO)
        if split == "train": self.samples = self.samples[:idx1]
        elif split == "val": self.samples = self.samples[idx1:idx2]
        else: self.samples = self.samples[idx2:]

    def __len__(self):
        return len(self.samples)

    def _augment(self, pc, mask, bbox, rgb):
        # 1. 3D Rotation (Z-axis) - Essential for orientation invariance
        if np.random.random() > 0.5:
            theta = np.random.uniform(-np.pi/4, np.pi/4) # +/- 45 degrees
            rot_mat = torch.tensor([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta),  np.cos(theta), 0],
                                    [0, 0, 1]], dtype=torch.float32)
            pc = pc @ rot_mat.T
            bbox = bbox @ rot_mat.T # Rotate all 8 corners per box
            
        # 2. 3D Translation
        # Shift the whole scene to teach the model objects can be anywhere.
        offset = torch.randn(3) * 0.1 # Small random shift
        pc += offset
        bbox += offset
        
        # 3. Horizontal Flip (Existing, synchronized)
        if np.random.random() > 0.5:
            pc[:,0] *= -1
            bbox[:,:,0] *= -1
            idx_map = [1,0,3,2,5,4,7,6]; bbox = bbox[:, idx_map]
            rgb, mask = torch.flip(rgb, [2]), torch.flip(mask, [2])
        
        # 4. Point Cloud Noise
        # Jitter: Simulate sensor precision errors.
        pc += torch.randn_like(pc) * 0.005
        
        # 5. Color Augmentation (Photometric)
        # Invert colors: 10% chance
        if np.random.random() > 0.9: rgb = 255 - rgb
        
        # 6. Scaling (Existing)
        scale = np.random.uniform(0.9, 1.1)
        pc *= scale; bbox *= scale
        
        return pc, mask, bbox, rgb

    def __getitem__(self, idx):
        # 1. Locate sample directory
        path = self.samples[idx]
        
        # 2. Load and Reshape Point Cloud
        # The raw point cloud is (3, N), we transpose to (N, 3) for standard point processing.
        pc_np = np.load(os.path.join(path, "pc.npy")).reshape(3, -1).T
        
        # 3. Fixed-size Point Resampling
        # Neural networks require fixed-size inputs for batching. We downsample to exactly num_points.
        if pc_np.shape[0] > self.num_points:
            idx_list = np.random.choice(pc_np.shape[0], self.num_points, replace=False)
            pc_np = pc_np[idx_list]
        elif pc_np.shape[0] < self.num_points:  # Bug #3: pad by repeating points
            repeat_idx = np.random.choice(pc_np.shape[0], self.num_points - pc_np.shape[0], replace=True)
            pc_np = np.concatenate([pc_np, pc_np[repeat_idx]], axis=0)
        pc = torch.from_numpy(pc_np).float()
        
        import torch.nn.functional as F
        
        # 4. Load & Resize Instance Masks
        # resnet18 expects 224x224 input. We interpolate the mask to match this resolution.
        mask_np = np.load(os.path.join(path, "mask.npy"))
        mask = F.interpolate(torch.from_numpy(mask_np).unsqueeze(0).float(), size=Config.IMG_SIZE).squeeze(0)
        
        # 5. Load Ground Truth 3D Bounding Boxes
        bbox = torch.from_numpy(np.load(os.path.join(path, "bbox3d.npy")))

        # Bug C: record valid object slots BEFORE augmentation.
        # Translation/scale augmentation can shift small corner coordinates below the 1e-6
        # threshold, causing real objects to be silently excluded from the loss.
        valid_slots = bbox.abs().sum(dim=(1, 2)) > 1e-6  # (N_obj,) bool

        # 6. Load & Resize RGB Image
        # Images are interpolated to 224x224 so they can be processed by our pretrained ResNet backbone.
        rgb = F.interpolate(io.read_image(os.path.join(path, "rgb.jpg")).unsqueeze(0).float(), size=Config.IMG_SIZE).squeeze(0)
        
        # 7. Apply Data Augmentation (Train only)
        # We only augment training data to help the model generalize to new viewpoints.
        if hasattr(self, 'split') and self.split == "train":
           pc, mask, bbox, rgb = self._augment(pc, mask, bbox, rgb)
        
        # 8. Fixed-length Padding for Batches
        # Different scenes have different numbers of objects (e.g., 5 vs 15).
        # We pad everything to 60 instances so the batch tensor has a uniform shape (B, 60, 8, 3).
        pad_m = max(0, Config.MAX_OBJECTS - mask.shape[0])
        mask = F.pad(mask, (0, 0, 0, 0, 0, pad_m))[:Config.MAX_OBJECTS]
        bbox = F.pad(bbox, (0, 0, 0, 0, 0, max(0, Config.MAX_OBJECTS - bbox.shape[0])))[:Config.MAX_OBJECTS]
        # Pad valid_slots to MAX_OBJECTS (new False entries mark padding slots)
        valid_slots = F.pad(valid_slots.float(),
                            (0, max(0, Config.MAX_OBJECTS - valid_slots.shape[0])))[:Config.MAX_OBJECTS].bool()

        return {"pc": pc, "mask": mask, "bbox": bbox, "rgb": rgb.byte(), "valid": valid_slots}
