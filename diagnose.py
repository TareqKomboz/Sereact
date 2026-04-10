"""
Diagnostic script to inspect the raw data and identify what the model is actually seeing.
Run with: python3 diagnose.py
"""
import numpy as np
import os
import torch
from dataset import BBox3DDataset
from torch.utils.data import DataLoader
from model import BBox3DModel
from config import Config

root = Config.DATA_ROOT
folders = sorted([f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))])
sample_path = os.path.join(root, folders[0])

print("=" * 60)
print("RAW DATA INSPECTION (first sample)")
print("=" * 60)
pc = np.load(os.path.join(sample_path, "pc.npy"))
bbox = np.load(os.path.join(sample_path, "bbox3d.npy"))
mask = np.load(os.path.join(sample_path, "mask.npy"))

print(f"pc.npy shape:      {pc.shape}  | dtype: {pc.dtype}")
print(f"bbox3d.npy shape:  {bbox.shape} | dtype: {bbox.dtype}")
print(f"mask.npy shape:    {mask.shape} | dtype: {mask.dtype}")
print()

pc_t = pc.reshape(3, -1).T
print(f"Point cloud (N,3) shape: {pc_t.shape}")
print(f"  X range: [{pc_t[:,0].min():.3f}, {pc_t[:,0].max():.3f}]")
print(f"  Y range: [{pc_t[:,1].min():.3f}, {pc_t[:,1].max():.3f}]")
print(f"  Z range: [{pc_t[:,2].min():.3f}, {pc_t[:,2].max():.3f}]")
print()

print(f"BBox raw shape: {bbox.shape}")
print(f"Number of objects (bbox.shape[0]): {bbox.shape[0]}")
print(f"BBox coordinate ranges:")
print(f"  X range: [{bbox[...,0].min():.3f}, {bbox[...,0].max():.3f}]")
print(f"  Y range: [{bbox[...,1].min():.3f}, {bbox[...,1].max():.3f}]")
print(f"  Z range: [{bbox[...,2].min():.3f}, {bbox[...,2].max():.3f}]")
print(f"First box corners:\n{bbox[0]}")
print()

print("=" * 60)
print("DATASET STATS (all samples)")
print("=" * 60)
num_objects_per_scene = []
for f in folders:
    b = np.load(os.path.join(root, f, "bbox3d.npy"))
    num_objects_per_scene.append(b.shape[0])
print(f"Total scenes: {len(folders)}")
print(f"Objects per scene - min: {min(num_objects_per_scene)}, max: {max(num_objects_per_scene)}, mean: {np.mean(num_objects_per_scene):.1f}")
print()

print("=" * 60)
print("DATALOADER TEST (one batch)")
print("=" * 60)
ds = BBox3DDataset(root, split="train")
loader = DataLoader(ds, batch_size=2)
batch = next(iter(loader))
print(f"Batch pc shape:   {batch['pc'].shape}")
print(f"Batch obj_pc shape: {batch['obj_pc'].shape}")
print(f"Batch mask shape: {batch['mask'].shape}")
print(f"Batch bbox shape: {batch['bbox'].shape}")
print(f"Batch rgb shape:  {batch['rgb'].shape}")
vb = batch['bbox'][0]
valid = batch['valid'][0]  # Pre-augmentation valid mask from dataset
print(f"Valid bbox slots in first sample: {valid.sum().item()} / {vb.shape[0]}")
print()

print("=" * 60)
print("MODEL OUTPUT TEST (untrained)")
print("=" * 60)
device = torch.device("cpu")
model = BBox3DModel().to(device)
with torch.no_grad():
    pred = model(batch['pc'], batch['obj_pc'], batch['mask'], batch['rgb'])
print(f"Model output shape: {pred.shape}")
print(f"Model output range: [{pred.min():.3f}, {pred.max():.3f}]")
gt_center = batch['bbox'][0][valid].mean(dim=(0,1))
pred_center = pred[0][valid].mean(dim=(0,1))
print(f"GT center (first sample, valid slots):   {gt_center.numpy()}")
print(f"Pred center (first sample, valid slots): {pred_center.numpy()}")

# Check L1 loss
from metrics import hybrid_3d_loss
loss, center, l1 = hybrid_3d_loss(pred, batch['bbox'], valid_mask=batch['valid'])
print(f"\nLoss check - total: {loss.item():.4f}, Center: {center.item():.4f}, L1: {l1.item():.4f}")
