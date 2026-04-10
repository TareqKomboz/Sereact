import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import BBox3DDataset
from model import BBox3DModel
from metrics import hybrid_3d_loss
from config import Config

# Setup
device = torch.device("cpu")
ds = BBox3DDataset(Config.DATA_ROOT, split="train")
loader = DataLoader(ds, batch_size=2, shuffle=True)
batch = next(iter(loader))

print("="*60)
print("DATALOADER TEST")
print("="*60)
print(f"Batch pc shape:   {batch['pc'].shape}")
print(f"Batch bbox shape: {batch['bbox'].shape}")
print(f"Valid mask sum:   {batch['valid'].sum().item()} / {batch['valid'].numel()}")

# Model test
model = BBox3DModel().to(device)
model.eval()

with torch.no_grad():
    pred, conf = model(batch['pc'], batch['obj_pc'], batch['mask'], batch['rgb'])

print("\n"+"="*60)
print("MODEL OUTPUT TEST")
print("="*60)
print(f"Corners shape: {pred.shape}")
print(f"Conf shape:    {conf.shape}")
print(f"Conf range:    [{conf.min():.3f}, {conf.max():.3f}]")

# Check losses
total, ctr, sz, ori, cnf = hybrid_3d_loss(pred, conf, batch['bbox'], batch['valid'])
print("\n"+"="*60)
print("LOSS COMPONENTS")
print("="*60)
print(f"Total:      {total:.4f}")
print(f"Center L1:  {ctr:.4f}")
print(f"Size L1:    {sz:.4f}")
print(f"Orient L1:  {ori:.4f}")
print(f"Conf BCE:   {cnf:.4f}")

probs = torch.sigmoid(conf)
print(f"\nMean Objectness Prob: {probs.mean():.4f}")
