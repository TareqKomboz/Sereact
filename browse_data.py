"""
inspect.py — Interactive training-data browser.

Usage
-----
# Show a specific sample by index (default: index 0)
python3 inspect.py --idx 5

# Show a random sample from the train / val / test split
python3 inspect.py --random
python3 inspect.py --random --split val
python3 inspect.py --random --split test

# Show N consecutive samples starting at idx
python3 inspect.py --idx 0 --count 3

# Skip the point cloud and show only the RGB + mask grid
python3 inspect.py --idx 5 --rgb-only

Keyboard shortcuts while a window is open
------------------------------------------
  q / Escape : close current window and move to next sample
  Mouse drag : rotate the 3D view
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Allow running from any directory
sys.path.insert(0, os.path.dirname(__file__))

from config import Config
from dataset import BBox3DDataset
import torchvision.io as io
import torch.nn.functional as F
import torch

BBOX_EDGES = [(0,1),(1,2),(2,3),(3,0),
              (4,5),(5,6),(6,7),(7,4),
              (0,4),(1,5),(2,6),(3,7)]


def _is_valid_box(b):
    return np.abs(b).sum() > 1e-6


def _draw_box(ax, b, color='lime', linewidth=2.0, alpha=1.0, label=None):
    for k, (i, j) in enumerate(BBOX_EDGES):
        lbl = label if (k == 0 and label) else None
        ax.plot([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]],
                color=color, linewidth=linewidth, alpha=alpha, label=lbl)


def _load_raw(sample_path):
    """Load raw files from a sample directory (no dataset preprocessing)."""
    pc_np  = np.load(os.path.join(sample_path, "pc.npy")).reshape(3, -1).T   # (N, 3)
    bbox   = np.load(os.path.join(sample_path, "bbox3d.npy"))                 # (M, 8, 3)
    mask   = np.load(os.path.join(sample_path, "mask.npy"))                   # (M, H, W) bool
    rgb_t  = io.read_image(os.path.join(sample_path, "rgb.jpg"))               # (3, H, W) uint8
    rgb_np = rgb_t.permute(1, 2, 0).numpy()                                   # (H, W, 3)
    return pc_np, bbox, mask, rgb_np


def inspect_sample(sample_path, sample_idx, split, rgb_only=False):
    """Open an interactive multi-panel figure for one training sample."""
    pc, bbox, mask, rgb = _load_raw(sample_path)

    n_obj = bbox.shape[0]
    valid  = np.abs(bbox).sum(axis=(1, 2)) > 1e-6
    n_valid = valid.sum()

    title = (f"Sample {sample_idx} | split={split} | "
             f"path={os.path.basename(sample_path)}\n"
             f"{n_valid}/{n_obj} valid objects  |  "
             f"PC: {pc.shape[0]:,} pts  |  "
             f"Image: {rgb.shape[1]}×{rgb.shape[0]}")

    print(f"\n{'='*60}")
    print(title.replace('\n', '  '))
    print(f"  PC X: [{pc[:,0].min():.3f}, {pc[:,0].max():.3f}]")
    print(f"  PC Y: [{pc[:,1].min():.3f}, {pc[:,1].max():.3f}]")
    print(f"  PC Z: [{pc[:,2].min():.3f}, {pc[:,2].max():.3f}]")
    for k, b in enumerate(bbox):
        flag = "✓" if valid[k] else "✗ (padding)"
        print(f"  Box {k:2d} {flag}  center=[{b.mean(0)[0]:.3f}, {b.mean(0)[1]:.3f}, {b.mean(0)[2]:.3f}]")

    # ------------------------------------------------------------------ #
    # Figure layout                                                        #
    # ------------------------------------------------------------------ #
    if rgb_only:
        _show_rgb_masks(rgb, mask, bbox, valid, title)
        return

    # Row 0: RGB image | Full 3D scene | Zoomed first object
    # Row 1: instance-mask grid (first min(n_obj, 10) objects)
    n_mask_cols = min(n_valid, 10)
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(title, fontsize=10, y=0.99)
    gs_top = gridspec.GridSpec(1, 3, figure=fig, top=0.88, bottom=0.38, wspace=0.35)
    gs_bot = gridspec.GridSpec(1, max(n_mask_cols, 1), figure=fig,
                               top=0.33, bottom=0.02, wspace=0.15)

    # --- RGB image ---
    ax_rgb = fig.add_subplot(gs_top[0, 0])
    ax_rgb.imshow(rgb); ax_rgb.set_title("RGB Image"); ax_rgb.axis('off')

    # --- Full 3D scene ---
    ax3d = fig.add_subplot(gs_top[0, 1], projection='3d')
    # Downsample for speed
    stride = max(1, pc.shape[0] // 8000)
    ax3d.scatter(pc[::stride,0], pc[::stride,1], pc[::stride,2],
                 s=1, c='deepskyblue', alpha=0.35, depthshade=True)
    legend_added = False
    for k, b in enumerate(bbox):
        if not valid[k]: continue
        lbl = 'GT box' if not legend_added else None
        _draw_box(ax3d, b, color='lime', linewidth=1.5, label=lbl)
        # Mark centroid
        c = b.mean(0)
        ax3d.text(c[0], c[1], c[2], str(k), fontsize=7, color='white',
                  ha='center', va='center',
                  bbox=dict(boxstyle='round,pad=0.1', facecolor='green', alpha=0.7))
        legend_added = True
    ax3d.set_xlabel('X', labelpad=2); ax3d.set_ylabel('Y', labelpad=2)
    ax3d.set_zlabel('Z', labelpad=2)
    ax3d.set_title(f"Full Scene — {n_valid} GT boxes")
    if legend_added: ax3d.legend(fontsize=8)

    # --- Zoomed first valid object ---
    ax_zoom = fig.add_subplot(gs_top[0, 2], projection='3d')
    first_b = next((b for b, v in zip(bbox, valid) if v), None)
    if first_b is not None:
        pad = 0.20
        lo = first_b.min(0) - pad; hi = first_b.max(0) + pad
        sel = ((pc[:,0] >= lo[0]) & (pc[:,0] <= hi[0]) &
               (pc[:,1] >= lo[1]) & (pc[:,1] <= hi[1]) &
               (pc[:,2] >= lo[2]) & (pc[:,2] <= hi[2]))
        pc_z = pc[sel] if sel.sum() > 10 else pc
        ax_zoom.scatter(pc_z[:,0], pc_z[:,1], pc_z[:,2],
                        s=3, c='deepskyblue', alpha=0.6, depthshade=True)
        _draw_box(ax_zoom, first_b, color='lime', linewidth=2.5, label='GT box 0')
        ax_zoom.legend(fontsize=8)
    ax_zoom.set_xlabel('X', labelpad=2); ax_zoom.set_ylabel('Y', labelpad=2)
    ax_zoom.set_zlabel('Z', labelpad=2); ax_zoom.set_title("Zoomed — Object 0")

    # --- Instance mask grid (bottom row) ---
    valid_indices = [k for k, v in enumerate(valid) if v][:n_mask_cols]
    for col_i, k in enumerate(valid_indices):
        ax_m = fig.add_subplot(gs_bot[0, col_i])
        ax_m.imshow(mask[k], cmap='gray', interpolation='nearest')
        ax_m.set_title(f"Mask {k}", fontsize=8)
        ax_m.axis('off')

    plt.show()
    plt.close(fig)


def _show_rgb_masks(rgb, mask, bbox, valid, title):
    """Compact view: RGB + all instance masks in a grid."""
    n_valid = valid.sum()
    cols = min(n_valid + 1, 6)
    rows = max(1, (n_valid + 1 + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).flatten()
    axes[0].imshow(rgb); axes[0].set_title("RGB"); axes[0].axis('off')
    slot = 1
    for k, (b, v) in enumerate(zip(bbox, valid)):
        if not v: continue
        axes[slot].imshow(mask[k], cmap='gray')
        c = b.mean(0)
        axes[slot].set_title(f"Obj {k}  c=[{c[0]:.2f},{c[1]:.2f},{c[2]:.2f}]", fontsize=7)
        axes[slot].axis('off')
        slot += 1
    for ax in axes[slot:]: ax.set_visible(False)
    fig.suptitle(title, fontsize=9); plt.tight_layout()
    plt.show(); plt.close(fig)


def get_sample_paths(split):
    root = Config.DATA_ROOT
    all_folders = sorted([f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))])
    n = len(all_folders)
    idx1 = int(n * Config.TRAIN_RATIO)
    idx2 = int(n * Config.VAL_RATIO)
    if   split == 'train': folders = all_folders[:idx1]
    elif split == 'val':   folders = all_folders[idx1:idx2]
    else:                  folders = all_folders[idx2:]
    return [os.path.join(root, f) for f in folders]


def main():
    parser = argparse.ArgumentParser(
        description="Interactively inspect 3D bounding box training data.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--idx',      type=int,  default=0,
                        help='Sample index within the chosen split (default: 0)')
    parser.add_argument('--count',    type=int,  default=1,
                        help='Number of consecutive samples to show (default: 1)')
    parser.add_argument('--random',   action='store_true',
                        help='Pick a random sample (overrides --idx)')
    parser.add_argument('--split',    default='train',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to inspect (default: train)')
    parser.add_argument('--rgb-only', action='store_true',
                        help='Show RGB + mask grid only (no 3D view)')
    args = parser.parse_args()

    paths = get_sample_paths(args.split)
    if not paths:
        print(f"No samples found for split '{args.split}'.")
        sys.exit(1)

    if args.random:
        import random
        start = random.randint(0, len(paths) - 1)
    else:
        start = args.idx % len(paths)

    for offset in range(args.count):
        i = (start + offset) % len(paths)
        inspect_sample(paths[i], sample_idx=i, split=args.split, rgb_only=args.rgb_only)


if __name__ == '__main__':
    main()
