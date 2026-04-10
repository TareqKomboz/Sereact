"""
visualize.py — 5-panel training visualizations.
"""

import os
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from config import Config

def _is_valid_box(bbox):
    return np.abs(bbox).sum() > 1e-6

def _draw_box(ax, corners, color='b', linewidth=1, alpha=0.8):
    edge_idx = [
        (0,1), (1,2), (2,3), (3,0),
        (4,5), (5,6), (6,7), (7,4),
        (0,4), (1,5), (2,6), (3,7)
    ]
    for i, j in edge_idx:
        ax.plot([corners[i, 0], corners[j, 0]],
                [corners[i, 1], corners[j, 1]],
                [corners[i, 2], corners[j, 2]], color=color, linewidth=linewidth, alpha=alpha)

def plot_result(pc, bbox_gt, bbox_pred, conf_logits=None, title="3D Detection Result"):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=1, c=pc[:, 2], cmap='viridis', alpha=0.3)
    
    for i in range(bbox_gt.shape[0]):
        if _is_valid_box(bbox_gt[i]):
            _draw_box(ax, bbox_gt[i], color='lime', linewidth=2)
    
    for i in range(bbox_pred.shape[0]):
        if not _is_valid_box(bbox_pred[i]): continue
        if conf_logits is not None:
            # Filter by confidence if available
            score = 1.0 / (1.0 + np.exp(-conf_logits[i]))
            if score < Config.CONF_THRESHOLD: continue
        _draw_box(ax, bbox_pred[i], color='tomato', linewidth=2)
            
    ax.set_title(title)
    plt.show()

def plot_comparison(pc, gt_bboxes, pred_bboxes, rgb=None, conf=None, title="Comparison", save_path=None):
    """
    2-Panel visualization with Confidence Filtering.
    """
    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2])
    
    # 1. RGB View
    ax_rgb = fig.add_subplot(gs[0])
    if rgb is not None:
        if rgb.shape[0] == 3: rgb = rgb.transpose(1, 2, 0)
        ax_rgb.imshow(rgb.astype(np.uint8))
        ax_rgb.set_title("Input RGB")
    ax_rgb.axis('off')
    
    # 2. 3D View
    ax_3d = fig.add_subplot(gs[1], projection='3d')
    ax_3d.scatter(pc[::2, 0], pc[::2, 1], pc[::2, 2], s=0.5, c=pc[::2, 2], cmap='Blues', alpha=0.2)
    
    for b in gt_bboxes:
        if _is_valid_box(b): _draw_box(ax_3d, b, color='lime', linewidth=1.5)
    
    # Filter predictions by confidence threshold
    scores = 1.0 / (1.0 + np.exp(-conf)) if conf is not None else np.ones(len(pred_bboxes))
    
    for b, s in zip(pred_bboxes, scores):
        if _is_valid_box(b) and s >= Config.CONF_THRESHOLD:
            _draw_box(ax_3d, b, color='tomato', linewidth=1.5)
        
    ax_3d.set_title(f"3D Detection Score > {Config.CONF_THRESHOLD}")
    ax_3d.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    
    # Non-blocking show
    if os.environ.get('DISPLAY') or matplotlib.get_backend().lower() == 'macosx':
        plt.show()
    plt.close(fig)

def plot_training_curves(history: dict, save_path: str = None, title: str = "Training Curves"):
    """5-Panel visualization without warmup shading."""
    epochs = history['epoch']
    if not epochs: return

    panels = [
        ('total',  'Weighted Total Loss'),
        ('center', 'Centroid Error (m)'),
        ('size',   'Dimension Error (m)'),
        ('orient', 'Rotation Error (L1 Axes)'),
        ('conf',   'Objectness (BCE)'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i >= len(panels):
            ax.axis('off')
            continue
            
        m, label = panels[i]
        ax.plot(epochs, history[f'train_{m}'], color='#4A90D9', lw=2, label='Train')
        ax.plot(epochs, history[f'val_{m}'],   color='#E05A4E', lw=2, label='Val', ls='--')
        
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, ls=':')
        ax.legend(fontsize=9)
        ax.set_xlim(left=-0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if os.environ.get('DISPLAY') or matplotlib.get_backend().lower() == 'macosx':
        plt.show()
    plt.close(fig)
