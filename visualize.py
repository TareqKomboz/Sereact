"""
visualize.py — 3D bounding box plotting helpers.

All functions open an interactive rotatable Matplotlib window AND optionally
save to disk. Pass save_path=None to skip saving.
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# Edges connecting the 8 corners of an axis-aligned bounding box
BBOX_EDGES = [(0,1),(1,2),(2,3),(3,0),   # bottom face
              (4,5),(5,6),(6,7),(7,4),   # top face
              (0,4),(1,5),(2,6),(3,7)]   # vertical pillars


def _is_valid_box(b):
    """A box is valid when at least one corner has a non-zero coordinate."""
    return np.abs(b).sum() > 1e-6


def _draw_box(ax, b, color, linewidth=1.5, alpha=1.0):
    for i, j in BBOX_EDGES:
        ax.plot([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]],
                color=color, linewidth=linewidth, alpha=alpha)


# ---------------------------------------------------------------------------
# plot_result — ground-truth only, whole scene
# ---------------------------------------------------------------------------
def plot_result(pc, bboxes, title="Ground Truth", save_path='predictions.png'):
    """
    Scatter the full point cloud and draw all valid GT boxes.
    Opens an interactive window; saves to save_path if not None.
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pc[:,0], pc[:,1], pc[:,2], s=1, c='deepskyblue', alpha=0.4, depthshade=True)

    plotted = False
    for b in bboxes:
        if not _is_valid_box(b): continue
        _draw_box(ax, b, color='lime', linewidth=1.5)
        plotted = True
    if plotted:
        ax.plot([], [], color='lime', label='Ground Truth')
        ax.legend(fontsize=9)

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(title)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# plot_comparison — GT vs Prediction side-by-side with RGB image
# ---------------------------------------------------------------------------
def plot_comparison(pc, gt_bboxes, pred_bboxes, rgb=None, title="GT vs Prediction",
                    save_path='comparison.png'):
    """
    Left panel : RGB image (if provided).
    Middle panel: full point cloud with GT (green) and Prediction (red) boxes.
    Right panel : zoomed view centred on the first valid GT box.

    Opens an interactive rotatable window; saves to save_path if not None.
    """
    # ---- layout ----
    n_cols = 3 if rgb is not None else 2
    fig = plt.figure(figsize=(6 * n_cols, 7))
    gs = gridspec.GridSpec(1, n_cols, figure=fig)

    col = 0
    if rgb is not None:
        ax_img = fig.add_subplot(gs[0, col]); col += 1
        # rgb can be (H,W,3) uint8 or (3,H,W)
        img = np.array(rgb)
        if img.ndim == 3 and img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        ax_img.imshow(img.astype(np.uint8))
        ax_img.set_title("RGB Image"); ax_img.axis('off')

    # ---- full scene ----
    ax_full = fig.add_subplot(gs[0, col], projection='3d'); col += 1
    ax_full.scatter(pc[:,0], pc[:,1], pc[:,2], s=1, c='deepskyblue', alpha=0.3, depthshade=True)
    plotted_gt = plotted_pred = False
    for b in gt_bboxes:
        if not _is_valid_box(b): continue
        _draw_box(ax_full, b, color='lime', linewidth=2)
        plotted_gt = True
    for b in pred_bboxes:
        if not _is_valid_box(b): continue
        _draw_box(ax_full, b, color='tomato', linewidth=2)
        plotted_pred = True
    handles = []
    if plotted_gt:   handles.append(plt.Line2D([],[],color='lime',  label='Ground Truth'))
    if plotted_pred: handles.append(plt.Line2D([],[],color='tomato',label='Prediction'))
    if handles: ax_full.legend(handles=handles, fontsize=9)
    ax_full.set_xlabel('X'); ax_full.set_ylabel('Y'); ax_full.set_zlabel('Z')
    ax_full.set_title("Full Scene"); ax_full.set_box_aspect([1,1,1])

    # ---- zoomed view around first valid GT box ----
    ax_zoom = fig.add_subplot(gs[0, col], projection='3d')
    first_valid = next((b for b in gt_bboxes if _is_valid_box(b)), None)
    pad = 0.3
    if first_valid is not None:
        lo = first_valid.min(0) - pad
        hi = first_valid.max(0) + pad
        sel = (pc[:,0] >= lo[0]) & (pc[:,0] <= hi[0]) & \
              (pc[:,1] >= lo[1]) & (pc[:,1] <= hi[1]) & \
              (pc[:,2] >= lo[2]) & (pc[:,2] <= hi[2])
        pc_zoom = pc[sel] if sel.sum() > 10 else pc
    else:
        pc_zoom = pc
    ax_zoom.scatter(pc_zoom[:,0], pc_zoom[:,1], pc_zoom[:,2],
                    s=3, c='deepskyblue', alpha=0.6, depthshade=True)
    if first_valid is not None:
        _draw_box(ax_zoom, first_valid, color='lime', linewidth=2.5)
        # find matching predicted box (closest center)
        gt_c = first_valid.mean(0)
        best_pred, best_dist = None, float('inf')
        for b in pred_bboxes:
            if not _is_valid_box(b): continue
            d = np.linalg.norm(b.mean(0) - gt_c)
            if d < best_dist: best_dist, best_pred = d, b
        if best_pred is not None:
            _draw_box(ax_zoom, best_pred, color='tomato', linewidth=2.5)
    ax_zoom.set_xlabel('X'); ax_zoom.set_ylabel('Y'); ax_zoom.set_zlabel('Z')
    ax_zoom.set_title("Zoomed — Object 0"); ax_zoom.set_box_aspect([1,1,1])

    fig.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close(fig)
