#!/usr/bin/env python3
"""
Convert Sereact dataset (pc.npy, bbox3d.npy, rgb.jpg) to a KITTI-like layout.

Output layout:
  <out_root>/
    training/
      velodyne/*.bin
      image_2/*.png
      calib/*.txt
      label_2/*.txt
    ImageSets/
      train.txt
      val.txt
      test.txt
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def _ordered_corners_to_center_size_yaw(corners: np.ndarray):
    """
    corners: (8, 3), ordered as:
      0(-x,-y,-z), 1(+x,-y,-z), 3(-x,+y,-z), 4(-x,-y,+z)
    Returns:
      center(3), size_xyz(3), yaw_z
    """
    center = corners.mean(axis=0)
    e1 = corners[1] - corners[0]  # +x edge
    e2 = corners[3] - corners[0]  # +y edge
    e3 = corners[4] - corners[0]  # +z edge

    sx = np.linalg.norm(e1)
    sy = np.linalg.norm(e2)
    sz = np.linalg.norm(e3)

    # Yaw around z from horizontal major axis.
    if sx >= sy:
        axis_xy = e1[:2]
        length = sx
        width = sy
    else:
        axis_xy = e2[:2]
        length = sy
        width = sx
    yaw = float(np.arctan2(axis_xy[1], axis_xy[0]))
    return center, np.array([sx, sy, sz], dtype=np.float32), float(length), float(width), float(sz), yaw


def _write_identity_calib(path: Path):
    p = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32).reshape(-1)
    r0 = np.eye(3, dtype=np.float32).reshape(-1)
    tr = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32).reshape(-1)
    with path.open("w") as f:
        f.write("P0: " + " ".join(map(str, p.tolist())) + "\n")
        f.write("P1: " + " ".join(map(str, p.tolist())) + "\n")
        f.write("P2: " + " ".join(map(str, p.tolist())) + "\n")
        f.write("P3: " + " ".join(map(str, p.tolist())) + "\n")
        f.write("R0_rect: " + " ".join(map(str, r0.tolist())) + "\n")
        f.write("Tr_velo_to_cam: " + " ".join(map(str, tr.tolist())) + "\n")
        f.write("Tr_imu_to_velo: " + " ".join(map(str, tr.tolist())) + "\n")


def convert(in_root: Path, out_root: Path, class_name: str, train_ratio: float, val_ratio: float, max_samples: int):
    samples = sorted([p for p in in_root.iterdir() if p.is_dir()])
    if max_samples > 0:
        samples = samples[:max_samples]
    if not samples:
        raise RuntimeError(f"No sample folders found in {in_root}")

    idx1 = int(len(samples) * train_ratio)
    idx2 = int(len(samples) * val_ratio)
    splits = {
        "train": samples[:idx1],
        "val": samples[idx1:idx2],
        "test": samples[idx2:],
    }

    train_dir = out_root / "training"
    velodyne_dir = train_dir / "velodyne"
    image_dir = train_dir / "image_2"
    calib_dir = train_dir / "calib"
    label_dir = train_dir / "label_2"
    imagesets_dir = out_root / "ImageSets"
    for d in [velodyne_dir, image_dir, calib_dir, label_dir, imagesets_dir]:
        d.mkdir(parents=True, exist_ok=True)

    all_ids = []
    split_ids = {"train": [], "val": [], "test": []}

    for i, sample_dir in enumerate(samples):
        sample_id = f"{i:06d}"
        all_ids.append(sample_id)
        if sample_dir in splits["train"]:
            split_ids["train"].append(sample_id)
        elif sample_dir in splits["val"]:
            split_ids["val"].append(sample_id)
        else:
            split_ids["test"].append(sample_id)

        # Point cloud: (3, H, W) -> (N, 4)
        pc = np.load(sample_dir / "pc.npy").astype(np.float32)  # (3,H,W)
        pts = pc.reshape(3, -1).T
        finite = np.isfinite(pts).all(axis=1)
        pts = pts[finite]
        intensity = np.ones((pts.shape[0], 1), dtype=np.float32)
        pts4 = np.concatenate([pts, intensity], axis=1)
        pts4.tofile(velodyne_dir / f"{sample_id}.bin")

        # RGB -> png
        img_src = sample_dir / "rgb.jpg"
        img_dst = image_dir / f"{sample_id}.png"
        if img_src.exists():
            Image.open(img_src).save(img_dst)
        else:
            Image.fromarray(np.zeros((375, 1242, 3), dtype=np.uint8)).save(img_dst)

        # Identity calib
        _write_identity_calib(calib_dir / f"{sample_id}.txt")

        # Labels
        bbox = np.load(sample_dir / "bbox3d.npy").astype(np.float32)  # (M,8,3)
        valid = np.abs(bbox).sum(axis=(1, 2)) > 1e-6
        with (label_dir / f"{sample_id}.txt").open("w") as f:
            for b in bbox[valid]:
                center, size_xyz, length, width, height, yaw = _ordered_corners_to_center_size_yaw(b)

                # KITTI label format (camera-like placeholders for 2D fields).
                # h, w, l, x, y, z, ry
                line = (
                    f"{class_name} 0 0 0 "
                    f"0 0 50 50 "
                    f"{height:.6f} {width:.6f} {length:.6f} "
                    f"{center[0]:.6f} {center[1]:.6f} {center[2]:.6f} "
                    f"{yaw:.6f}\n"
                )
                f.write(line)

    for split_name in ["train", "val", "test"]:
        with (imagesets_dir / f"{split_name}.txt").open("w") as f:
            for sid in split_ids[split_name]:
                f.write(sid + "\n")

    with (imagesets_dir / "trainval.txt").open("w") as f:
        for sid in split_ids["train"] + split_ids["val"]:
            f.write(sid + "\n")

    print(f"Converted {len(samples)} samples to {out_root}")
    print({k: len(v) for k, v in split_ids.items()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_root", type=str, default="dl_challenge")
    parser.add_argument("--out_root", type=str, default="data/sereact_kitti")
    parser.add_argument("--class_name", type=str, default="Object")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.85)
    parser.add_argument("--max_samples", type=int, default=0, help="0 = all samples")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    in_root = Path(args.in_root).resolve()
    out_root = Path(args.out_root).resolve()
    if args.clean and out_root.exists():
        shutil.rmtree(out_root)

    convert(
        in_root=in_root,
        out_root=out_root,
        class_name=args.class_name,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
