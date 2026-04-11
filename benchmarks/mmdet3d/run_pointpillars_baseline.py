#!/usr/bin/env python3
"""
Run a quick MMDetection3D PointPillars baseline on converted Sereact KITTI data.

Expected:
  1) mmdetection3d repo is available locally.
  2) Dataset converted via convert_sereact_to_kitti.py.
"""

import argparse
import subprocess
from pathlib import Path


BASE_CONFIG_CANDIDATES = [
    "configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py",
    "configs/pointpillars/pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d.py",
]


def _run(cmd, cwd: Path, dry_run: bool = False):
    cmd_str = " ".join(cmd)
    print(f"[cmd] (cwd={cwd}) {cmd_str}")
    if not dry_run:
        subprocess.run(cmd, cwd=str(cwd), check=True)


def _resolve_base_config(mmdet3d_root: Path) -> Path:
    for rel in BASE_CONFIG_CANDIDATES:
        p = mmdet3d_root / rel
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find a supported base config in {mmdet3d_root}. "
        f"Tried: {BASE_CONFIG_CANDIDATES}"
    )


def _write_custom_config(base_cfg_path: Path, out_cfg_path: Path, data_root: Path, epochs: int, batch_size: int):
    text = f"""_base_ = ['{base_cfg_path.as_posix()}']

class_names = ('Object',)
metainfo = dict(classes=class_names)
data_root = '{data_root.as_posix()}/'

for _loader in [train_dataloader, val_dataloader, test_dataloader]:
    _loader['batch_size'] = {batch_size}
    _loader['dataset']['data_root'] = data_root
    _loader['dataset']['metainfo'] = metainfo

train_dataloader['dataset']['ann_file'] = 'sereact_infos_train.pkl'
val_dataloader['dataset']['ann_file'] = 'sereact_infos_val.pkl'
test_dataloader['dataset']['ann_file'] = 'sereact_infos_test.pkl'

train_dataloader['dataset']['data_prefix'] = dict(pts='training/velodyne')
val_dataloader['dataset']['data_prefix'] = dict(pts='training/velodyne')
test_dataloader['dataset']['data_prefix'] = dict(pts='training/velodyne')

model['bbox_head']['num_classes'] = 1

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs={epochs}, val_interval=1)
default_hooks['checkpoint'] = dict(type='CheckpointHook', interval=1, max_keep_ckpts=3, save_best='auto')

visualizer = dict(type='Det3DLocalVisualizer')
"""
    out_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    out_cfg_path.write_text(text)
    print(f"[info] Wrote config: {out_cfg_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mmdet3d_root", required=True, type=str, help="Local mmdetection3d repo root")
    parser.add_argument("--kitti_root", default="data/sereact_kitti", type=str, help="Converted KITTI-like root")
    parser.add_argument("--work_dir", default="scratch/mmdet3d_pp_sereact", type=str)
    parser.add_argument("--epochs", default=40, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--skip_prepare", action="store_true", help="Skip create_data.py")
    parser.add_argument("--run_test", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    mmdet3d_root = Path(args.mmdet3d_root).resolve()
    kitti_root = Path(args.kitti_root).resolve()
    work_dir = Path(args.work_dir).resolve()

    if not mmdet3d_root.exists():
        raise FileNotFoundError(f"mmdet3d_root not found: {mmdet3d_root}")
    if not kitti_root.exists():
        raise FileNotFoundError(f"kitti_root not found: {kitti_root}")

    base_cfg = _resolve_base_config(mmdet3d_root)
    cfg_path = work_dir / "pointpillars_sereact.py"
    _write_custom_config(base_cfg, cfg_path, kitti_root, args.epochs, args.batch_size)

    if not args.skip_prepare:
        _run(
            [
                "python3",
                "tools/create_data.py",
                "kitti",
                "--root-path",
                str(kitti_root),
                "--out-dir",
                str(kitti_root),
                "--extra-tag",
                "sereact",
            ],
            cwd=mmdet3d_root,
            dry_run=args.dry_run,
        )

    _run(
        [
            "python3",
            "tools/train.py",
            str(cfg_path),
            "--work-dir",
            str(work_dir),
        ],
        cwd=mmdet3d_root,
        dry_run=args.dry_run,
    )

    if args.run_test:
        ckpt = work_dir / "best_auto.pth"
        if not ckpt.exists():
            ckpt = work_dir / "latest.pth"
        _run(
            [
                "python3",
                "tools/test.py",
                str(cfg_path),
                str(ckpt),
            ],
            cwd=mmdet3d_root,
            dry_run=args.dry_run,
        )

    print("[done] MMDetection3D baseline command sequence finished.")


if __name__ == "__main__":
    main()
