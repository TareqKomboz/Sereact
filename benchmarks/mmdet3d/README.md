# MMDetection3D Baseline (PointPillars)

This folder provides a **benchmark path**, not a full migration.

## 1) Convert dataset to KITTI-like layout

From repo root:

```bash
python3 benchmarks/mmdet3d/convert_sereact_to_kitti.py \
  --in_root dl_challenge \
  --out_root data/sereact_kitti \
  --clean
```

## 2) Run MMDetection3D baseline

Assuming local `mmdetection3d` checkout at `/path/to/mmdetection3d`:

```bash
python3 benchmarks/mmdet3d/run_pointpillars_baseline.py \
  --mmdet3d_root /path/to/mmdetection3d \
  --kitti_root data/sereact_kitti \
  --work_dir scratch/mmdet3d_pp_sereact \
  --epochs 40 \
  --batch_size 4 \
  --run_test
```

The runner will:
1. Generate a custom config (`pointpillars_sereact.py`)
2. Run `tools/create_data.py kitti ...` to build info files
3. Run `tools/train.py ...`
4. Optionally run `tools/test.py ...`

## Notes

- This is for **relative benchmarking** against your in-repo model.
- Conversion uses one class (`Object`) and identity calibration placeholders.
- Keep this baseline separate from your custom multimodal pipeline; compare validation metrics before deciding on a full rewrite.

