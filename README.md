# Sereact 3D OBB Detection

This repository contains the current Sereact multimodal 3D bounding box pipeline. The model predicts valid oriented bounding boxes (OBBs) for up to 30 object slots per scene by fusing:

- a global scene point cloud stream
- a per-object point cloud stream extracted from instance masks
- a mask-pooled RGB feature stream

The README below reflects the code that is currently in this repository and the latest archived experiment results under `results/good/`.

## Current Pipeline

### Inputs

Each sample lives under `dl_challenge/<sample_id>/` and contains:

- `pc.npy`: structured point cloud in `(3, H, W)`
- `rgb.jpg`: RGB image
- `mask.npy`: per-instance masks
- `bbox3d.npy`: ground-truth 3D box corners in `(M, 8, 3)`

The dataset currently contains 200 samples, split by index into:

- train: 140
- val: 30
- test: 30

### Model

The current model in [`model.py`](model.py) is a three-stream fusion network:

1. Global point cloud encoder
   A PointNet-style MLP with batch norm processes `8192` scene points and max-pools them into a global feature.
2. Per-object point cloud encoder
   A second PointNet-style encoder processes `512` points per object slot. Batch norm is intentionally disabled here because many slots are padded.
3. RGB encoder
   A frozen `ResNet50` backbone extracts a spatial feature map from the resized `224 x 224` RGB image.

Per-object image features are produced by downsampling the instance mask and average-pooling only over the masked region. The three streams are concatenated into a `2560`-dim fusion vector per slot and decoded with specialized heads for:

- center residual
- log box size
- 6D orientation representation

The 6D orientation is converted to a valid rotation matrix with Gram-Schmidt orthonormalization, and corners are reconstructed analytically. This guarantees geometrically valid OBB predictions.

## Training Logic

### Supervision

The training loss in [`metrics.py`](metrics.py) is component-based rather than raw corner regression:

- center loss: L1 on box centroids
- size loss: L1 in log-dimension space
- orientation loss: symmetry-aware geodesic distance on `SO(3)`

Orientation loss is down-weighted for near-cubic boxes, where heading is poorly defined. During training and evaluation, the code also reports:

- 3D IoU using BEV-overlap times height-overlap
- corner RMSE in meters

### Augmentation

Train-time augmentation in [`dataset.py`](dataset.py) is synchronized across point cloud, masks, RGB, and per-object indices:

- random 90 degree rotations plus `+/- 7` degree jitter
- optional horizontal and vertical flips, currently disabled by default
- local per-object scale jitter
- global and per-object point jitter
- occasional RGB color inversion

### Optimization

The current default config in [`config.py`](config.py) uses:

- `AdamW`
- `OneCycleLR`
- batch size `16`
- max epochs `100`
- early stopping patience `40`
- frozen image backbone

The model has `30,486,476` parameters in total, with `6,978,444` trainable under the default frozen-backbone setup.

## Usage

### Install

```bash
pip install -r requirements.txt
```

### Train + final evaluation

```bash
python3 main.py
```

This creates a fresh `results/run_YYYYMMDD_HHMMSS/` directory with:

- `best_model.pth`
- `model.onnx` if ONNX export is enabled
- `logs/train.log`
- `logs/test.log`
- `visualizations/training_curves.png`
- `visualizations/test_prediction.png`

### Evaluate the latest run under `results/`

```bash
python3 eval.py
```

The script prints test metrics and saves `eval_prediction.png`.

### Export ONNX manually

```bash
python3 export_onnx.py --run_dir results/run_YYYYMMDD_HHMMSS
```

Optional controls:

- `ONNX_EXPORT=0` disables automatic export from `main.py`
- `ONNX_OPSET_VERSION=18` overrides the opset

### Weights & Biases

```bash
WANDB_ENABLED=1 WANDB_PROJECT=sereact-3d-detection python3 main.py
```

Optional environment variables:

- `WANDB_ENTITY=<team_or_username>`
- `WANDB_MODE=online|offline|disabled`
- `WANDB_TAGS=exp1,debug,augfix`
- `WANDB_WATCH_MODEL=1`
- `WANDB_LOG_ARTIFACTS=1`

### MMDetection3D baseline

```bash
python3 benchmarks/mmdet3d/convert_sereact_to_kitti.py --in_root dl_challenge --out_root data/sereact_kitti --clean
python3 benchmarks/mmdet3d/run_pointpillars_baseline.py --mmdet3d_root /path/to/mmdetection3d --kitti_root data/sereact_kitti --work_dir scratch/mmdet3d_pp_sereact --epochs 40 --batch_size 4 --run_test
```

See [benchmarks/mmdet3d/README.md](benchmarks/mmdet3d/README.md) for details.

## Latest Archived Results

The repository currently contains three archived runs under `results/good/`:

| Run | Date | Notes | Test metrics |
| --- | --- | --- | --- |
| `run_20260411_173236` | 2026-04-11 | Latest archived run with the current OBB metric stack | IoU `0.2014`, RMSE `0.0886 m`, Ctr `0.0232`, Sz `0.6064`, Orient `1.7683` |
| `run_20260411_165518` | 2026-04-11 | Best archived IoU among the current OBB runs | IoU `0.2216`, RMSE `0.0969 m`, Ctr `0.0159`, Sz `0.6150`, Orient `0.5448` |
| `run_20260410_175156` | 2026-04-10 | Older training objective, not directly comparable to the OBB runs above | Ctr `0.0129`, Sz `0.0380`, L1 `0.0591` |

The most representative current-result artifacts are:

- `results/good/run_20260411_165518/model.onnx`
- `results/good/run_20260411_165518/logs/test.log`
- `results/good/run_20260411_165518/visualizations/test_prediction.png`
- `results/good/run_20260411_165518/visualizations/training_curves.png`

## Repository Map

- [`main.py`](main.py): training entrypoint, logging, final evaluation, ONNX export
- [`dataset.py`](dataset.py): dataset loading, object extraction, synchronized augmentation
- [`model.py`](model.py): three-stream fusion model and OBB reconstruction
- [`metrics.py`](metrics.py): component losses and evaluation metrics
- [`eval.py`](eval.py): checkpoint evaluation utility
- [`export_onnx.py`](export_onnx.py): ONNX export
- [`architecture.md`](architecture.md): design notes
