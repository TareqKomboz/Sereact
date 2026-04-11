# 3D Bounding Box Detection Pipeline

This repository contains a deep learning pipeline for detecting 3D bounding boxes from multimodal data (Point Clouds and RGB images). The project implements an early-fusion architecture to leverage both spatial geometry and visual textures.

## 🏗 Architecture

The system utilizes an **Early-Fusion Multimodal Architecture**:

- **Point Cloud Encoder**: A PointNet-style MLP that maps `[X, Y, Z]` coordinates to a high-dimensional feature space, followed by global max pooling to ensure permutation invariance.
- **RGB Image Encoder**: A standard 2D CNN (ResNet-18) that extracts texture and local contextual features.
- **Fusion & Decoder**: Features from both encoders are concatenated into a single dense representation and decoded into coordinate predictions for 60 potential bounding boxes, each defined by 8 3D corner points.

## 📊 Metrics & Loss

We employ a dual-metric approach for training and evaluation:

1.  **DIoU Loss (Distance-IoU)**: The primary differentiable loss function that optimizes for volume overlap and center-point localization simultaneously.
2.  **3D IoU (Intersection Over Union)**: The primary interpretive metric for assessing volumetric overlap between predictions and ground truth.
3.  **L1 Loss**: Used for stable coordinate regression during training.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- NumPy
- Matplotlib

### Installation

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Dataset Structure

The pipeline expects data in a `./dl_challenge` directory. Each sample should be in its own subdirectory containing:
- `pc.npy`: Point cloud data.
- `rgb.jpg`: RGB image of the scene.
- `mask.npy`: Instance masks.
- `bbox3d.npy`: Ground truth 3D bounding box coordinates.

## 💻 Usage

### Full Pipeline (Train + Eval)

To run the complete training loop followed by a final evaluation on the test set:
```bash
python main.py
```
This script will:
- Set up a unique run directory in `results/`.
- Train the model with early stopping.
- Log metrics to `results/run_YYYYMMDD_HHMMSS/logs/`.
- Save visualizations of test predictions.
- Export `best_model.pth` to `model.onnx` (enabled by default).

### Weights & Biases Tracking

This project includes native W&B logging in `main.py`.

1. Install dependencies (includes `wandb`):
   ```bash
   pip install -r requirements.txt
   ```
2. Authenticate once:
   ```bash
   wandb login
   ```
3. Run training with tracking enabled:
   ```bash
   WANDB_ENABLED=1 WANDB_PROJECT=sereact-3d-detection python3 main.py
   ```

Optional environment variables:
- `WANDB_ENTITY=<team_or_username>`
- `WANDB_MODE=online|offline|disabled`
- `WANDB_TAGS=exp1,debug,augfix`
- `WANDB_WATCH_MODEL=1`
- `WANDB_LOG_ARTIFACTS=1`

### Standalone Evaluation

To evaluate a specific trained model:
1. Open `eval.py`.
2. Update the `best_model` path to point to your saved `.pth` file.
3. Run the script:
   ```bash
   python eval.py
   ```

### Standalone ONNX Export

To export a trained run checkpoint manually:
```bash
python export_onnx.py --run_dir results/run_YYYYMMDD_HHMMSS
```

Optional controls:
- `ONNX_EXPORT=0` to disable automatic export in `main.py`.
- `ONNX_OPSET_VERSION=18` to override ONNX opset.

### MMDetection3D Baseline Benchmark

To benchmark against a high-level library without migrating this repo:

```bash
python3 benchmarks/mmdet3d/convert_sereact_to_kitti.py --in_root dl_challenge --out_root data/sereact_kitti --clean
python3 benchmarks/mmdet3d/run_pointpillars_baseline.py --mmdet3d_root /path/to/mmdetection3d --kitti_root data/sereact_kitti --work_dir scratch/mmdet3d_pp_sereact --epochs 40 --batch_size 4 --run_test
```

Detailed guide: [benchmarks/mmdet3d/README.md](/Users/tareqabuelkomboz/Documents/Career/2_University/Promotion/Projects/Sereact/benchmarks/mmdet3d/README.md)

## 📂 Results

- **Models**: The best model weights are saved as `best_model.pth` within the run directory.
- **Logs**: Training and testing logs are stored in the `logs/` folder.
- **Visualizations**: Representative 3D wireframe plots comparing predictions against ground truth are saved in the `visualizations/` folder.
