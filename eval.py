import torch, os
from torch.utils.data import DataLoader
from dataset import BBox3DDataset
from model import BBox3DModel
from train import test_epoch
from visualize import plot_comparison
from config import Config

def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def evaluate(model_path, root_dir=Config.DATA_ROOT):
    device = get_device()
    ds = BBox3DDataset(root_dir, split="test")
    loader = DataLoader(ds, batch_size=Config.BATCH_SIZE)
    model = BBox3DModel().to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    loss, diou, l1 = test_epoch(model, loader, device)
    print(f"Eval Results -> DIoU: {diou:.4f} | L1: {l1:.4f}")
    
    sample = next(iter(loader))
    pc, mask, rgb = sample['pc'].to(device), sample['mask'].to(device), sample['rgb'].to(device)
    pred = model(pc, mask, rgb)[0]
    plot_comparison(sample['pc'][0].detach().numpy(), sample['bbox'][0].detach().numpy(),
                    pred.detach().cpu().numpy(),
                    rgb=sample['rgb'][0].numpy(),
                    title=f"Eval | DIoU={diou:.4f} | L1={l1:.4f}",
                    save_path="eval_prediction.png")

if __name__ == '__main__':
    # Use best model from last successful run as default
    best_model = "results/run_20260410_003940/best_model.pth"
    if os.path.exists(best_model):
        evaluate(best_model)
    else:
        print("Model file not found. Please provide a valid path.")
