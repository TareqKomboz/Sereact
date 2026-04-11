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
    print(f"Evaluating model: {os.path.basename(model_path)}")
    print(f"Device: {device}")
    print(f"Dataset: {root_dir}")
    
    if not os.path.exists(root_dir):
        print(f"Error: Dataset directory '{root_dir}' not found.")
        return

    ds = BBox3DDataset(root_dir, split="test")
    if len(ds) == 0:
        print("Error: Test split is empty.")
        return
        
    loader = DataLoader(ds, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    model = BBox3DModel().to(device)
    # Load with map_location to handle mps/cpu crossing
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    
    # 1. Run full test epoch to get average component metrics
    # h = {'total', 'center', 'size', 'orient', 'conf'}
    h = test_epoch(model, loader, device)
    
    print("\n" + "="*40)
    print("      TEST PERFORMANCE SUMMARY")
    print("="*40)
    print(f"Total Loss:        {h['total']:.6f}")
    print(f"Center Error (m):  {h['center']:.6f}")
    print(f"Size Error (m):    {h['size']:.6f}")
    print(f"Orient Dist (Tr):  {h['orient']:.6f}")
    print(f"Objectness (BCE):  {h['conf']:.6f}")
    print(f"Mask BCE:          {h['mask']:.6f}")
    print(f"3D IoU:            {h['iou']:.6f}")
    print(f"Corner RMSE (m):   {h['rmse']:.6f}")
    print("="*40)

    # 2. Visualize first batch sample
    sample = next(iter(loader))
    model.eval()
    with torch.no_grad():
        p_c, p_s, p_R, p_conf, p_s_log, p_mask = model(
            sample['pc'].to(device), 
            sample['obj_pc'].to(device),
            sample['obj_indices'].to(device), 
            sample['rgb'].to(device)
        )
        # Reconstruct corners for visualization mapping
        p_corners = model.reconstruct_corners(p_c, p_s, p_R)
    
    # Save comparison as PNG
    plot_comparison(
        sample['pc'][0].detach().cpu().numpy(), 
        sample['bbox'][0].detach().cpu().numpy(),
        p_corners[0].detach().cpu().numpy(),
        rgb=sample['rgb'][0].cpu().numpy(),
        conf=p_conf[0].detach().cpu().numpy(),
        pred_s=p_s[0].detach().cpu().numpy(),
        pred_orient=p_R[0].detach().cpu().numpy(),
        pred_mask=p_mask[0].detach().cpu().numpy(),
        save_path="eval_prediction.png"
    )
    print(f"Visualization saved to: eval_prediction.png")

if __name__ == '__main__':
    # Find the latest model in results/ if no specific path is given
    model_to_eval = None
    if os.path.exists("results"):
        folders = sorted([f for f in os.listdir("results") if f.startswith("run_")], reverse=True)
        for folder in folders:
            path = os.path.join("results", folder, "best_model.pth")
            if os.path.exists(path):
                model_to_eval = path
                break
    
    if model_to_eval:
        evaluate(model_to_eval)
    else:
        print("Error: Could not locate any model weights (.pth) in results/")
