import torch
import os
import argparse
from model import BBox3DModel
from config import Config

def export_to_onnx(model_path, onnx_path, device="cpu"):
    """
    Exports the BBox3DModel to ONNX format for deployment.
    """
    print(f"Loading model from {model_path}...")
    model = BBox3DModel().to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # Create dummy inputs matching the forward() signature
    # (B, N, 3), (B, M, N_p, 3), (B, M, H, W), (B, 3, H, W)
    batch_size = 1
    dummy_pc     = torch.randn(batch_size, Config.NUM_POINTS, 3).to(device)
    dummy_obj_pc = torch.randn(batch_size, Config.MAX_OBJECTS, Config.N_OBJ_POINTS, 3).to(device)
    dummy_mask   = torch.randn(batch_size, Config.MAX_OBJECTS, Config.IMG_SIZE[0], Config.IMG_SIZE[1]).to(device)
    dummy_rgb    = torch.randint(0, 255, (batch_size, 3, Config.IMG_SIZE[0], Config.IMG_SIZE[1])).to(device).byte()

    print(f"Exporting to {onnx_path}...")
    torch.onnx.export(
        model,
        (dummy_pc, dummy_obj_pc, dummy_mask, dummy_rgb),
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['pc', 'obj_pc', 'mask', 'rgb'],
        output_names=['center', 'size', 'R', 'conf', 'log_size'],
        dynamic_axes={
            'pc':     {0: 'batch_size'},
            'obj_pc': {0: 'batch_size'},
            'mask':   {0: 'batch_size'},
            'rgb':    {0: 'batch_size'},
            'center': {0: 'batch_size'},
            'size':   {0: 'batch_size'},
            'R':      {0: 'batch_size'},
            'conf':   {0: 'batch_size'}
        }
    )
    print("Export complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True, help="Path to the results/run_xxx directory")
    args = parser.parse_args()

    model_file = os.path.join(args.run_dir, "best_model.pth")
    output_file = os.path.join(args.run_dir, "model.onnx")
    
    if os.path.exists(model_file):
        export_to_onnx(model_file, output_file)
    else:
        print(f"Error: {model_file} not found.")
