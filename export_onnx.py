import torch
import os
import argparse
from model import BBox3DModel
from config import Config

def export_to_onnx(model_path, onnx_path, device="cpu", opset_version=Config.ONNX_OPSET_VERSION):
    """
    Exports the BBox3DModel to ONNX format for deployment.
    """
    onnx_dir = os.path.dirname(onnx_path)
    if onnx_dir:
        os.makedirs(onnx_dir, exist_ok=True)
    print(f"Loading model from {model_path}...")
    model = BBox3DModel().to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: missing keys while loading checkpoint: {missing}")
    if unexpected:
        print(f"Warning: unexpected keys while loading checkpoint: {unexpected}")
    model.eval()

    # Create dummy inputs matching the forward() signature
    # (B, N, 3), (B, M, N_p, 3), (B, M, N_p, 2), (B, 3, H, W), (B, M, H, W)
    batch_size = 1
    dummy_pc     = torch.randn(batch_size, Config.NUM_POINTS, 3).to(device)
    dummy_obj_pc = torch.randn(batch_size, Config.MAX_OBJECTS, Config.N_OBJ_POINTS, 3).to(device)
    dummy_obj_indices = torch.zeros(batch_size, Config.MAX_OBJECTS, Config.N_OBJ_POINTS, 2).to(device)
    dummy_obj_indices[..., 0] = torch.randint(
        low=0, high=Config.IMG_SIZE[1], size=(batch_size, Config.MAX_OBJECTS, Config.N_OBJ_POINTS), device=device
    )
    dummy_obj_indices[..., 1] = torch.randint(
        low=0, high=Config.IMG_SIZE[0], size=(batch_size, Config.MAX_OBJECTS, Config.N_OBJ_POINTS), device=device
    )
    dummy_rgb    = torch.randint(0, 255, (batch_size, 3, Config.IMG_SIZE[0], Config.IMG_SIZE[1])).to(device).byte()
    dummy_mask   = torch.randint(
        low=0, high=2, size=(batch_size, Config.MAX_OBJECTS, Config.IMG_SIZE[0], Config.IMG_SIZE[1]), device=device
    ).float()

    print(f"Exporting to {onnx_path} (opset={opset_version})...")
    torch.onnx.export(
        model,
        (dummy_pc, dummy_obj_pc, dummy_obj_indices, dummy_rgb, dummy_mask),
        onnx_path,
        dynamo=False,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['pc', 'obj_pc', 'obj_indices', 'rgb', 'mask'],
        output_names=['center', 'size', 'R', 'log_size'],
        dynamic_axes={
            'pc':     {0: 'batch_size'},
            'obj_pc': {0: 'batch_size'},
            'obj_indices': {0: 'batch_size'},
            'rgb':    {0: 'batch_size'},
            'mask':   {0: 'batch_size'},
            'center': {0: 'batch_size'},
            'size':   {0: 'batch_size'},
            'R':      {0: 'batch_size'},
            'log_size': {0: 'batch_size'},
        }
    )
    print("Export complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True, help="Path to the results/run_xxx directory")
    parser.add_argument("--device", type=str, default="cpu", help="Device used for export graph tracing.")
    parser.add_argument("--opset", type=int, default=Config.ONNX_OPSET_VERSION, help="ONNX opset version.")
    args = parser.parse_args()

    model_file = os.path.join(args.run_dir, "best_model.pth")
    output_file = os.path.join(args.run_dir, "model.onnx")
    
    if os.path.exists(model_file):
        export_to_onnx(model_file, output_file, device=args.device, opset_version=args.opset)
    else:
        print(f"Error: {model_file} not found.")
