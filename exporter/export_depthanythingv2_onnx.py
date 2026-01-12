# pth_to_onnx.py
import torch
import argparse
import os
import sys

from depth_anything_v2.dpt import DepthAnythingV2


def find_local_checkpoint():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pths = [f for f in os.listdir(script_dir) if f.endswith(".pth")]

    if len(pths) == 0:
        print("ERROR: No .pth file found in the script directory.")
        sys.exit(1)

    if len(pths) > 1:
        print("ERROR: Multiple .pth files found. Please specify --checkpoint explicitly:")
        for p in pths:
            print("  ", p)
        sys.exit(1)

    return os.path.join(script_dir, pths[0])


def export(checkpoint, encoder, output, input_size=518, opset=12):
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
    }

    cfg = model_configs[encoder]
    model = DepthAnythingV2(**cfg)

    print("Loading weights:", checkpoint)
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, 3, input_size, input_size)

    print("Exporting to ONNX:", output)
    torch.onnx.export(
        model,
        dummy,
        output,
        opset_version=opset,
        input_names=["input"],
        output_names=["depth"],
        dynamic_axes={
            "input": {0: "batch", 2: "h", 3: "w"},
            "depth": {0: "batch", 2: "h", 3: "w"},
        },
        do_constant_folding=True,
    )

    print("Export completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None, help="path to .pth checkpoint")
    parser.add_argument("--encoder", default="vits", choices=["vits", "vitb", "vitl", "vitg"])
    parser.add_argument("--output", default="depth_anything_v2.onnx")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--opset", type=int, default=12)

    args = parser.parse_args()

    checkpoint = args.checkpoint or find_local_checkpoint()

    export(
        checkpoint=checkpoint,
        encoder=args.encoder,
        output=args.output,
        input_size=args.input_size,
        opset=args.opset,
    )
