#!/usr/bin/env python3
"""Export Sapiens TorchScript model to ONNX for TensorRT conversion."""

import torch

MODEL_PATH = "/var/lib/mocap-toolkit/sapiens_0.3b_goliath_best_goliath_AP_573_torchscript.pt2"
OUTPUT_PATH = "/var/lib/mocap-toolkit/sapiens.onnx"
BATCH_SIZE = 1
CHANNELS = 3
HEIGHT = 1024
WIDTH = 768

def main():
    print(f"Loading model from {MODEL_PATH}...")
    model = torch.jit.load(MODEL_PATH, map_location="cuda")
    model.eval()

    dummy_input = torch.randn(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH, device="cuda")

    print(f"Exporting to ONNX with shape [{BATCH_SIZE}, {CHANNELS}, {HEIGHT}, {WIDTH}]...")
    # Use the legacy TorchScript export path
    torch.onnx.export(
        model,
        dummy_input,
        OUTPUT_PATH,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
