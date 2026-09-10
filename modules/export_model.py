"""
Re-exports an already-trained generator checkpoint (.pt) as a single,
self-contained .onnx file — no external .onnx.data companion file needed.

Use this if train_gan.py already finished and saved a checkpoint, but you
either lost/forgot the companion .onnx.data file, or just want a simpler
single-file deployment (recommended — matches how every other model in
this project is packaged).

Run this in the SAME Colab session where training just finished (it needs
the .pt checkpoint file, which only exists in that session's storage).

Usage:
    python export_model.py --checkpoint ./gan_checkpoints/generator_epoch30.pt --output ./gan_checkpoints/boundary_refinement_gan.onnx
"""

import argparse
import torch
from gan_architecture import BoundaryRefinementGenerator, PATCH_SIZE


def export(checkpoint_path, output_path):
    device = "cpu"  # export doesn't need GPU, and keeps the exported graph device-agnostic
    generator = BoundaryRefinementGenerator()
    generator.load_state_dict(torch.load(checkpoint_path, map_location=device))
    generator.eval()

    dummy_input = torch.randn(1, 4, PATCH_SIZE, PATCH_SIZE)

    # dynamo=False forces the traditional exporter, which embeds all
    # weights directly in the .onnx file — a single portable file, no
    # external .onnx.data companion to lose track of.
    torch.onnx.export(
        generator, dummy_input, output_path,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=13,
        dynamo=False,
    )
    print(f"Exported single-file ONNX model to: {output_path}")
    print("This file is self-contained — download just this one file, "
          "no companion .data file needed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="./gan_checkpoints/boundary_refinement_gan.onnx")
    args = parser.parse_args()

    export(args.checkpoint, args.output)