#!/usr/bin/env python3
"""Export the SRU high-level depth encoder to ONNX."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_SOURCE = REPO_ROOT / "deploy"
if str(DEPLOY_SOURCE) not in sys.path:
    sys.path.insert(0, str(DEPLOY_SOURCE))

from instinct_nav_onboard.depth_encoder import DepthEncoderRuntimeConfig, DepthNoiseEncoderRuntime  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export SRU high-level depth encoder ONNX.")
    parser.add_argument("--weights_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_name", type=str, default="sru_high_level_depth_encoder.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--add_noise", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights_path = Path(args.weights_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_name

    model = DepthNoiseEncoderRuntime(
        weights_path,
        DepthEncoderRuntimeConfig(add_noise=bool(args.add_noise)),
    ).eval().cpu()
    dummy_depth = torch.full((1, 1, 36, 64), 2.5, dtype=torch.float32)
    with torch.no_grad():
        encoded, filtered = model(dummy_depth)

    torch.onnx.export(
        model,
        dummy_depth,
        str(output_path),
        input_names=["depth_m"],
        output_names=["encoded_depth", "filtered_depth"],
        dynamic_axes={
            "depth_m": {0: "batch"},
            "encoded_depth": {0: "batch"},
            "filtered_depth": {0: "batch"},
        },
        opset_version=int(args.opset),
        do_constant_folding=True,
    )
    print(f"[INFO] Exported high-level depth encoder ONNX: {output_path}")
    print(f"[INFO] input depth shape: (1, 1, 36, 64)")
    print(f"[INFO] encoded shape: {tuple(encoded.shape)}")
    print(f"[INFO] filtered shape: {tuple(filtered.shape)}")
    print(f"[INFO] encoded mean/std: {float(encoded.mean()):.6f}/{float(encoded.std()):.6f}")


if __name__ == "__main__":
    main()
