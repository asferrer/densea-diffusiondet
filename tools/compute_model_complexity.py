"""
Compute model complexity (parameters, FLOPs, model size) for DiffusionDet.

Profiles backbone (ResNet+FPN) and head separately to avoid issues with
DiffusionDet's dict-based input format and diffusion sampling loop.

Usage (inside Docker container):
    cd <repository root>
    python tools/compute_model_complexity.py \
        --config-file output/resnet-50/config.yaml \
        --weights output/resnet-50/model_final.pth
"""

import argparse
import os
import sys

import torch
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from fvcore.nn import FlopCountAnalysis

# Add DiffusionDet to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from diffusiondet import add_diffusiondet_config
from diffusiondet.util.model_ema import add_model_ema_configs


def profile_submodule(module, dummy_input, name):
    """Profile a submodule with fvcore, fallback to torch profiler."""
    try:
        flops = FlopCountAnalysis(module, (dummy_input,))
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        total = flops.total()
        print(f"  {name}: {total / 1e9:.2f} GFLOPs (fvcore)")
        return total
    except Exception:
        pass

    # Fallback: torch profiler
    try:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            with_flops=True,
        ) as prof:
            with torch.no_grad():
                module(dummy_input)

        total = sum(e.flops for e in prof.key_averages() if e.flops is not None)
        print(f"  {name}: {total / 1e9:.2f} GFLOPs (torch profiler)")
        return total
    except Exception as e:
        print(f"  {name}: FAILED ({e})")
        return 0


def compute_complexity(config_file, weights_path, input_size=640):
    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    add_model_ema_configs(cfg)
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = weights_path

    model = build_model(cfg)
    model.eval()
    device = cfg.MODEL.DEVICE

    # Parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Per-component parameter count
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    head_params = sum(p.numel() for p in model.head.parameters())

    # FLOPs: profile backbone and head separately
    dummy_img = torch.randn(1, 3, input_size, input_size, device=device)

    print("\nProfiling components:")

    # 1) Backbone (ResNet + FPN)
    backbone_flops = profile_submodule(model.backbone, dummy_img, "Backbone (ResNet+FPN)")

    # 2) Head - needs feature maps from backbone
    with torch.no_grad():
        features = model.backbone(dummy_img)

    # The head expects a list of feature tensors
    feature_list = [features[k] for k in sorted(features.keys())]

    head_flops = 0
    try:
        # Head expects: features as list of tensors, init_bboxes, t, init_features
        num_proposals = cfg.MODEL.DiffusionDet.NUM_PROPOSALS
        # Head uses ROI_HEADS.IN_FEATURES (p2-p5), not all FPN levels (p2-p6)
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES  # e.g. ["p2","p3","p4","p5"]
        feature_list = [features[k] for k in in_features]
        dummy_boxes = torch.randn(1, num_proposals, 4, device=device).clamp(0, input_size)
        t = torch.zeros(1, device=device, dtype=torch.long)

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            with_flops=True,
        ) as prof:
            with torch.no_grad():
                model.head(feature_list, dummy_boxes, t, None)

        head_flops = sum(e.flops for e in prof.key_averages() if e.flops is not None)
        print(f"  Head (DiffusionDet): {head_flops / 1e9:.2f} GFLOPs (torch profiler)")
    except Exception as e:
        print(f"  Head (DiffusionDet): FAILED ({e})")

    total_flops = backbone_flops + head_flops

    # Model size
    model_size_mb = None
    if os.path.isfile(weights_path):
        model_size_mb = os.path.getsize(weights_path) / (1024 * 1024)

    # Print results
    print("\n" + "=" * 60)
    print("DiffusionDet Model Complexity")
    print("=" * 60)
    print(f"Config:              {config_file}")
    print(f"Input size:          {input_size}x{input_size}")
    print(f"Num proposals:       {cfg.MODEL.DiffusionDet.NUM_PROPOSALS}")
    print("-" * 60)
    print(f"Total params:        {total_params:,} ({total_params / 1e6:.1f}M)")
    print(f"  Backbone params:   {backbone_params:,} ({backbone_params / 1e6:.1f}M)")
    print(f"  Head params:       {head_params:,} ({head_params / 1e6:.1f}M)")
    print(f"Trainable params:    {trainable_params:,} ({trainable_params / 1e6:.1f}M)")
    print("-" * 60)
    print(f"Backbone GFLOPs:     {backbone_flops / 1e9:.2f}")
    print(f"Head GFLOPs:         {head_flops / 1e9:.2f}")
    print(f"Total GFLOPs:        {total_flops / 1e9:.2f}")
    if model_size_mb is not None:
        print(f"Weights file size:   {model_size_mb:.1f} MB")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute DiffusionDet model complexity"
    )
    parser.add_argument(
        "--config-file", type=str, required=True, help="Detectron2 config YAML"
    )
    parser.add_argument(
        "--weights", type=str, required=True, help="Model weights .pth"
    )
    parser.add_argument(
        "--input-size", type=int, default=640, help="Input resolution"
    )
    args = parser.parse_args()

    compute_complexity(args.config_file, args.weights, args.input_size)
