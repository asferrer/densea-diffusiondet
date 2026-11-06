"""
Traceable inference-latency benchmark for DiffusionDet (PyTorch, batch 1).

Measures the full detector forward pass, DDIM sampling loop included, so the
number of denoising steps (SAMPLE_STEP) is part of the measurement.

Protocol: >=50 warmup iterations, >=300 timed iterations, torch.cuda.synchronize()
around every timed forward pass, reports mean / std / median / p95 and FPS.

Usage (inside the densea_train container):
    cd <repository root>
    python tools/benchmark_latency.py \
        --config-file output/resnet-50/config.yaml \
        --weights output/resnet-50/model_final.pth \
        --steps 1 --size 640 640
"""

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.modeling import build_model

from diffusiondet import add_diffusiondet_config
from diffusiondet.util.model_ema import add_model_ema_configs


def driver_version() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unavailable (nvidia-smi not reachable from container)"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--size", type=int, nargs=2, default=[640, 640], metavar=("H", "W"))
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--json", default="results/latency.json")
    args = parser.parse_args()

    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    add_model_ema_configs(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.DiffusionDet.SAMPLE_STEP = args.steps
    cfg.MODEL.DEVICE = "cuda"
    cfg.freeze()

    model = build_model(cfg)
    DetectionCheckpointer(model).load(args.weights)
    model.eval()

    h, w = args.size
    device = torch.device(cfg.MODEL.DEVICE)
    # Detectron2 meta-arch input: raw (unnormalised) image tensor in CHW, 0-255.
    image = torch.randint(0, 256, (3, h, w), dtype=torch.uint8).to(device).float()
    batched_inputs = [{"image": image, "height": h, "width": w}]

    with torch.no_grad():
        for _ in range(args.warmup):
            model(batched_inputs)
        torch.cuda.synchronize()

        timings = []
        for _ in range(args.iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(batched_inputs)
            torch.cuda.synchronize()
            timings.append((time.perf_counter() - t0) * 1000.0)

    mean = statistics.mean(timings)
    std = statistics.pstdev(timings)
    median = statistics.median(timings)
    p95 = sorted(timings)[int(0.95 * len(timings)) - 1]

    env = {
        "gpu": torch.cuda.get_device_name(0),
        "driver_version": driver_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "python": platform.python_version(),
        "amp": False,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
    }

    print("=" * 64)
    print("DiffusionDet inference latency (PyTorch FP32, batch 1)")
    print("=" * 64)
    print(f"Config:        {args.config_file}")
    print(f"Weights:       {args.weights}")
    print(f"Input:         1x3x{h}x{w}")
    print(f"SAMPLE_STEP:   {args.steps}")
    print(f"NUM_PROPOSALS: {cfg.MODEL.DiffusionDet.NUM_PROPOSALS}")
    print(f"Warmup iters:  {args.warmup}")
    print(f"Timed iters:   {args.iters}")
    print("-" * 64)
    for k, v in env.items():
        print(f"{k:16s} {v}")
    print("-" * 64)
    print(f"Mean latency:   {mean:8.3f} ms")
    print(f"Std dev:        {std:8.3f} ms")
    print(f"Median:         {median:8.3f} ms")
    print(f"p95:            {p95:8.3f} ms")
    print(f"Throughput:     {1000.0 / mean:8.2f} FPS")
    print("=" * 64)

    out = {
        "model": "DiffusionDet-R50",
        "config": args.config_file,
        "weights": args.weights,
        "sample_step": args.steps,
        "num_proposals": cfg.MODEL.DiffusionDet.NUM_PROPOSALS,
        "input_shape": [1, 3, h, w],
        "warmup_iters": args.warmup,
        "timed_iters": args.iters,
        "mean_ms": mean,
        "std_ms": std,
        "median_ms": median,
        "p95_ms": p95,
        "fps": 1000.0 / mean,
        "environment": env,
    }
    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"JSON saved to: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
