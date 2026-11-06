# Changes with respect to upstream

Upstream: [ShoufaChen/DiffusionDet](https://github.com/ShoufaChen/DiffusionDet) at commit
`18b528f8f84ee6fbf872859fbacf77208413ec81`. The first commit of this repository is that tree, unchanged. This file
lists the files of upstream that were modified, added or removed.

## Modified files

| File | Change |
|---|---|
| `train_net.py` -> `train_net_densea.py` | Rewritten trainer (see the README): registration of the DenSea-Aggregated splits, AMP and gradient accumulation, EMA evaluation, `DENSEA.*` options (best-model metric, early stopping, sparse periodic checkpoints), resume from the evaluation history, scheduler hooks registered once, training mapper rescaling by `INPUT.MIN_SIZE_TRAIN`. |
| `diffusiondet/config.py` | New options `SOLVER.AMP.*` and `SOLVER.GRAD_ACCUM_STEPS`. |
| `diffusiondet/dataset_mapper.py` | Albumentations-based augmentation for underwater images; options driven by the config. |
| `diffusiondet/detector.py` | Optional `DENSEA_FAST_LOSS`: skips the NaN/inf warning checks that synchronise with the GPU. |
| `diffusiondet/head.py` | Box decoding in fp32 (`apply_deltas`), scale clamp of 4.0. |
| `diffusiondet/loss.py` | Optional matcher and loss without CPU/GPU synchronisations (`DENSEA_FAST_LOSS`), with results identical to the reference path. |
| `diffusiondet/swintransformer.py` | `timm.layers` import (`timm.models.layers` was removed in recent timm versions). |
| `diffusiondet/util/box_ops.py` | IoU and GIoU with an epsilon; the assertion on degenerate boxes is removed. |
| `diffusiondet/util/model_ema.py` | Optional foreach EMA update (`DENSEA_FAST_EMA`). |

## Added files

- `diffusiondet/eval_history.py`, `diffusiondet/fast_opt.py`: evaluation history for resuming and the optional
  speed-ups.
- `configs/diffdet.densea.res50.yaml` (the configuration of the study) and the ResNet-101 and Swin variants.
- `tools/launch_train.py` (training and test launcher), `tools/benchmark_latency.py`,
  `tools/compute_model_complexity.py`.
- `tests/` (CPU only): fast-path equivalence, resume, sparse checkpointer.
- `Dockerfile`, `docker-compose.yml`, `requirements.txt`, `.gitignore`, `.gitattributes`, `.dockerignore`,
  `README.md`, `NOTICE`, `CHANGES.md`, `CITATION.cff`.

## Removed files

`configs/diffdet.coco.*.yaml` and `configs/diffdet.lvis.*.yaml`, `demo.py` and `diffusiondet/predictor.py` (the
visualization demo), `GETTING_STARTED.md`, `teaser.png`, `train_net.py` (replaced by `train_net_densea.py`). The
`LICENSE` of upstream is kept.
