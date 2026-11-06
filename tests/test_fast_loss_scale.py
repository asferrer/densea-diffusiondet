"""CPU test: DENSEA_FAST_LOSS does not change the loss or the matching at a realistic scale.

Complements tests/test_fast_opt.py: 500 queries, batch 8, up to 24 GT per image, fp32, bf16 and bf16 autocast.
Does not need detectron2 or fvcore (if missing, an equivalent focal loss is substituted in the test itself).
"""
import copy
import importlib
import os
import sys
import types
from types import SimpleNamespace as NS

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_loss():
    try:
        from diffusiondet import loss
        return loss
    except Exception:
        pass

    def focal(x, t, alpha=0.25, gamma=2, reduction="none"):
        p = torch.sigmoid(x)
        ce = torch.nn.functional.binary_cross_entropy_with_logits(x, t, reduction="none")
        pt = p * t + (1 - p) * (1 - t)
        return (alpha * t + (1 - alpha) * (1 - t)) * ce * (1 - pt) ** gamma

    fv, fvn = types.ModuleType("fvcore"), types.ModuleType("fvcore.nn")
    fvn.sigmoid_focal_loss_jit = focal
    sys.modules.update({"fvcore": fv, "fvcore.nn": fvn})
    pkg = types.ModuleType("diffusiondet")
    pkg.__path__ = [os.path.join(ROOT, "diffusiondet")]
    sys.modules["diffusiondet"] = pkg
    return importlib.import_module("diffusiondet.loss")


loss = _load_loss()
CFG = NS(MODEL=NS(DiffusionDet=NS(USE_FED_LOSS=False, OTA_K=5, ALPHA=0.25, GAMMA=2.0)))


def criterion(fast):
    os.environ["DENSEA_FAST_LOSS"] = "1" if fast else "0"
    try:
        m = loss.HungarianMatcherDynamicK(CFG, 2, 5, 2, use_focal=True)
        wd = {"loss_ce": 2.0, "loss_bbox": 5.0, "loss_giou": 2.0}
        return loss.SetCriterionDynamicK(CFG, 4, m, wd, 0.1, ["labels", "boxes"], True)
    finally:
        os.environ.pop("DENSEA_FAST_LOSS", None)


def case(seed, dt, bs=8, Q=500, C=4):
    g = torch.Generator().manual_seed(seed)
    whwh = torch.tensor([640.0, 480.0, 640.0, 480.0])
    targets, boxes = [], []
    for _ in range(bs):
        n = int(torch.randint(1, 25, (1,), generator=g))
        c = torch.rand(n, 2, generator=g) * 0.8 + 0.1
        s = torch.rand(n, 2, generator=g) * 0.2 + 0.02
        xy = torch.cat([c - s / 2, c + s / 2], 1) * whwh
        targets.append({"labels": torch.randint(0, C, (n,), generator=g), "boxes": torch.cat([c, s], 1),
                        "boxes_xyxy": xy, "image_size_xyxy": whwh, "image_size_xyxy_tgt": whwh})
        boxes.append(xy[torch.randint(0, n, (Q,), generator=g)] + torch.randn(Q, 4, generator=g) * 40)
    out = {"pred_logits": torch.randn(bs, Q, C, generator=g) * 2, "pred_boxes": torch.stack(boxes).to(dt)}
    out["aux_outputs"] = [{k: v + 0.05 * (i + 1) * torch.sin(v) for k, v in out.items() if k != "aux_outputs"}
                          for i in range(3)]
    return out, targets


@pytest.mark.parametrize("dt,autocast", [(torch.float32, False), (torch.bfloat16, False), (torch.bfloat16, True)])
@pytest.mark.parametrize("seed", range(6))
def test_fast_loss_identical_at_scale(seed, dt, autocast):
    out, targets = case(seed, dt)
    res = []
    for fast in (False, True):
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
            res.append(criterion(fast)(copy.deepcopy(out), targets))
    assert res[0].keys() == res[1].keys()
    for k in res[0]:
        assert torch.equal(res[0][k], res[1][k]), k


def test_fast_loss_does_not_consume_torch_rng():
    """Neither the criterion nor the matcher touches the random generator, with or without the variable."""
    out, targets = case(0, torch.float32)
    states = []
    for fast in (False, True):
        torch.manual_seed(7)
        criterion(fast)(copy.deepcopy(out), targets)
        states.append(torch.get_rng_state().clone())
    assert torch.equal(states[0], states[1])
    torch.manual_seed(7)
    assert torch.equal(torch.get_rng_state(), states[0])  # and it did not advance relative to the initial state either
