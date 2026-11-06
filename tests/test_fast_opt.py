"""CPU tests of the optimizations DENSEA_FAST_OPT / FAST_LOSS / FAST_EMA / NO_SCALER.

  CUDA_VISIBLE_DEVICES= python -m pytest -q tests/test_fast_opt.py

They compare the fast path with the original one under a fixed seed and require IDENTICAL results (torch.equal).
"""
import copy
import os
import sys
import types

import pytest
import torch
import torch.nn.functional as F
from detectron2.config import CfgNode as CN

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diffusiondet.fast_opt import foreach_clip_, regroup_state_dict  # noqa: E402
from diffusiondet.loss import HungarianMatcherDynamicK, SetCriterionDynamicK  # noqa: E402
from diffusiondet.util.model_ema import EMAState, EMAUpdater  # noqa: E402

torch.set_num_threads(1)


# ------------------------------------------------------------------ (a) optimizer
class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.BatchNorm2d(4))
        self.head = torch.nn.Sequential(torch.nn.Linear(4, 6), torch.nn.LayerNorm(6), torch.nn.Linear(6, 2))
        self.frozen = torch.nn.Linear(2, 2)
        self.frozen.requires_grad_(False)
        self.head_alias = self.head[0]  # shared parameters: must not be duplicated
        self.unused = torch.nn.Linear(2, 2)  # no gradient (grad None)


def solver_cfg(opt="ADAMW", clip_type="norm", clip_value=1.0, enabled=True):
    c = CN()
    c.SOLVER = CN()
    c.SOLVER.OPTIMIZER = opt
    c.SOLVER.BASE_LR = 5e-5
    c.SOLVER.BACKBONE_MULTIPLIER = 0.1
    c.SOLVER.WEIGHT_DECAY = 0.05
    c.SOLVER.MOMENTUM = 0.9
    c.SOLVER.CLIP_GRADIENTS = CN()
    c.SOLVER.CLIP_GRADIENTS.ENABLED = enabled
    c.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = clip_type
    c.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = clip_value
    c.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0
    return c


def build(cfg, fast, foreach=None):
    os.environ["DENSEA_FAST_OPT"] = "1" if fast else "0"
    try:
        import train_net_densea as T
        model = Tiny()
        torch.manual_seed(0)
        for p in model.parameters():
            p.data = torch.randn_like(p)
        opt = T.Trainer.build_optimizer(cfg, model)
    finally:
        os.environ.pop("DENSEA_FAST_OPT", None)
    if foreach is not None:
        for g in opt.param_groups:
            g["foreach"] = foreach
    return model, opt


def set_grads(model, step, scale, all_grads=False):
    g = torch.Generator().manual_seed(100 + step)
    for i, p in enumerate(model.parameters()):
        unused = p is model.unused.weight or p is model.unused.bias
        if p.requires_grad and (all_grads or not unused):
            p.grad = torch.randn(p.shape, generator=g) * scale * (10.0 if i == 2 else 1.0)


def train(model, opt, steps, scale, start=0, all_grads=False):
    for s in range(start, start + steps):
        opt.zero_grad()
        set_grads(model, s, scale, all_grads)
        # same as run_step: global model clipping, then opt.step() (which clips per parameter)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2.0)
        opt.step()


def hyper(model, opt):
    return {id(p): (g["lr"], g["weight_decay"]) for g in opt.param_groups for p in g["params"]}


@pytest.mark.parametrize("foreach", [None, False, True])
@pytest.mark.parametrize("clip_type,clip_value", [("norm", 1.0), ("norm", 1e-3), ("value", 0.01)])
@pytest.mark.parametrize("opt_name", ["ADAMW", "SGD"])
def test_optimizer_groups_and_steps(opt_name, clip_type, clip_value, foreach):
    cfg = solver_cfg(opt_name, clip_type, clip_value)
    m0, o0 = build(cfg, fast=False, foreach=foreach)
    m1, o1 = build(cfg, fast=True, foreach=foreach)
    assert len(o0.param_groups) > len(o1.param_groups) == 2
    # same hyperparameters per parameter (compared by name)
    h0, h1 = hyper(m0, o0), hyper(m1, o1)
    for (n0, p0), (n1, p1) in zip(m0.named_parameters(), m1.named_parameters()):
        if p0.requires_grad:
            assert h0[id(p0)] == h1[id(p1)], n0
    assert sum(len(g["params"]) for g in o1.param_groups) == sum(len(g["params"]) for g in o0.param_groups)
    assert o1.defaults.get("fused") in (None, False) if opt_name == "ADAMW" else True
    # (detectron2 with CLIP_TYPE=value fails with grad None: there every parameter has a gradient)
    allg = clip_type == "value"
    train(m0, o0, 6, scale=3.0, all_grads=allg)
    train(m1, o1, 6, scale=3.0, all_grads=allg)
    for (n, p0), (_, p1) in zip(m0.named_parameters(), m1.named_parameters()):
        assert torch.equal(p0, p1), (n, (p0 - p1).abs().max().item())


def test_clip_actually_changes_gradients():
    """Per-parameter clipping is not a no-op after global clipping (~1e-6 relative effect)."""
    p = torch.nn.Parameter(torch.ones(4))
    p.grad = torch.full((4,), 5.0)
    ref = copy.deepcopy(p.grad)
    torch.nn.utils.clip_grad_norm_([p], 1.0, 2.0)
    p2 = torch.nn.Parameter(torch.ones(4))
    p2.grad = torch.full((4,), 5.0)
    foreach_clip_([p2], "norm", 1.0, 2.0)
    assert torch.equal(p.grad, p2.grad) and not torch.equal(p.grad, ref)


def test_foreach_clip_equals_per_param_random():
    g = torch.Generator().manual_seed(1)
    ps0 = [torch.nn.Parameter(torch.zeros(n)) for n in (3, 17, 1, 64, 5)]
    for p in ps0:
        p.grad = torch.randn(p.shape, generator=g) * 10
    ps1 = [torch.nn.Parameter(torch.zeros_like(p)) for p in ps0]
    for a, b in zip(ps0, ps1):
        b.grad = a.grad.clone()
    for p in ps0:
        torch.nn.utils.clip_grad_norm_(p, 1.0, 2.0)
    foreach_clip_(ps1, "norm", 1.0, 2.0)
    assert all(torch.equal(a.grad, b.grad) for a, b in zip(ps0, ps1))


def test_resume_from_legacy_checkpoint():
    """A state_dict with one group per tensor (older checkpoints) loads into the grouped optimizer."""
    cfg = solver_cfg()
    m0, o0 = build(cfg, fast=False)
    train(m0, o0, 3, scale=3.0)
    sd = copy.deepcopy(o0.state_dict())
    m1, o1 = build(cfg, fast=True)
    m1.load_state_dict(m0.state_dict())
    o1.load_state_dict(sd)
    assert len(o1.param_groups) == 2
    train(m0, o0, 3, scale=3.0, start=3)
    train(m1, o1, 3, scale=3.0, start=3)
    for (n, p0), (_, p1) in zip(m0.named_parameters(), m1.named_parameters()):
        assert torch.equal(p0, p1), n
    # and the grouped state_dict loads back as is
    assert regroup_state_dict(o1.state_dict(), o1) is not None
    m2, o2 = build(cfg, fast=True)
    o2.load_state_dict(o1.state_dict())


# ------------------------------------------------------------------ (b) matcher and losses
def make_cfg(focal=True, fed=False):
    c = CN()
    c.MODEL = CN()
    c.MODEL.DiffusionDet = CN()
    d = c.MODEL.DiffusionDet
    d.USE_FED_LOSS, d.OTA_K, d.ALPHA, d.GAMMA, d.USE_FOCAL = fed, 5, 0.25, 2.0, focal
    return c


def make_matcher(fast, focal=True):
    os.environ["DENSEA_FAST_LOSS"] = "1" if fast else "0"
    try:
        return HungarianMatcherDynamicK(make_cfg(focal), cost_class=2, cost_bbox=5, cost_giou=2, use_focal=focal)
    finally:
        os.environ.pop("DENSEA_FAST_LOSS", None)


def rand_boxes(g, n, grid):
    xy = torch.randint(0, 20, (n, 2), generator=g).float()
    wh = torch.randint(1, 12, (n, 2), generator=g).float()
    b = torch.cat([xy, xy + wh], 1) * grid
    return b


def make_case(seed, ties):
    g = torch.Generator().manual_seed(seed)
    bs, Q, C = 3, 60, 4
    grid = 8.0 if ties else 1.0
    boxes = torch.stack([rand_boxes(g, Q, grid) + (0 if ties else torch.rand(Q, 4, generator=g)) for _ in range(bs)])
    if ties:  # duplicated queries -> tied costs
        boxes[:, 30:] = boxes[:, :30]
    logits = torch.randn(bs, Q, C, generator=g)
    if ties:
        logits = torch.round(logits)
        logits[:, 30:] = logits[:, :30]
    targets = []
    for b in range(bs):
        n = int(torch.randint(0, 14, (1,), generator=g)) if seed % 7 else 0  # some cases without any GT
        if seed % 5 == 0 and b == 1:
            n = 0
        gt = rand_boxes(g, n, grid)
        if n > 2 and (ties or seed % 3 == 0):
            gt[1] = gt[0]
            gt[2] = gt[0]  # several identical GT -> conflicts and the while branch
        wh = torch.tensor([[200.0, 160.0, 200.0, 160.0]])
        targets.append({"labels": torch.randint(0, C, (n,), generator=g), "boxes_xyxy": gt,
                        "boxes": gt / wh, "image_size_xyxy": wh[0], "image_size_xyxy_tgt": wh[0]})
    return {"pred_logits": logits, "pred_boxes": boxes}, targets


def same_indices(a, b):
    ia, ma = a
    ib, mb = b
    assert len(ia) == len(ib)
    for (qa, ga), (qb, gb) in zip(ia, ib):
        assert torch.equal(qa, qb) and torch.equal(ga, gb)
    for x, y in zip(ma, mb):
        assert torch.equal(x, y)


@pytest.mark.parametrize("seed", range(60))
def test_matcher_same_assignments(seed):
    outputs, targets = make_case(seed, ties=seed % 2 == 0)
    ref, fast = make_matcher(False), make_matcher(True)
    same_indices(ref(outputs, targets), fast(outputs, targets))


@pytest.mark.parametrize("seed", range(60))
def test_dynamic_k_direct_integer_costs(seed):
    """Integer costs and IoU (many ties), including GT that compete for the same queries."""
    g = torch.Generator().manual_seed(seed)
    Q, G = 40, int(torch.randint(1, 15, (1,), generator=g))
    cost = torch.randint(0, 4, (Q, G), generator=g).float()
    if seed % 3 == 0:
        cost[:, 1:] = cost[:, :1]  # all GT request the same queries
    ious = torch.randint(0, 3, (Q, G), generator=g).float() * 0.7
    ref, fast = make_matcher(False), make_matcher(True)
    (qa, ga), ma = ref.dynamic_k_matching_ref(cost.clone(), ious, G)
    (qb, gb), mb = fast.dynamic_k_matching_fast(cost.clone(), ious, G)
    assert torch.equal(qa, qb) and torch.equal(ga, gb) and torch.equal(ma, mb)


def make_criterion(fast):
    os.environ["DENSEA_FAST_LOSS"] = "1" if fast else "0"
    try:
        cfg = make_cfg()
        matcher = HungarianMatcherDynamicK(cfg, cost_class=2, cost_bbox=5, cost_giou=2, use_focal=True)
        wd = {"loss_ce": 2.0, "loss_bbox": 5.0, "loss_giou": 2.0}
        return SetCriterionDynamicK(cfg, 4, matcher, wd, 0.1, ["labels", "boxes"], True)
    finally:
        os.environ.pop("DENSEA_FAST_LOSS", None)


@pytest.mark.parametrize("seed", range(30))
def test_criterion_same_losses(seed):
    outputs, targets = make_case(seed, ties=seed % 2 == 0)
    outputs["aux_outputs"] = [{k: v + 0.01 * (i + 1) * torch.sin(v) for k, v in outputs.items()}
                              for i in range(3)]
    if seed % 4 == 1:
        outputs["pred_logits"][0, 3, 1] = float("inf")  # clamp-to-[-15, 15] branch
    ref, fast = make_criterion(False), make_criterion(True)
    out = copy.deepcopy(outputs)
    la, lb = ref(outputs, targets), fast(out, targets)
    assert la.keys() == lb.keys()
    for k in la:
        assert torch.equal(la[k], lb[k]), k


def test_cases_include_empty_targets():
    empties = sum(len(x["labels"]) == 0 for seed in range(60) for x in make_case(seed, seed % 2 == 0)[1])
    assert empties > 10


# ------------------------------------------------------------------ (c) EMA
def make_ema_model():
    m = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.BatchNorm2d(4), torch.nn.Flatten(),
                            torch.nn.Linear(4, 3))
    return m


@pytest.mark.parametrize("decay", [0.999, 0.9998])
def test_ema_after_n_steps(decay):
    def run(fast):
        os.environ["DENSEA_FAST_EMA"] = "1" if fast else "0"
        try:
            torch.manual_seed(0)
            model = make_ema_model()
            upd = EMAUpdater(EMAState(), decay=decay, device="cpu")
        finally:
            os.environ.pop("DENSEA_FAST_EMA", None)
        upd.init_state(model)
        g = torch.Generator().manual_seed(5)
        for i in range(40):
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(torch.randn(p.shape, generator=g) * 0.1)
                model(torch.randn(2, 3, 3, 3, generator=g))  # updates the BN buffers
            upd.update(model)
        return upd.state.state
    a, b = run(False), run(True)
    assert a.keys() == b.keys()
    for k in a:
        assert a[k].dtype == b[k].dtype and torch.equal(a[k], b[k]), k


def test_ema_cache_invalidated_after_state_reload():
    os.environ["DENSEA_FAST_EMA"] = "1"
    try:
        model = make_ema_model()
        upd = EMAUpdater(EMAState(), decay=0.99, device="cpu")
    finally:
        os.environ.pop("DENSEA_FAST_EMA", None)
    upd.init_state(model)
    upd.update(model)
    new = {k: torch.zeros_like(v) for k, v in upd.state.state.items()}
    upd.state.load_state_dict(new)  # as when resuming: the EMA tensors are different ones
    upd.update(model)
    k = next(iter(new))
    assert torch.equal(upd.state.state[k], dict(model.named_parameters())[k] * (1 - 0.99)) or k in dict(model.named_buffers())


# ------------------------------------------------------------------ (d) without GradScaler
def amp_train(use_scaler, steps=25):
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    scaler = torch.amp.GradScaler("cpu", init_scale=2.0 ** 16) if use_scaler else None
    g = torch.Generator().manual_seed(3)
    for _ in range(steps):
        x, y = torch.randn(16, 8, generator=g), torch.randn(16, 4, generator=g)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = F.mse_loss(model(x).float(), y)
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        opt.zero_grad()
    return [p.detach().clone() for p in model.parameters()]


def test_no_scaler_equivalent_with_bf16():
    a, b = amp_train(True), amp_train(False)
    for x, y in zip(a, b):
        assert torch.equal(x, y)


def test_step_without_scaler_skips_nonfinite():
    import train_net_densea as T
    cfg = solver_cfg()
    model = torch.nn.Linear(3, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    stub = types.SimpleNamespace(cfg=cfg, model=model, optimizer=opt, iter=0,
                                 _check_loss=lambda *a: T.Trainer._check_loss(stub, *a))
    w0 = model.weight.detach().clone()
    for p in model.parameters():
        p.grad = torch.full_like(p, float("nan"))
    T.Trainer._step_without_scaler(stub, torch.tensor(1.0), {})
    assert torch.equal(model.weight, w0)  # skipped step
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    T.Trainer._step_without_scaler(stub, torch.tensor(1.0), {})
    assert not torch.equal(model.weight, w0)
    with pytest.raises(FloatingPointError):
        T.Trainer._step_without_scaler(stub, torch.tensor(float("nan")), {})


def test_direct_cases_enter_while_loop():
    """Ensure that the integer cases above go through the `while` branch (GT without a query after tie-breaking)."""
    entered = 0
    for seed in range(60):
        g = torch.Generator().manual_seed(seed)
        Q, G = 40, int(torch.randint(1, 15, (1,), generator=g))
        cost = torch.randint(0, 4, (Q, G), generator=g).float()
        if seed % 3 == 0:
            cost[:, 1:] = cost[:, :1]
        ious = torch.randint(0, 3, (Q, G), generator=g).float() * 0.7
        ks = torch.clamp(torch.topk(ious, 5, dim=0)[0].sum(0).int(), min=1).tolist()
        mm = torch.zeros_like(cost)
        for i, k in enumerate(ks):
            mm[torch.topk(cost[:, i], k, largest=False)[1], i] = 1
        multi = mm.sum(1) > 1
        mm[multi] *= 0
        mm[multi, cost[multi].min(1)[1]] = 1
        entered += int((mm.sum(0) == 0).any())
    assert entered > 20


@pytest.mark.parametrize("clip_enabled", [True, False])
def test_run_step_fast_loss_same_as_reference(clip_enabled):
    """Full run_step (no AMP, CPU): same weights and same metrics with and without DENSEA_FAST_LOSS."""
    import train_net_densea as T
    from detectron2.utils.events import EventStorage

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l = torch.nn.Linear(4, 2)

        def forward(self, data):
            out = self.l(data)
            return {"loss_a": out.pow(2).mean(), "loss_b": out.abs().mean()}

    def run(fast):
        torch.manual_seed(0)
        model = M()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
        stub = types.SimpleNamespace(
            model=model, optimizer=opt, cfg=solver_cfg(enabled=clip_enabled), iter=0, accumulation_steps=1,
            use_amp=False, amp_dtype=None, grad_scaler=None, _fast_loss=fast,
            _data_loader_iter=iter([torch.randn(8, 4) for _ in range(3)]), data_loader=None)
        stub._check_loss = lambda *a: T.Trainer._check_loss(stub, *a)
        seen = []
        with EventStorage(0) as storage:
            for i in range(3):
                stub.iter = i
                T.Trainer.run_step(stub)
                seen.append({k: v[0] for k, v in storage.latest().items() if k.startswith(("loss", "total"))})
        return model.l.weight.detach().clone(), seen

    (w0, s0), (w1, s1) = run(False), run(True)
    assert torch.equal(w0, w1) and s0 == s1
