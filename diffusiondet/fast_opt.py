"""Speed optimizations that do NOT change training (all disabled by default).

Environment variables (set to "1" to enable):
  DENSEA_FAST_OPT   groups parameters by (lr, weight_decay) and clips gradients with torch._foreach_*
  DENSEA_FAST_LOSS  removes CPU/GPU synchronizations in the loss and in the matching (same result)
  DENSEA_FAST_EMA   updates the EMA with torch._foreach_*
  DENSEA_NO_SCALER  does not use GradScaler (only with bf16 AMP)
"""
import os
from collections import defaultdict

import torch


def flag(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------- parameter grouping
def trainable_params(model):
    """Trainable parameters in the order (and without duplicates) of the original optimizer."""
    memo, out = set(), []
    for key, p in model.named_parameters(recurse=True):
        if p.requires_grad and p not in memo:
            memo.add(p)
            out.append((key, p))
    return out


def group_params(model, base_lr, weight_decay, backbone_multiplier):
    """One group per (lr, weight_decay) instead of one per tensor. Returns (groups, original_order)."""
    groups = {}
    legacy = trainable_params(model)
    for key, p in legacy:
        lr = base_lr * backbone_multiplier if "backbone" in key else base_lr
        groups.setdefault((lr, weight_decay), []).append(p)
    out = [{"params": ps, "lr": lr, "weight_decay": wd} for (lr, wd), ps in groups.items()]
    return out, [p for _, p in legacy]


# ---------------------------------------------------------------- gradient clipping
@torch.no_grad()
def foreach_clip_(params, clip_type, clip_value, norm_type=2.0):
    """Same as applying torch.nn.utils.clip_grad_norm_ / clip_grad_value_ to EACH parameter separately
    (what detectron2 does with CLIP_TYPE norm/value), but with foreach kernels and a SINGLE synchronization.
    """
    by_key = defaultdict(list)
    for p in params:
        if p.grad is not None:
            by_key[(p.grad.device, p.grad.dtype)].append(p.grad)
    for grads in by_key.values():
        if clip_type == "value":
            torch._foreach_clamp_min_(grads, -clip_value)
            torch._foreach_clamp_max_(grads, clip_value)
            continue
        norms = torch.stack(torch._foreach_norm(grads, norm_type))
        # norm of a 1-element vector per parameter, as in clip_grad_norm_ with a single tensor
        total = torch.linalg.vector_norm(norms.unsqueeze(1), norm_type, dim=1)
        coef = torch.clamp(clip_value / (total + 1e-6), max=1.0)
        torch._foreach_mul_(grads, coef.tolist())  # the float32 -> float conversion is exact


def make_fast_optimizer_class(optim_cls, clip_cfg):
    """Optimizer subclass with per-parameter foreach clipping that can load older checkpoints."""
    enabled = bool(clip_cfg.ENABLED) and clip_cfg.CLIP_TYPE in ("norm", "value")
    clip_type, clip_value, norm_type = clip_cfg.CLIP_TYPE, clip_cfg.CLIP_VALUE, clip_cfg.NORM_TYPE

    class FastOptimizer(optim_cls):
        legacy_order = None  # parameters in the order of the old optimizer (one group per tensor)

        def step(self, closure=None):
            if enabled:
                foreach_clip_([p for g in self.param_groups for p in g["params"]],
                              clip_type, clip_value, norm_type)
            return super().step(closure)

        def load_state_dict(self, state_dict):
            super().load_state_dict(regroup_state_dict(state_dict, self))

    return FastOptimizer


def regroup_state_dict(sd, optimizer):
    """Convert a state_dict with one group per tensor (older checkpoints) to the grouped layout."""
    saved = sd["param_groups"]
    legacy = optimizer.legacy_order
    if len(saved) == len(optimizer.param_groups) or legacy is None:
        return sd
    n_saved = sum(len(g["params"]) for g in saved)
    if n_saved != len(legacy):
        raise ValueError(f"Incompatible state dict: {n_saved} saved parameters, {len(legacy)} in the model")
    old_ids = [i for g in saved for i in g["params"]]  # saved ids, in the order of the old groups
    new_id, k = {}, 0
    for g in optimizer.param_groups:
        for p in g["params"]:
            new_id[id(p)] = k
            k += 1
    old_to_new = {old: new_id[id(p)] for old, p in zip(old_ids, legacy)}
    state = {old_to_new[i]: v for i, v in sd["state"].items()}
    groups, first_saved = [], {}
    for g in saved:
        for i in g["params"]:
            first_saved.setdefault(old_to_new[i], g)
    for g in optimizer.param_groups:
        ids = [new_id[id(p)] for p in g["params"]]
        ng = {k_: v for k_, v in first_saved[ids[0]].items() if k_ != "params"}
        ng["params"] = ids
        groups.append(ng)
    return {"state": state, "param_groups": groups}


# ---------------------------------------------------------------- EMA
@torch.no_grad()
def fast_ema_update(updater, model):
    """Same computation as EMAUpdater.update (ema = ema*d + val*(1-d), rounding each operation)."""
    updater.updates += 1
    d = updater.decay(updater.updates) if updater.yolox else updater.decay
    state = updater.state
    cache = getattr(updater, "_fast_cache", None)
    cur = list(state.state.values())
    if (cache is None or cache[0] != id(model) or len(cache[1]) != len(cur)
            or any(a is not b for a, b in zip(cache[1], cur))):
        names = [n for n, _ in state.get_model_state_iterator(model)]
        vals = [v for _, v in state.get_model_state_iterator(model)]
        emas = [state.state[n] for n in names]
        fl = [i for i, e in enumerate(emas) if e.is_floating_point()]
        rest = [i for i in range(len(emas)) if i not in set(fl)]
        cache = (id(model), cur, [emas[i] for i in fl], [vals[i] for i in fl],
                 [(emas[i], vals[i]) for i in rest])
        updater._fast_cache = cache
    _, _, emas, vals, rest = cache
    if updater.device:
        vals = [v.to(updater.device) for v in vals]
    tmp = torch._foreach_mul(vals, 1.0 - d)
    torch._foreach_mul_(emas, d)
    torch._foreach_add_(emas, tmp)
    for e, v in rest:  # integer buffers (num_batches_tracked): original path
        v = v.to(updater.device) if updater.device else v
        e.copy_(e * d + v * (1.0 - d))
