"""SparsePeriodicCheckpointer: saves every `period` iterations, keeps the multiples of
`keep_every` and only the latest of the others (for resuming). CPU only.

  CUDA_VISIBLE_DEVICES= python -m pytest -q tests/test_sparse_checkpointer.py
"""
import os
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import train_net_densea as T  # noqa: E402

IPE = 7                      # iterations per epoch
PERIOD, KEEP = 10 * IPE, 20 * IPE


class FakeCkpt:
    def __init__(self, d):
        self.save_dir = d
        self.path_manager = types.SimpleNamespace(exists=os.path.exists, rm=os.remove)

    def save(self, name, **kw):
        Path(self.save_dir, f"{name}.pth").write_bytes(b"x")


def run(out, start, stop, max_iter):
    hook = T.SparsePeriodicCheckpointer(FakeCkpt(out), PERIOD, KEEP)
    hook.max_iter = max_iter
    for it in range(start, stop):
        hook.step(it)
    return sorted(p.name for p in Path(out).glob("*.pth"))


def test_keeps_multiples_and_the_last():
    with tempfile.TemporaryDirectory() as out:
        files = run(out, 0, 75 * IPE, max_iter=200 * IPE)
        # epochs 10, 20, ..., 70: 20, 40, 60 and the last one (70) are kept
        want = [f"model_{e * IPE - 1:07d}.pth" for e in (20, 40, 60, 70)]
        assert files == want, files


def test_model_final_is_not_deleted():
    with tempfile.TemporaryDirectory() as out:
        files = run(out, 0, 30 * IPE, max_iter=30 * IPE)
        assert "model_final.pth" in files and f"model_{20 * IPE - 1:07d}.pth" in files, files


def test_without_keep_every_original_hook_is_kept():
    cfg = T.get_cfg()
    T.add_densea_configs(cfg)
    assert cfg.DENSEA.CHECKPOINT_KEEP_EVERY == 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
