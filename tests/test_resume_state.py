"""DiffusionDet resume: BestCheckpointer and EarlyStoppingHook are seeded with the
previous evaluations (eval_history.jsonl / metrics.json). CPU only.

  CUDA_VISIBLE_DEVICES= python -m pytest -q tests/test_resume_state.py
"""
import json
import os
import sys
import types
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import train_net_densea as T  # noqa: E402

METRIC, PERIOD, PATIENCE, MIN_DELTA = "bbox/AP", 128, 20, 0.1


def ap(k):
    """AP (0-100 scale) of evaluation k (0-based): the best one is the 10th (k=9)."""
    return 30.0 + k if k < 10 else 39.0 - 0.1 * (k - 9)


def eval_iter(k):
    return (k + 1) * PERIOD - 1


def write_history(out, n):  # same format as diffusiondet.eval_history.append_eval
    with open(os.path.join(out, "eval_history.jsonl"), "a") as f:
        for k in range(n):
            f.write(json.dumps({"iteration": eval_iter(k), METRIC: ap(k)}) + chr(10))


class FakeStorage:
    def __init__(self):
        self.vals = {}

    def latest(self):
        return self.vals


class FakeCkpt:
    def __init__(self):
        self.saved = []

    def save(self, name, **kw):
        self.saved.append(kw["iteration"])


def make(out, start_iter):
    ck = FakeCkpt()
    try:
        best = T.SeededBestCheckpointer(PERIOD, ck, METRIC, mode="max", file_prefix="model_best",
                                        output_dir=str(out))
        stop = T.EarlyStoppingHook(PERIOD, METRIC, PATIENCE, MIN_DELTA, output_dir=str(out))
    except (AttributeError, TypeError):  # older code: hooks without seeding
        best = T.hooks.BestCheckpointer(PERIOD, ck, METRIC, mode="max", file_prefix="model_best")
        stop = T.EarlyStoppingHook(PERIOD, METRIC, PATIENCE, MIN_DELTA)
    tr = types.SimpleNamespace(start_iter=start_iter, iter=start_iter, max_iter=10**9,
                               storage=FakeStorage())
    for h in (best, stop):
        h.trainer = tr
        h.before_train()
    return best, stop, ck, tr


def run_evals(best, stop, tr, ks):
    """Simulate EvalHook + hooks for each evaluation k; return the k at which EarlyStop triggers."""
    for k in ks:
        tr.iter = eval_iter(k)
        tr.storage.vals[METRIC] = (ap(k), tr.iter)
        best.after_step()
        try:
            stop.after_step()
        except T.EarlyStop:
            return k
    return None


def test_resume_keeps_best_and_counter(tmp_path):
    write_history(tmp_path, 25)
    best, stop, ck, tr = make(tmp_path, eval_iter(24) + 1)
    assert best.best_metric == ap(9) and best.best_iter == eval_iter(9)
    assert stop.best == ap(9) and stop.bad == 15


def test_model_best_not_overwritten_by_worse_eval(tmp_path):
    write_history(tmp_path, 25)
    best, stop, ck, tr = make(tmp_path, eval_iter(24) + 1)
    run_evals(best, stop, tr, [25])
    assert ck.saved == []


def test_stops_after_5_evaluations(tmp_path):
    write_history(tmp_path, 25)
    best, stop, ck, tr = make(tmp_path, eval_iter(24) + 1)
    assert run_evals(best, stop, tr, range(25, 60)) == 29


def test_equivalent_to_uninterrupted_run(tmp_path):
    out0 = tmp_path / "a"; out0.mkdir()
    best0, stop0, ck0, tr0 = make(out0, 0)
    assert run_evals(best0, stop0, tr0, range(25)) is None
    out1 = tmp_path / "b"; out1.mkdir()
    write_history(out1, 25)
    best1, stop1, ck1, tr1 = make(out1, eval_iter(24) + 1)
    assert (best1.best_metric, best1.best_iter, stop1.best, stop1.bad) == \
           (best0.best_metric, best0.best_iter, stop0.best, stop0.bad)


def test_only_evaluations_before_start_iter_count(tmp_path):
    write_history(tmp_path, 25)
    best, stop, ck, tr = make(tmp_path, eval_iter(19) + 1)  # resume after eval 19
    assert stop.bad == 10 and best.best_iter == eval_iter(9)


def test_fallback_metrics_json_indented_block_and_duplicates(tmp_path):
    # real metrics.json: indented block (reopened with 'w') + JSONWriter JSON lines,
    # possibly duplicated; no eval_history.jsonl
    lines = [json.dumps({METRIC: ap(k), "iteration": eval_iter(k), "loss": 1.0}) for k in range(25)]
    lines += lines[-3:]
    block = '{' + chr(10) + '    "bbox": {"AP": 1.0}' + chr(10) + '}'
    (tmp_path / "metrics.json").write_text(block + chr(10).join(lines) + chr(10))
    best, stop, ck, tr = make(tmp_path, eval_iter(24) + 1)
    assert stop.bad == 15 and best.best_iter == eval_iter(9)


def test_no_history_starts_from_zero(tmp_path):
    best, stop, ck, tr = make(tmp_path, 500)
    assert best.best_metric is None and stop.best is None and stop.bad == 0


def test_new_run_does_not_seed(tmp_path):
    write_history(tmp_path, 5)
    best, stop, ck, tr = make(tmp_path, 0)
    assert best.best_metric is None and stop.bad == 0


if __name__ == '__main__':
    import tempfile
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            with tempfile.TemporaryDirectory() as d:
                try:
                    fn(Path(d))
                    print('OK', name)
                except Exception as exc:  # noqa: BLE001
                    print('FAIL', name, type(exc).__name__, str(exc)[:90])
