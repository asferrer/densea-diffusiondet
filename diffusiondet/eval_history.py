"""Evaluation history, used to resume with the same best value and the same patience.

`metrics.json` cannot serve as the history on its own: the evaluation function reopens it in
'w' mode at every evaluation (leaving only the last indented block) and detectron2's JSONWriter
only appends lines every 20 iterations. Therefore each evaluation is also appended to
`eval_history.jsonl`; on resume both are read (JSON lines of metrics.json that contain the metric
and `iteration`) and deduplicated by iteration.
"""
import json
import math
import os

HISTORY_FILE = "eval_history.jsonl"


def append_eval(output_dir, iteration, metric, value):
    with open(os.path.join(output_dir, HISTORY_FILE), "a") as f:
        f.write(json.dumps({"iteration": int(iteration), metric: float(value)}) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _json_lines(path):
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except ValueError:
                continue  # indented block of metrics.json, or a line truncated by a crash
            if isinstance(rec, dict):
                yield rec


def load_evals(output_dir, metric, before_iter):
    """Sorted [(iteration, value)] with the evaluations of iteration < before_iter."""
    by_iter = {}
    for name in ("metrics.json", HISTORY_FILE):  # our own history takes precedence
        for rec in _json_lines(os.path.join(output_dir, name)):
            it, val = rec.get("iteration"), rec.get(metric)
            if isinstance(it, int) and isinstance(val, (int, float)) and it < before_iter:
                by_iter[it] = float(val)
    return sorted(by_iter.items())


def replay_best(evals):
    """Same logic as detectron2 BestCheckpointer (mode='max'): (best_metric, best_iter)."""
    best, best_iter = None, None
    for it, val in evals:
        if math.isnan(val) or math.isinf(val):
            continue
        if best is None or val > best:
            best, best_iter = val, it
    return best, best_iter


def replay_patience(evals, min_delta):
    """Same logic as EarlyStoppingHook: (best, bad)."""
    best, bad = None, 0
    for _, val in evals:
        if best is None or val > best + min_delta:
            best, bad = val, 0
        else:
            bad += 1
    return best, bad
