"""Train and/or evaluate one DiffusionDet experiment with the common protocol of the study.

    python tools/launch_train.py --train-json annotations/train.json --val-json annotations/val.json \
        --test-json annotations/test.json --image-root /data/densea_aggregated --output-dir runs/real_only

Protocol: at most 200 epochs (MAX_ITER = 200 x ceil(N / batch), batch 8), evaluation on the validation split once
per epoch, ``model_best.pth`` = best bbox/AP of the EMA weights on validation, and early stopping after 20
evaluations without an improvement of 0.1 AP points (= 0.001 on a 0-1 scale). Periodic checkpoints are written every
10 epochs and those of every 20 epochs are kept, plus the last one. After training, ``model_best.pth`` is evaluated
once on the test split; the detections are written to
``<output-dir>/eval_test/inference/densea_test/coco_instances_results.json``.
Training resumes from the last checkpoint of ``--output-dir`` if there is one.
"""
import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "diffdet.densea.res50.yaml"


def schedule(train_json, epochs, batch, eval_every):
    """Iteration-based solver settings equivalent to ``epochs`` epochs over the training split."""
    with open(train_json, encoding="utf8") as f:
        n = len(json.load(f)["images"])
    ipe = math.ceil(n / batch)
    return {"SOLVER.MAX_ITER": epochs * ipe, "TEST.EVAL_PERIOD": eval_every * ipe,
            "SOLVER.CHECKPOINT_PERIOD": 10 * ipe, "DENSEA.CHECKPOINT_KEEP_EVERY": 20 * ipe,
            "SOLVER.IMS_PER_BATCH": batch, "_images": n}


def environment(args):
    """Dataset locations and speed options are passed to train_net_densea.py through the environment."""
    return {**os.environ,
            "DENSEA_TRAIN_JSON": str(args.train_json), "DENSEA_VALID_JSON": str(args.val_json),
            "DENSEA_TEST_JSON": str(args.test_json or args.val_json), "DENSEA_IMAGE_ROOT": str(args.image_root),
            # Predictions identical to 0 workers; fewer seconds per evaluation.
            "DENSEA_TEST_WORKERS": os.environ.get("DENSEA_TEST_WORKERS", "4"),
            # Same training trajectory as the reference path (see tests/test_fast_opt.py), 1.14x faster.
            "DENSEA_FAST_OPT": os.environ.get("DENSEA_FAST_OPT", "1")}


def opts_list(d):
    return [str(x) for k, v in d.items() if not k.startswith("_") for x in (k, v)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-json", required=True)
    ap.add_argument("--val-json", required=True)
    ap.add_argument("--test-json", default=None, help="needed for the final evaluation")
    ap.add_argument("--image-root", required=True, help="directory the file_name entries are relative to")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=1, help="evaluation period in epochs")
    ap.add_argument("--patience", type=int, default=20, help="evaluations without improvement")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--no-test", action="store_true", help="do not evaluate on the test split at the end")
    args = ap.parse_args()
    if not args.no_test and not args.test_json:
        ap.error("--test-json is required unless --no-test is given")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    sch = schedule(args.train_json, args.epochs, args.batch, args.eval_every)
    sch["DENSEA.EARLY_STOP_PATIENCE"] = args.patience
    sch["SEED"] = args.seed
    sch["OUTPUT_DIR"] = str(out)
    with open(out / "schedule.json", "w", encoding="utf8") as f:
        json.dump(sch, f, indent=1)

    base = [sys.executable, str(ROOT / "train_net_densea.py"), "--num-gpus", "1", "--config-file", args.config]
    env = environment(args)
    if not args.eval_only:
        cmd = base + ["--resume"] + opts_list(sch)
        print(" ".join(cmd), flush=True)
        if subprocess.call(cmd, cwd=ROOT, env=env) != 0:
            sys.exit("training failed")
    if args.no_test:
        return
    cmd = base + ["--eval-only", "MODEL.WEIGHTS", str(out / "model_best.pth"), "DATASETS.TEST", '("densea_test",)',
                  "OUTPUT_DIR", str(out / "eval_test")]
    print(" ".join(cmd), flush=True)
    sys.exit(subprocess.call(cmd, cwd=ROOT, env=env))


if __name__ == "__main__":
    main()
