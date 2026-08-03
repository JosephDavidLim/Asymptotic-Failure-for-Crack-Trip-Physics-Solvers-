"""
Runs the full sweep: for each S in the config's S_values, trains n_seeds
independent enriched elastic pihnn models (fresh random seed each), and
writes one raw output directory per (S, seed) run under results/raw/.

Safe to interrupt and re-run: run_training() skips any (S, seed) that
already has a summary.json, so raw per-run outputs are never overwritten.

Usage:
    python src/sweep.py [path/to/config.json]
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from pihnn_wrapper import run_training  # noqa: E402


def main(config_path: str):
    with open(config_path) as f:
        cfg = json.load(f)

    S_values = cfg["S_values"]
    n_seeds = cfg["n_seeds"]
    seed_start = cfg.get("seed_start", 0)
    n_epochs = cfg["n_epochs"]
    raw_dir = cfg["raw_dir"]

    total_runs = len(S_values) * n_seeds
    done = 0
    t_start = time.time()

    for S in S_values:
        for seed in range(seed_start, seed_start + n_seeds):
            run_id = f"S{S:.3f}_seed{seed:03d}"
            out_dir = os.path.join(raw_dir, run_id)
            already = os.path.exists(os.path.join(out_dir, "summary.json"))
            summary = run_training(S, seed, n_epochs, out_dir)
            done += 1
            tag = "skip " if already else "train"
            elapsed = time.time() - t_start
            print(
                f"[{done}/{total_runs}] {tag} {run_id}  "
                f"K_I={summary['K_I']:.4f}  train_loss={summary['final_train_loss']:.2e}  "
                f"lefm_verif_err={summary['lefm_verification_error']:.2%}  "
                f"elapsed={elapsed:.0f}s",
                flush=True,
            )


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/sweep_config.json"
    main(config_path)
