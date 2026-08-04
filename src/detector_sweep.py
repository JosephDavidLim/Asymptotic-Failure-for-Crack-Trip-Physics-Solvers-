"""
Post-hoc detector sweep: for every (S, seed) run already produced by the
prior study's src/sweep.py, compute both candidate detectors WITHOUT any
retraining:

  Detector 1 (naive field-fit, src/field_fit.py): free power-law exponent
  fit to the run's own converged near-tip field, over a radius window fixed
  identically across all S.

  Detector 2 (HRR overlap-breakdown ratio, src/hrr_reference.py): closed-form
  zeta(S) = r_hrr(S)/a, computed from the run's own reported K_I via the
  cited HRR constants.

Also carries forward the reference "true error" and "loss" columns from the
prior study's results/processed/all_runs.csv so everything needed for the
correlation analysis lives in one place.

Writes results/processed/detectors.csv. Idempotent-safe in the sense that it
only reads existing raw outputs (results/raw/**), never modifies them, and
this script can be re-run any time without affecting raw data.

Usage:
    python src/detector_sweep.py
"""
import csv
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import reference as prior_ref  # noqa: E402
import hrr_reference as hrr  # noqa: E402
from field_fit import naive_field_fit_detector  # noqa: E402

RAW_DIR = "results/raw"
OUT_CSV = "results/processed/detectors.csv"

FIELDS = [
    "S", "seed",
    "K_I", "final_train_loss", "normalized_train_loss",
    "lefm_verification_error",
    "K_ref_dugdale", "plastic_reference_error",
    "field_fit_exponent", "field_fit_exponent_deviation", "field_fit_resid_rms_log",
    "zeta_detector",
]


def main():
    run_dirs = sorted(glob.glob(os.path.join(RAW_DIR, "*")))
    rows = []
    t0 = time.time()
    for i, run_dir in enumerate(run_dirs):
        summary_path = os.path.join(run_dir, "summary.json")
        model_path = os.path.join(run_dir, "model.dict")
        if not (os.path.exists(summary_path) and os.path.exists(model_path)):
            continue
        with open(summary_path) as f:
            s = json.load(f)
        S = s["S"]
        K_I = s["K_I"]

        K_ref = prior_ref.k_ref_dugdale(S)
        plastic_err = abs(K_I - K_ref) / K_ref if K_ref not in (0, float("inf")) else float("nan")

        field_res = naive_field_fit_detector(S, model_path)

        zeta = hrr.detector_zeta(K_I)

        row = {
            "S": S,
            "seed": s["seed"],
            "K_I": K_I,
            "final_train_loss": s["final_train_loss"],
            "normalized_train_loss": s["normalized_train_loss"],
            "lefm_verification_error": s["lefm_verification_error"],
            "K_ref_dugdale": K_ref,
            "plastic_reference_error": plastic_err,
            "field_fit_exponent": field_res["field_fit_exponent"],
            "field_fit_exponent_deviation": field_res["field_fit_exponent_deviation"],
            "field_fit_resid_rms_log": field_res["field_fit_resid_rms_log"],
            "zeta_detector": zeta,
        }
        rows.append(row)
        if (i + 1) % 20 == 0:
            print(f"[{i+1}/{len(run_dirs)}] elapsed={time.time()-t0:.0f}s", flush=True)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Wrote {len(rows)} runs to {OUT_CSV} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
