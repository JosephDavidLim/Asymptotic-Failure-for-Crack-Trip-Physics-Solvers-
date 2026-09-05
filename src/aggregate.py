"""
Scans results/raw/*/summary.json (one per (S, seed) run) and regenerates
results/processed/all_runs.csv. Purely derived data -- safe to re-run any
time; never reads or writes the raw per-run directories except to read
their summary.json.

Also computes, per run, the plasticity-reference error (against reference.py's
Dugdale model) and the reference-uncertainty flag, since those depend on a
module (reference.py) that may be edited/refined independently of re-running
the (expensive) training sweep.

Usage:
    python src/aggregate.py [results/raw] [results/processed/all_runs.csv]
"""
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import reference  # noqa: E402


FIELDS = [
    "S", "seed", "n_epochs", "wall_time_sec",
    "K_I", "K_II",
    "K_c_I", "K_c_II", "K_J_I", "K_J_II", "g", "g_status",
    "sif_tip_index", "sif_contour_radius", "sif_integration_points",
    "sif_library_complex_convention", "sif_report_complex_convention",
    "final_train_loss", "final_test_loss",
    "normalized_train_loss", "normalized_test_loss",
    "plateau_ratio",
    "K_el_exact", "lefm_verification_error",
    "K_ref_dugdale", "K_ref_irwin", "reference_disagreement",
    "plastic_reference_error",
    "dugdale_zone_ratio",
    "out_dir",
]


def load_runs(raw_dir: str):
    rows = []
    for path in sorted(glob.glob(os.path.join(raw_dir, "*", "summary.json"))):
        with open(path) as f:
            s = json.load(f)
        S = s["S"]
        K_ref = reference.k_ref_dugdale(S)
        K_irwin = reference.k_ref_irwin_plane_strain(S)
        row = dict(s)
        row["K_ref_dugdale"] = K_ref
        row["K_ref_irwin"] = K_irwin
        row["reference_disagreement"] = reference.reference_disagreement(S)
        row["plastic_reference_error"] = abs(s["K_I"] - K_ref) / K_ref if K_ref not in (0, float("inf")) else float("nan")
        row["dugdale_zone_ratio"] = reference.dugdale_zone_ratio(S)
        rows.append(row)
    return rows


def main(raw_dir: str, out_csv: str):
    rows = load_runs(raw_dir)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in FIELDS})
    print(f"Wrote {len(rows)} runs to {out_csv}")


if __name__ == "__main__":
    raw_dir = sys.argv[1] if len(sys.argv) > 1 else "results/raw"
    out_csv = sys.argv[2] if len(sys.argv) > 2 else "results/processed/all_runs.csv"
    main(raw_dir, out_csv)
