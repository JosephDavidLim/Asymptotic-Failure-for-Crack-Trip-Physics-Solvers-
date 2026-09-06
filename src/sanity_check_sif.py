"""Print K_c, K_J, and g for one existing trained model without modifying it.

Usage:
    python src/sanity_check_sif.py [results/raw/S0.200_seed000]
"""
import json
import os
import sys

from field_fit import load_trained_model
from pihnn_wrapper import compute_sif_diagnostics


def main(run_dir: str) -> None:
    summary_path = os.path.join(run_dir, "summary.json")
    model_path = os.path.join(run_dir, "model.dict")
    with open(summary_path) as f:
        summary = json.load(f)

    model, _, crack = load_trained_model(summary["S"], model_path)
    diagnostic = compute_sif_diagnostics(model, crack)

    # Display complex values in the requested K = K_I - i*K_II convention.
    K_c = complex(diagnostic["K_c_I"], -diagnostic["K_c_II"])
    K_J = complex(diagnostic["K_J_I"], -diagnostic["K_J_II"])
    print(f"run: {run_dir}")
    print(f"K_c = {K_c:.12g}  (K_I - i*K_II)")
    print(f"K_J = {K_J:.12g}  (K_I - i*K_II)")
    print(f"g   = {diagnostic['g']:.12g}" if diagnostic["g"] is not None else "g   = undefined (|K_J| = 0)")
    print(
        f"contour: tip={diagnostic['sif_tip_index']}, "
        f"radius={diagnostic['sif_contour_radius']}, "
        f"points={diagnostic['sif_integration_points']}"
    )


if __name__ == "__main__":
    default_run = "results/raw/S0.200_seed000"
    main(sys.argv[1] if len(sys.argv) > 1 else default_run)
