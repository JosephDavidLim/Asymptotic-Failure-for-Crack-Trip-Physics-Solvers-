"""Measure K_J and g sensitivity to contour radius on saved trained models.

This script only reads ``summary.json`` and ``model.dict`` from each run. It
does not train a model or modify any existing run artifact.

Usage:
    python src/radius_sweep.py [RUN_DIR ...]

With no RUN_DIR, ``results/raw/S0.200_seed000`` is used. CSV and, when
matplotlib is installed, PNG outputs are written under ``results/processed``.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics

from field_fit import load_trained_model
from pihnn_wrapper import SIF_INTEGRATION_POINTS, compute_sif_diagnostics
from reference import A_CRACK, H_HALF, L_HALF


DEFAULT_RUN = "results/raw/S0.200_seed000"
DEFAULT_MIN_RADIUS = 0.05
DEFAULT_MAX_RADIUS = 1.5
DEFAULT_NUM_RADII = 18


def log_spaced(start: float, stop: float, count: int) -> list[float]:
    """Return ``count`` logarithmically spaced values, including endpoints."""
    if start <= 0 or stop <= start:
        raise ValueError("radii must satisfy 0 < min_radius < max_radius")
    if count < 2:
        raise ValueError("num_radii must be at least 2")
    log_start = math.log(start)
    step = (math.log(stop) - log_start) / (count - 1)
    return [math.exp(log_start + i * step) for i in range(count)]


def geometry_radius_limit(crack, tip_index: int) -> tuple[float, dict[str, float]]:
    """Find the first other-tip or rectangular-boundary distance from a tip."""
    tip = crack.tips[tip_index]
    x = float(tip.coords.real)
    y = float(tip.coords.imag)
    distances = {
        "left_boundary": x + L_HALF,
        "right_boundary": L_HALF - x,
        "bottom_boundary": y + H_HALF,
        "top_boundary": H_HALF - y,
    }
    for index, other in enumerate(crack.tips):
        if index != tip_index:
            distances[f"tip_{index}"] = float(abs(other.coords - tip.coords))
    return min(distances.values()), distances


def sweep_run(
    run_dir: Path,
    radii: list[float],
    tip_index: int,
    integration_points: int,
) -> tuple[list[dict], dict]:
    summary_path = run_dir / "summary.json"
    model_path = run_dir / "model.dict"
    if not summary_path.is_file() or not model_path.is_file():
        raise FileNotFoundError(
            f"{run_dir} must contain both summary.json and model.dict"
        )

    with summary_path.open() as file:
        saved_summary = json.load(file)
    model, _, crack = load_trained_model(saved_summary["S"], str(model_path))

    if not 0 <= tip_index < len(crack.tips):
        raise IndexError(f"tip_index {tip_index} is outside [0, {len(crack.tips)})")
    radius_limit, distances = geometry_radius_limit(crack, tip_index)
    if max(radii) >= radius_limit:
        raise ValueError(
            f"maximum radius {max(radii):g} must be below the geometry limit "
            f"{radius_limit:g}; distances are {distances}"
        )

    coefficient = model.sif[tip_index].detach()
    K_c_I = float(coefficient.real)
    K_c_II = float(coefficient.imag)
    K_c_abs = math.hypot(K_c_I, K_c_II)

    rows = []
    for radius in radii:
        diagnostic = compute_sif_diagnostics(
            model,
            crack,
            tip_index=tip_index,
            radius=radius,
            integration_points=integration_points,
        )
        if diagnostic["K_c_I"] != K_c_I or diagnostic["K_c_II"] != K_c_II:
            raise RuntimeError("K_c changed during the radius sweep")
        rows.append(
            {
                "run_dir": str(run_dir),
                "S": saved_summary["S"],
                "seed": saved_summary["seed"],
                "tip_index": tip_index,
                "integration_points": integration_points,
                "radius": radius,
                "K_c_I": K_c_I,
                "K_c_II": K_c_II,
                "K_c_abs": K_c_abs,
                "K_J_I": diagnostic["K_J_I"],
                "K_J_II": diagnostic["K_J_II"],
                "K_J_abs": math.hypot(
                    diagnostic["K_J_I"], diagnostic["K_J_II"]
                ),
                "g": diagnostic["g"],
            }
        )

    magnitudes = [row["K_J_abs"] for row in rows]
    defined_g = [row["g"] for row in rows if row["g"] is not None]
    mean_magnitude = statistics.fmean(magnitudes)
    stats = {
        "K_J_abs_mean": mean_magnitude,
        "K_J_abs_std_population": statistics.pstdev(magnitudes),
        "K_J_abs_cv_population": (
            statistics.pstdev(magnitudes) / mean_magnitude
            if mean_magnitude != 0.0
            else None
        ),
        "g_min": min(defined_g) if defined_g else None,
        "g_max": max(defined_g) if defined_g else None,
        "geometry_radius_limit": radius_limit,
        "geometry_distances": distances,
    }
    return rows, stats


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_plot(rows: list[dict], path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    radii = [row["radius"] for row in rows]
    figure, axis_k = plt.subplots(figsize=(7.5, 4.8))
    axis_g = axis_k.twinx()
    line_i = axis_k.plot(radii, [row["K_J_I"] for row in rows], "o-", label="K_J_I")
    line_ii = axis_k.plot(radii, [row["K_J_II"] for row in rows], "s-", label="K_J_II")
    line_g = axis_g.plot(radii, [row["g"] for row in rows], "^-", color="tab:red", label="g")
    axis_k.set_xscale("log")
    axis_k.set_xlabel("Contour radius")
    axis_k.set_ylabel("Interaction-integral SIF")
    axis_g.set_ylabel("g", color="tab:red")
    axis_g.tick_params(axis="y", colors="tab:red")
    axis_k.grid(True, which="both", alpha=0.25)
    lines = line_i + line_ii + line_g
    axis_k.legend(lines, [line.get_label() for line in lines], loc="best")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return True


def print_report(run_dir: Path, rows: list[dict], stats: dict, csv_path: Path, plot_path: Path, plotted: bool) -> None:
    first = rows[0]
    print(f"\nRun: {run_dir}")
    print(f"K_c fixed: K_I={first['K_c_I']:.10f}, K_II={first['K_c_II']:.10f}, |K_c|={first['K_c_abs']:.10f}")
    print(f"Geometry radius limit: r < {stats['geometry_radius_limit']:.6g}")
    print(" radius       K_J_I        K_J_II       |K_J|          g")
    print("--------  ------------  ------------  ------------  ------------")
    for row in rows:
        g_text = f"{row['g']:.10f}" if row["g"] is not None else "undefined"
        print(
            f"{row['radius']:8.5f}  {row['K_J_I']:12.9f}  "
            f"{row['K_J_II']:12.9f}  {row['K_J_abs']:12.9f}  {g_text:>12}"
        )
    cv = stats["K_J_abs_cv_population"]
    print("Summary (population standard deviation):")
    print(f"  mean(|K_J|) = {stats['K_J_abs_mean']:.12g}")
    print(f"  std(|K_J|)  = {stats['K_J_abs_std_population']:.12g}")
    print(f"  CV(|K_J|)   = {cv:.12g}" if cv is not None else "  CV(|K_J|)   = undefined")
    print(f"  min(g)      = {stats['g_min']:.12g}" if stats["g_min"] is not None else "  min(g)      = undefined")
    print(f"  max(g)      = {stats['g_max']:.12g}" if stats["g_max"] is not None else "  max(g)      = undefined")
    print(f"CSV: {csv_path}")
    print(f"Plot: {plot_path}" if plotted else "Plot: skipped (matplotlib unavailable)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="*", default=[DEFAULT_RUN])
    parser.add_argument("--min-radius", type=float, default=DEFAULT_MIN_RADIUS)
    parser.add_argument("--max-radius", type=float, default=DEFAULT_MAX_RADIUS)
    parser.add_argument("--num-radii", type=int, default=DEFAULT_NUM_RADII)
    parser.add_argument("--tip-index", type=int, default=0)
    parser.add_argument("--integration-points", type=int, default=SIF_INTEGRATION_POINTS)
    parser.add_argument("--output-dir", default="results/processed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    radii = log_spaced(args.min_radius, args.max_radius, args.num_radii)
    output_dir = Path(args.output_dir)
    for run_name in args.run_dirs:
        run_dir = Path(run_name)
        output_stem = f"radius_sweep_{run_dir.name}_tip{args.tip_index}"
        csv_path = output_dir / f"{output_stem}.csv"
        plot_path = output_dir / f"{output_stem}.png"
        rows, stats = sweep_run(
            run_dir, radii, args.tip_index, args.integration_points
        )
        write_csv(rows, csv_path)
        plotted = save_plot(rows, plot_path)
        print_report(run_dir, rows, stats, csv_path, plot_path, plotted)


if __name__ == "__main__":
    main()
