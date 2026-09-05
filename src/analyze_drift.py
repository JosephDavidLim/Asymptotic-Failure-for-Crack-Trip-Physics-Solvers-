"""Separate within-geometry convergence effects from between-geometry drift.

This script reads an existing multi-run radius-sweep summary CSV. It does not
load model checkpoints, evaluate models, or rerun any radius sweeps.

Usage:
    python src/analyze_drift.py
    python src/analyze_drift.py path/to/summary.csv --output path/to/output.csv
"""
import argparse
import csv
import math
from pathlib import Path
import statistics

try:
    from scipy.stats import spearmanr
except ImportError as error:
    raise SystemExit(
        "scipy is required to compute Spearman correlations and p-values"
    ) from error


DEFAULT_INPUT = Path("results/processed/radius_sweep_multi_summary.csv")
DEFAULT_OUTPUT = Path("results/processed/drift_analysis.csv")
DEFAULT_PLOT = Path("results/processed/drift_analysis_small_multiples.png")
REQUIRED_COLUMNS = {"run", "S", "seed", "final_train_loss", "K_J_abs_cv"}


def read_rows(path: Path) -> list[dict]:
    """Read and validate the radius-sweep summary."""
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        rows = list(reader)

    for row_number, row in enumerate(rows, start=2):
        try:
            row["S"] = float(row["S"])
            row["seed"] = int(row["seed"])
            row["final_train_loss"] = float(row["final_train_loss"])
            row["K_J_abs_cv"] = float(row["K_J_abs_cv"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid numeric value on CSV row {row_number}") from error
        if not all(math.isfinite(row[key]) for key in
                   ("S", "final_train_loss", "K_J_abs_cv")):
            raise ValueError(f"non-finite value on CSV row {row_number}")
        if row["final_train_loss"] <= 0:
            raise ValueError(
                f"final_train_loss must be positive on CSV row {row_number}"
            )
    if not rows:
        raise ValueError(f"{path} contains no data rows")
    return rows


def group_by_s(rows: list[dict]) -> dict[float, list[dict]]:
    groups: dict[float, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["S"], []).append(row)
    return dict(sorted(groups.items()))


def correlation(x: list[float], y: list[float]) -> tuple[float, float]:
    result = spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def analyze(groups: dict[float, list[dict]]) -> tuple[list[dict], dict]:
    per_s = []
    for s_value, rows in groups.items():
        losses = [row["final_train_loss"] for row in rows]
        cvs = [row["K_J_abs_cv"] for row in rows]
        rho, p_value = correlation(losses, cvs)
        per_s.append({
            "S": s_value,
            "n_seeds": len(rows),
            "rho_loss_vs_cv": rho,
            "p_value": p_value,
            "median_final_loss": statistics.median(losses),
            "median_cv": statistics.median(cvs),
            "min_cv": min(cvs),
            "max_cv": max(cvs),
        })

    median_losses = [row["median_final_loss"] for row in per_s]
    median_cvs = [row["median_cv"] for row in per_s]
    between_rho, between_p = correlation(median_losses, median_cvs)
    size_rho, size_p = correlation([row["S"] for row in per_s], median_cvs)
    summary = {
        "mean_within_s_rho": statistics.fmean(
            row["rho_loss_vs_cv"] for row in per_s
        ),
        "median_within_s_rho": statistics.median(
            row["rho_loss_vs_cv"] for row in per_s
        ),
        "between_s_rho": between_rho,
        "between_s_p_value": between_p,
        "size_vs_cv_rho": size_rho,
        "size_vs_cv_p_value": size_p,
    }
    return per_s, summary


def print_results(per_s: list[dict], summary: dict) -> None:
    print("Within each S: final training loss vs CV(|K_J|)")
    print("S      n_seeds       rho       p-value")
    print("-----  -------  --------  ------------")
    for row in per_s:
        print(f"{row['S']:5.3f}  {row['n_seeds']:7d}  "
              f"{row['rho_loss_vs_cv']:8.4f}  {row['p_value']:12.6g}")
    print(f"Mean within-S rho:   {summary['mean_within_s_rho']:.4f}")
    print(f"Median within-S rho: {summary['median_within_s_rho']:.4f}")

    print("\nBetween S values: medians over seeds")
    print("S      median loss   median CV")
    print("-----  ------------  ----------")
    for row in per_s:
        print(f"{row['S']:5.3f}  {row['median_final_loss']:12.6e}  "
              f"{100.0 * row['median_cv']:9.4f}%")
    print("Spearman rho(median loss, median CV) = "
          f"{summary['between_s_rho']:.4f}, "
          f"p = {summary['between_s_p_value']:.6g}")
    print("Spearman rho(S, median CV) = "
          f"{summary['size_vs_cv_rho']:.4f}, "
          f"p = {summary['size_vs_cv_p_value']:.6g}")
    direction = "more" if summary["size_vs_cv_rho"] > 0 else "less"
    print(f"Plain meaning: bigger S means {direction} relative drift.")

    print("\nDrift magnitude by S")
    print("S      median CV      min CV      max CV")
    print("-----  ---------  ----------  ----------")
    for row in per_s:
        print(f"{row['S']:5.3f}  {100.0 * row['median_cv']:8.4f}%  "
              f"{100.0 * row['min_cv']:9.4f}%  "
              f"{100.0 * row['max_cv']:9.4f}%")


def write_results(per_s: list[dict], summary: dict, path: Path) -> None:
    """Write per-S and aggregate tables to one tidy CSV."""
    fieldnames = [
        "analysis", "S", "n_seeds", "rho", "p_value",
        "median_final_loss", "median_cv", "min_cv", "max_cv", "value",
    ]
    output_rows = []
    for row in per_s:
        output_rows.append({
            "analysis": "within_s",
            "S": row["S"],
            "n_seeds": row["n_seeds"],
            "rho": row["rho_loss_vs_cv"],
            "p_value": row["p_value"],
            "median_final_loss": row["median_final_loss"],
            "median_cv": row["median_cv"],
            "min_cv": row["min_cv"],
            "max_cv": row["max_cv"],
        })
    output_rows.extend([
        {"analysis": "mean_within_s_rho",
         "value": summary["mean_within_s_rho"]},
        {"analysis": "median_within_s_rho",
         "value": summary["median_within_s_rho"]},
        {"analysis": "between_s_medians", "n_seeds": len(per_s),
         "rho": summary["between_s_rho"],
         "p_value": summary["between_s_p_value"]},
        {"analysis": "s_vs_median_cv", "n_seeds": len(per_s),
         "rho": summary["size_vs_cv_rho"],
         "p_value": summary["size_vs_cv_p_value"]},
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)


def save_plot(groups: dict[float, list[dict]], path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter
    except ImportError:
        return False

    count = len(groups)
    columns = min(5, count)
    rows_count = math.ceil(count / columns)
    figure, axes = plt.subplots(
        rows_count, columns, figsize=(3.25 * columns, 3.0 * rows_count),
        sharey=True, squeeze=False,
    )
    color_map = plt.get_cmap("viridis")
    for index, (s_value, rows) in enumerate(groups.items()):
        axis = axes.flat[index]
        losses = [row["final_train_loss"] for row in rows]
        cvs = [row["K_J_abs_cv"] for row in rows]
        rho, p_value = correlation(losses, cvs)
        color = color_map(index / max(count - 1, 1))
        axis.scatter(losses, cvs, s=28, color=color, alpha=0.8)
        axis.set_xscale("log")
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        axis.grid(True, alpha=0.25)
        axis.set_title(f"S={s_value:.3f}\nρ={rho:.2f}, p={p_value:.3g}")
        axis.set_xlabel("Final training loss")
        if index % columns == 0:
            axis.set_ylabel(r"CV of $|K_J|$")
    for index in range(count, rows_count * columns):
        axes.flat[index].set_visible(False)
    figure.suptitle("Within-S convergence versus contour-radius drift", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return True


def print_summary(summary: dict) -> None:
    mean_rho = summary["mean_within_s_rho"]
    median_rho = summary["median_within_s_rho"]
    between_rho = summary["between_s_rho"]
    within_strength = max(abs(mean_rho), abs(median_rho))
    print("\nPlain-language summary")
    if mean_rho > 0:
        print("(a) Within fixed geometry, worse-converged runs tend to drift more, "
              "but the average relationship is weak.")
    elif mean_rho < 0:
        print("(a) Within fixed geometry, the correlations do not support the "
              "claim that worse-converged runs drift more.")
    else:
        print("(a) Within fixed geometry, loss and drift show no rank relationship.")
    if between_rho < 0:
        print("(b) Across geometries, larger-S groups have less relative drift.")
    elif between_rho > 0:
        print("(b) Across geometries, larger-S groups have more relative drift.")
    else:
        print("(b) Across geometries, drift has no monotonic relationship with S.")
    stronger = "crack-size" if abs(between_rho) > within_strength else "within-S convergence"
    print(f"(c) The {stronger} effect is stronger by Spearman-correlation magnitude.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plot", type=Path, default=DEFAULT_PLOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input)
    groups = group_by_s(rows)
    per_s, summary = analyze(groups)
    print_results(per_s, summary)
    print_summary(summary)
    write_results(per_s, summary, args.output)
    plotted = save_plot(groups, args.plot)
    print(f"\nCSV: {args.output}")
    print(f"Plot: {args.plot}" if plotted else
          "Plot: skipped (matplotlib unavailable)")


if __name__ == "__main__":
    main()
