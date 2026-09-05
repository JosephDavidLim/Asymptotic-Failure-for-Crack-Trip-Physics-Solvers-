"""Compare contour-radius drift across saved trained models, without training.

With no positional arguments, all ``S*_seed*`` directories under
``results/raw`` are swept. Existing run artifacts are only read.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

from pihnn_wrapper import SIF_INTEGRATION_POINTS
from radius_sweep import (
    DEFAULT_MAX_RADIUS,
    DEFAULT_MIN_RADIUS,
    DEFAULT_NUM_RADII,
    log_spaced,
    sweep_run,
)


DEFAULT_RAW_DIR = Path("results/raw")
DEFAULT_CSV = Path("results/processed/radius_sweep_multi_summary.csv")
DEFAULT_PLOT = Path("results/processed/radius_sweep_multi_loss_vs_cv.png")


def final_training_loss(run_dir: Path) -> tuple[float, str]:
    """Read final loss from summary.json, falling back to loss.dat."""
    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        with summary_path.open() as file:
            value = json.load(file).get("final_train_loss")
        if value is not None and math.isfinite(float(value)):
            return float(value), "summary.json:final_train_loss"

    loss_path = run_dir / "loss.dat"
    if loss_path.is_file():
        lines = [line for line in loss_path.read_text().splitlines() if line.strip()]
        if lines:
            value = float(lines[-1].split()[0])
            if math.isfinite(value):
                return value, "loss.dat:last_row_train_column"
    raise ValueError("no finite final training loss in summary.json or loss.dat")


def rankdata(values: list[float]) -> list[float]:
    """Return average ranks (one-based), including correct handling of ties."""
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        average_rank = (start + 1 + stop) / 2.0
        for position in range(start, stop):
            ranks[order[position]] = average_rank
        start = stop
    return ranks


def spearman(x: list[float], y: list[float]) -> float | None:
    """Compute Spearman's rho as Pearson correlation of average ranks."""
    if len(x) != len(y) or len(x) < 2:
        return None
    rx, ry = rankdata(x), rankdata(y)
    mean_x, mean_y = statistics.fmean(rx), statistics.fmean(ry)
    dx = [value - mean_x for value in rx]
    dy = [value - mean_y for value in ry]
    denominator = math.sqrt(sum(v * v for v in dx) * sum(v * v for v in dy))
    return sum(a * b for a, b in zip(dx, dy)) / denominator if denominator else None


def summarize_run(run_dir: Path, radii: list[float], tip_index: int,
                  integration_points: int) -> dict:
    loss, loss_source = final_training_loss(run_dir)
    rows, stats = sweep_run(run_dir, radii, tip_index, integration_points)
    magnitudes = [row["K_J_abs"] for row in rows]
    g_small, g_large = rows[0]["g"], rows[-1]["g"]
    mean_magnitude = stats["K_J_abs_mean"]
    return {
        "run": run_dir.name,
        "S": rows[0]["S"],
        "seed": rows[0]["seed"],
        "final_train_loss": loss,
        "loss_source": loss_source,
        "K_c_abs": rows[0]["K_c_abs"],
        "K_J_abs_mean": mean_magnitude,
        "K_J_abs_cv": stats["K_J_abs_cv_population"],
        "K_J_abs_total_drift": (
            (max(magnitudes) - min(magnitudes)) / mean_magnitude
            if mean_magnitude else None
        ),
        "g_smallest_radius": g_small,
        "g_largest_radius": g_large,
        "g_endpoint_spread": (
            abs(g_large - g_small) if g_small is not None and g_large is not None
            else None
        ),
    }


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_plot(rows: list[dict], path: Path) -> tuple[bool, bool]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False, False

    losses = [row["final_train_loss"] for row in rows]
    cvs = [row["K_J_abs_cv"] for row in rows]
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    axis.scatter(losses, cvs, alpha=0.75)
    positive_losses = [loss for loss in losses if loss > 0]
    log_x = (len(positive_losses) == len(losses)
             and max(losses) / min(losses) >= 100.0)
    if log_x:
        axis.set_xscale("log")
    axis.set_xlabel("Final training loss")
    axis.set_ylabel(r"CV of $|K_J|$")
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return True, log_x


def formatted(value, percentage: bool = False) -> str:
    if value is None:
        return "NA"
    return f"{100.0 * value:.4f}%" if percentage else f"{value:.7g}"


def print_table(rows: list[dict]) -> None:
    headings = ("run", "loss", "|K_c|", "mean|K_J|", "CV", "drift",
                "g(r_min)", "g(r_max)", "g spread")
    rendered = []
    for row in rows:
        rendered.append((
            row["run"], f'{row["final_train_loss"]:.6e}',
            formatted(row["K_c_abs"]), formatted(row["K_J_abs_mean"]),
            formatted(row["K_J_abs_cv"], True),
            formatted(row["K_J_abs_total_drift"], True),
            formatted(row["g_smallest_radius"], True),
            formatted(row["g_largest_radius"], True),
            formatted(row["g_endpoint_spread"], True),
        ))
    widths = [max(len(headings[i]), *(len(row[i]) for row in rendered))
              for i in range(len(headings))]
    print("  ".join(text.ljust(widths[i]) for i, text in enumerate(headings)))
    print("  ".join("-" * width for width in widths))
    for row in rendered:
        print("  ".join(text.ljust(widths[i]) for i, text in enumerate(row)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="*", help="saved run directories")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--plot", type=Path, default=DEFAULT_PLOT)
    parser.add_argument("--min-radius", type=float, default=DEFAULT_MIN_RADIUS)
    parser.add_argument("--max-radius", type=float, default=DEFAULT_MAX_RADIUS)
    parser.add_argument("--num-radii", type=int, default=DEFAULT_NUM_RADII)
    parser.add_argument("--tip-index", type=int, default=0)
    parser.add_argument("--integration-points", type=int,
                        default=SIF_INTEGRATION_POINTS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs = ([Path(name) for name in args.run_dirs] if args.run_dirs else
                sorted(args.raw_dir.glob("S*_seed*")))
    radii = log_spaced(args.min_radius, args.max_radius, args.num_radii)
    print(f"Settings: radii={args.min_radius:g}..{args.max_radius:g} "
          f"({args.num_radii} log-spaced), integration_points="
          f"{args.integration_points}, tip_index={args.tip_index}")

    summaries, skipped = [], []
    for index, run_dir in enumerate(run_dirs, 1):
        try:
            summaries.append(summarize_run(
                run_dir, radii, args.tip_index, args.integration_points))
            print(f"[{index}/{len(run_dirs)}] {run_dir.name}", flush=True)
        except Exception as error:
            skipped.append((str(run_dir), f"{type(error).__name__}: {error}"))
            print(f"[{index}/{len(run_dirs)}] SKIP {run_dir}: {error}", flush=True)

    if not summaries:
        raise SystemExit("No runs completed successfully; no output written.")
    summaries.sort(key=lambda row: row["final_train_loss"])
    write_csv(summaries, args.csv)
    plotted, log_x = save_plot(summaries, args.plot)
    rho = spearman(
        [row["final_train_loss"] for row in summaries],
        [row["K_J_abs_cv"] for row in summaries],
    )

    print("\nSummary (sorted by final training loss):")
    print_table(summaries)
    print(f"\nSpearman rho(loss, CV(|K_J|)) = "
          f"{rho:.6f}" if rho is not None else
          "\nSpearman rho(loss, CV(|K_J|)) = undefined")
    if rho is not None:
        direction = "more" if rho > 0 else "less"
        print(f"Plain meaning: higher-loss runs tend to have {direction} "
              "radius drift." if rho != 0 else
              "Plain meaning: loss rank and radius-drift rank show no association.")
    print(f"CSV: {args.csv}")
    print(f"Plot: {args.plot} ({'log' if log_x else 'linear'} x-axis)" if plotted
          else "Plot: skipped (matplotlib unavailable)")
    if skipped:
        print("Skipped runs:")
        for run_dir, reason in skipped:
            print(f"  {run_dir}: {reason}")
    else:
        print("Skipped runs: none")


if __name__ == "__main__":
    main()
