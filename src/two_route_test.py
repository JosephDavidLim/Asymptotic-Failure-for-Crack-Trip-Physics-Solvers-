"""Ground-truth test of a two-route singular-coefficient indicator.

Reads no saved models.  It imports the manufactured RK+ISBFM solver, solves
fresh fine-scale systems in memory, and writes new analysis artifacts only.
"""
import argparse
import csv
import math
from pathlib import Path
import statistics

import numpy as np

from poisson_rk_isbfm import (
    ALPHA, THETA_MAX, assemble, errors, exact_value_gradient,
    singular_value_gradient,
)
import poisson_rk_isbfm as solver


TRUE_COEFFICIENT = 1.0
DEFAULT_RHO = (0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.75, 0.90)


def route_b(nodes, basis, coefficients, rho, order=64):
    """Extract F1 coefficient by angular projection on r=rho.

    The known smooth manufactured component s=x^2-y^2 is subtracted.  The
    remaining field is projected onto sin(2theta/3), normalized by rho**alpha.
    """
    points, weights = np.polynomial.legendre.leggauss(order)
    theta = 0.5 * THETA_MAX * (points + 1.0)
    values = []
    phi = np.sin(ALPHA * theta)
    for t in theta:
        x, y = rho * math.cos(t), rho * math.sin(t)
        active, shape, _ = basis.evaluate(x, y)
        uh = float(shape @ coefficients[active]) + coefficients[-1] * singular_value_gradient(x, y)[0]
        smooth = x * x - y * y
        values.append(uh - smooth)
    values = np.asarray(values)
    integral = 0.5 * THETA_MAX * np.sum(weights * values * phi)
    normalization = 0.5 * THETA_MAX * np.sum(weights * phi * phi)
    return float(integral / (rho ** ALPHA * normalization))


def solve_case(n, quadrature_order=4, extra_mode=0.0):
    solver.EXTRA_MODE_AMPLITUDE = extra_mode
    nodes, basis, coefficients, h, condition = assemble(
        n, domain_order=quadrature_order, boundary_order=quadrature_order
    )
    e0, e1, egrad = errors(n, basis, coefficients)
    return nodes, basis, coefficients, {
        "n": n, "h": h, "nodes": len(nodes), "normalized_L2": e0,
        "normalized_H1": e1, "normalized_H1_seminorm": egrad,
        "condition_number": condition, "extra_mode": extra_mode,
    }


def metrics(nodes, basis, coefficients, metadata, radii):
    c_a = float(coefficients[-1])
    extracted = [(rho, route_b(nodes, basis, coefficients, rho)) for rho in radii]
    c_b = statistics.fmean(c for _, c in extracted)
    g_values = [abs(c_a - c) / abs(c) for _, c in extracted]
    return {
        **metadata, "c_A": c_a, "c_B_mean": c_b,
        "c_B_min": min(c for _, c in extracted),
        "c_B_max": max(c for _, c in extracted),
        "c_B_spread": max(c for _, c in extracted) - min(c for _, c in extracted),
        "g_mean": statistics.fmean(g_values), "g_max": max(g_values),
        "err_A": abs(c_a - TRUE_COEFFICIENT), "err_B": abs(c_b - TRUE_COEFFICIENT),
        "radius_values": extracted, "radius_g": g_values,
    }


def spearman(x, y):
    from scipy.stats import spearmanr
    result = spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def write_csv(rows, path):
    fields = ["experiment", "level", "n", "quadrature_order", "extra_mode", "nodes",
              "normalized_L2", "normalized_H1", "normalized_H1_seminorm",
              "c_A", "c_B_mean", "c_B_min", "c_B_max", "c_B_spread",
              "g_mean", "g_max", "err_A", "err_B", "condition_number"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def save_plot(rows, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    usable = [r for r in rows if r["experiment"] in ("coarsen", "underintegrate", "wrong_enrichment")]
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for name, marker in (("coarsen", "o"), ("underintegrate", "s"), ("wrong_enrichment", "^")):
        subset = [r for r in usable if r["experiment"] == name]
        if subset:
            ax.loglog([r["normalized_H1"] for r in subset], [r["g_mean"] for r in subset], marker + "-", label=name)
    ax.set_xlabel("True normalized H1 error")
    ax.set_ylabel("Two-route disagreement g")
    ax.grid(True, which="both", alpha=0.25); ax.legend(); fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=180); plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/processed"))
    args = parser.parse_args()
    radii = DEFAULT_RHO
    all_rows = []

    print("Experiment 1: baseline")
    nodes, basis, coefficients, metadata = solve_case(24, 4, 0.0)
    baseline = metrics(nodes, basis, coefficients, metadata, radii)
    print("rho       c_B          g")
    for rho, c in baseline["radius_values"]:
        print(f"{rho:5.2f}  {c: .10f}  {abs(baseline['c_A']-c)/abs(c):.4e}")
    print(f"c_A={baseline['c_A']:.12g}, c_true=1, err_A={baseline['err_A']:.4e}, "
          f"mean c_B={baseline['c_B_mean']:.12g}, err_B={baseline['err_B']:.4e}, "
          f"mean g={baseline['g_mean']:.4e}")
    baseline["experiment"], baseline["level"] = "baseline", "n24"
    all_rows.append(baseline)

    print("\nExperiment 2: contour-radius robustness")
    print(f"c_B range={baseline['c_B_min']:.10f}..{baseline['c_B_max']:.10f}; "
          f"spread={baseline['c_B_spread']:.4e}; g range="
          f"{min(baseline['radius_g']):.4e}..{max(baseline['radius_g']):.4e}")

    print("\nExperiment 3: degradation")
    for n in (8, 12, 16, 24):
        nodes, basis, coefficients, metadata = solve_case(n, 4, 0.0)
        row = metrics(nodes, basis, coefficients, metadata, radii); row["experiment"], row["level"] = "coarsen", f"n{n}"; all_rows.append(row)
        print(f"coarsen n={n}: H1={row['normalized_H1']:.5g}, err_A={row['err_A']:.4e}, err_B={row['err_B']:.4e}, g={row['g_mean']:.4e}")
    for q in (2, 3, 4):
        nodes, basis, coefficients, metadata = solve_case(24, q, 0.0)
        row = metrics(nodes, basis, coefficients, metadata, radii); row["experiment"], row["level"] = "underintegrate", f"q{q}"; all_rows.append(row)
        print(f"underintegrate q={q}: H1={row['normalized_H1']:.5g}, err_A={row['err_A']:.4e}, err_B={row['err_B']:.4e}, g={row['g_mean']:.4e}")
    nodes, basis, coefficients, metadata = solve_case(24, 4, 0.25)
    row = metrics(nodes, basis, coefficients, metadata, radii); row["experiment"], row["level"] = "wrong_enrichment", "F2_amp0.25"; all_rows.append(row)
    print(f"wrong enrichment: H1={row['normalized_H1']:.5g}, err_A={row['err_A']:.4e}, err_B={row['err_B']:.4e}, g={row['g_mean']:.4e}")

    degradation = [r for r in all_rows if r["experiment"] != "baseline"]
    rho, p = spearman([r["g_mean"] for r in degradation], [r["err_A"] for r in degradation])
    print(f"\nSpearman rho(g, err_A)={rho:.6f}, p={p:.6g}")
    print("Verdict: Route B is radius-stable when the enrichment is exact; g is "
          "a direct two-route consistency signal, but its tracking of true field "
          "error must be judged from the degradation table and is not guaranteed.")
    print("Wrong-enrichment verdict: the F2 mismatch is included explicitly above; "
          "compare its g and true errors with the exact-enrichment rows.")
    write_csv(all_rows, args.output_dir / "two_route_test.csv")
    plotted = save_plot(all_rows, args.output_dir / "two_route_test_g_vs_true_error.png")
    print("CSV:", args.output_dir / "two_route_test.csv")
    print("Plot:", args.output_dir / "two_route_test_g_vs_true_error.png" if plotted else "skipped")


if __name__ == "__main__":
    main()
