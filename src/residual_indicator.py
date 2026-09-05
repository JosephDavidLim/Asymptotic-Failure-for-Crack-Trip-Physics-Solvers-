"""Residual-based a posteriori indicator for the manufactured RK+ISBFM solve.

This script imports the solver, solves only in memory, and writes new analysis
artifacts.  The enrichment is harmonic and is excluded analytically from the
interior Laplacian; the RK smooth component is differentiated with a symmetric
five-point stencil.
"""
import argparse
import csv
import math
from pathlib import Path
import statistics

import numpy as np

import poisson_rk_isbfm as solver
from poisson_rk_isbfm import assemble, domain_cells, errors, gauss_rule, make_nodes, singular_value_gradient


EXCLUSIONS = (0.0, 0.05, 0.10, 0.20, 0.30)
WEIGHT_RADIUS = 0.30


def valid_domain(x, y):
    return -1.0 <= x <= 1.0 and -1.0 <= y <= 1.0 and not (x > 0.0 and y < 0.0)


def smooth_value(basis, coefficients, x, y):
    active, values, _ = basis.evaluate(x, y)
    return float(values @ coefficients[active])


def smooth_gradient(basis, coefficients, x, y):
    active, _, gradients = basis.evaluate(x, y)
    return gradients.T @ coefficients[active]


def laplacian_smooth(basis, coefficients, x, y, h):
    delta = 0.1 * h
    points = ((x + delta, y), (x - delta, y), (x, y + delta), (x, y - delta))
    if not all(valid_domain(*point) for point in points):
        return None
    center = smooth_value(basis, coefficients, x, y)
    return (sum(smooth_value(basis, coefficients, *point) for point in points) - 4.0 * center) / delta ** 2


def interior_indicators(n, basis, coefficients, h, order, exclusions):
    points, weights = gauss_rule(order)
    sums = {radius: 0.0 for radius in exclusions}
    weighted = 0.0
    unweighted = 0.0
    for x0, y0, cell_h in domain_cells(n):
        for px, wx in zip(points, weights):
            x = x0 + 0.5 * cell_h * (px + 1.0)
            for py, wy in zip(points, weights):
                y = y0 + 0.5 * cell_h * (py + 1.0)
                residual = laplacian_smooth(basis, coefficients, x, y, h)
                if residual is None:
                    continue
                jac = 0.25 * cell_h * cell_h * wx * wy
                radius = math.hypot(x, y)
                for exclusion in exclusions:
                    if radius >= exclusion:
                        sums[exclusion] += jac * residual * residual
                weight = min(1.0, (radius / WEIGHT_RADIUS) ** 2)
                weighted += jac * (weight * residual) ** 2
                unweighted += jac * residual * residual
    return {f"eta_excl_{radius:g}": math.sqrt(value) for radius, value in sums.items()} | {
        "eta_weighted": math.sqrt(weighted), "eta_naive": math.sqrt(unweighted)
    }


def boundary_indicators(n, basis, coefficients, h, order):
    points, weights = gauss_rule(order)
    dirichlet = 0.0
    neumann = 0.0
    for kind, x0, y0, dx, dy in solver.boundary_segments(n):
        length = math.hypot(dx, dy)
        normal = (np.array([1.0, 0.0]) if kind == "g" and abs(dx) < 1e-15 else
                  np.array([0.0, -1.0]) if kind == "g" else
                  np.array([0.0, -1.0]) if kind == "h_bottom" else
                  np.array([1.0, 0.0]) if kind == "h_right" else
                  np.array([0.0, 1.0]) if kind == "h_top" else
                  np.array([-1.0, 0.0]))
        for p, w in zip(points, weights):
            x, y = x0 + 0.5 * (p + 1.0) * dx, y0 + 0.5 * (p + 1.0) * dy
            value, gradient = solver.exact_value_gradient(x, y)
            uh = smooth_value(basis, coefficients, x, y) + coefficients[-1] * singular_value_gradient(x, y)[0]
            grad_uh = smooth_gradient(basis, coefficients, x, y) + coefficients[-1] * singular_value_gradient(x, y)[1]
            jac = 0.5 * length * w
            if kind == "g":
                dirichlet += jac * (uh - value) ** 2
            else:
                neumann += jac * (float(grad_uh @ normal) - float(gradient @ normal)) ** 2
    return math.sqrt(dirichlet), math.sqrt(neumann)


def smooth_error(n, basis, coefficients, order=8):
    points, weights = gauss_rule(order)
    e0 = e1 = norm0 = norm1 = 0.0
    for x0, y0, cell_h in domain_cells(n):
        for px, wx in zip(points, weights):
            x = x0 + 0.5 * cell_h * (px + 1.0)
            for py, wy in zip(points, weights):
                y = y0 + 0.5 * cell_h * (py + 1.0)
                uh = smooth_value(basis, coefficients, x, y)
                gh = smooth_gradient(basis, coefficients, x, y)
                exact = x * x - y * y
                ge = np.array([2.0 * x, -2.0 * y])
                jac = 0.25 * cell_h * cell_h * wx * wy
                e0 += jac * (uh - exact) ** 2; norm0 += jac * exact ** 2
                e1 += jac * ((uh - exact) ** 2 + np.dot(gh - ge, gh - ge))
                norm1 += jac * (exact * exact + np.dot(ge, ge))
    return math.sqrt(e0 / norm0), math.sqrt(e1 / norm1)


def solve_case(n, q):
    solver.EXTRA_MODE_AMPLITUDE = 0.0
    nodes, basis, coefficients, h, condition = assemble(n, domain_order=q, boundary_order=q)
    e0, e1, _ = errors(n, basis, coefficients)
    smooth0, smooth1 = smooth_error(n, basis, coefficients)
    indicators = interior_indicators(n, basis, coefficients, h, 8, EXCLUSIONS)
    dirichlet, neumann = boundary_indicators(n, basis, coefficients, h, 8)
    indicators.update(eta_dirichlet=dirichlet, eta_neumann=neumann,
                      eta_boundary=math.hypot(dirichlet, neumann),
                      eta_total=math.sqrt(indicators["eta_excl_0"] ** 2 + dirichlet ** 2 + neumann ** 2))
    return {"n": n, "quadrature_order": q, "nodes": len(nodes),
            "normalized_L2": e0, "normalized_H1": e1,
            "smooth_normalized_L2": smooth0, "smooth_normalized_H1": smooth1,
            **indicators, "condition_number": condition}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/processed"))
    args = parser.parse_args()
    cases = [(8, 4), (12, 4), (16, 4), (24, 4), (24, 2)]
    rows = []
    for n, q in cases:
        print(f"Solving n={n}, quadrature={q} ...", flush=True)
        row = solve_case(n, q)
        rows.append(row)
        print(f"  smooth H1={row['smooth_normalized_H1']:.6g}, "
              f"eta_excl_0.1={row['eta_excl_0.1']:.6g}, "
              f"eta_weighted={row['eta_weighted']:.6g}")
    fields = list(rows[0])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "residual_indicator.csv"
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader(); writer.writerows(rows)

    from scipy.stats import spearmanr
    print("\nIndicator correlations with smooth normalized H1 error:")
    for key in ["eta_naive", "eta_excl_0", "eta_excl_0.05", "eta_excl_0.1", "eta_excl_0.2", "eta_excl_0.3", "eta_weighted", "eta_boundary", "eta_total"]:
        rho, p = spearmanr([row[key] for row in rows], [row["smooth_normalized_H1"] for row in rows])
        print(f"{key:15s} rho={rho: .6f}, p={p:.6g}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        for key, label, marker in (("eta_naive", "naive", "o"), ("eta_excl_0.1", "exclude r<0.1", "s"), ("eta_weighted", "weighted", "^")):
            ax.loglog([row["smooth_normalized_H1"] for row in rows], [row[key] for row in rows], marker + "-", label=label)
        ax.set_xlabel("True smooth normalized H1 error"); ax.set_ylabel("Residual indicator")
        ax.grid(True, which="both", alpha=.25); ax.legend(); fig.tight_layout()
        plot = args.output_dir / "residual_indicator_vs_true_error.png"; fig.savefig(plot, dpi=180); plt.close(fig)
        print("Plot:", plot)
    except ImportError:
        print("Plot: skipped (matplotlib unavailable)")
    print("CSV:", output)


if __name__ == "__main__":
    main()
