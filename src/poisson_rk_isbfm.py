"""Fine-scale RK + ISBFM Galerkin solve for a manufactured L-corner problem.

The enrichment is integrated only on boundaries (ISBFM); no domain quadrature
of the singular function or its gradient is performed.  The manufactured
solution is F_1 + x**2-y**2, with F_1=r**(2/3)sin(2theta/3).
"""
import argparse
import csv
import math
from pathlib import Path

import numpy as np


ALPHA = 2.0 / 3.0
THETA_MAX = 1.5 * math.pi
SUPPORT_FACTOR = 1.51
BETA_FACTOR = 100.0
DEFAULT_LEVELS = (8, 12, 16, 24)
EXTRA_MODE_AMPLITUDE = 0.0


def polar(x, y):
    r = math.hypot(x, y)
    theta = math.atan2(y, x)
    if theta < 0.0:
        theta += 2.0 * math.pi
    return r, theta


def singular_value_gradient(x, y):
    r, theta = polar(x, y)
    if r == 0.0:
        return 0.0, np.array([0.0, 0.0])
    st = math.sin(ALPHA * theta)
    ct = math.cos(ALPHA * theta)
    value = r ** ALPHA * st
    radial = ALPHA * r ** (ALPHA - 1.0) * st
    tangential = ALPHA * r ** (ALPHA - 1.0) * ct
    ex = x / r
    ey = y / r
    # e_theta=(-sin(theta), cos(theta)); this avoids a branch derivative.
    grad = np.array([radial * ex - tangential * ey,
                     radial * ey + tangential * ex])
    return value, grad


def exact_value_gradient(x, y):
    f, grad_f = singular_value_gradient(x, y)
    value, gradient = f + x * x - y * y, grad_f + np.array([2.0 * x, -2.0 * y])
    if EXTRA_MODE_AMPLITUDE:
        # F2 has exponent 4/3 and angular factor 4/3.
        beta, _, theta = 4.0 / 3.0, None, polar(x, y)[1]
        r = math.hypot(x, y)
        if r:
            st, ct = math.sin(beta * theta), math.cos(beta * theta)
            f2 = r ** beta * st
            radial = beta * r ** (beta - 1.0) * st
            tangential = beta * r ** (beta - 1.0) * ct
            ex, ey = x / r, y / r
            grad2 = np.array([radial * ex - tangential * ey,
                              radial * ey + tangential * ex])
        else:
            f2, grad2 = 0.0, np.zeros(2)
        value += EXTRA_MODE_AMPLITUDE * f2
        gradient += EXTRA_MODE_AMPLITUDE * grad2
    return value, gradient


def cubic_bspline(q):
    """Cubic B-spline kernel used by the RK approximation, q=distance/a."""
    if q < 0.0 or q > 1.0:
        return 0.0, 0.0
    if q <= 0.5:
        return 2.0 / 3.0 - 4.0 * q * q + 4.0 * q ** 3, -8.0 * q + 12.0 * q * q
    return 4.0 / 3.0 * (1.0 - q) ** 3, -4.0 * (1.0 - q) ** 2


class RKBasis:
    def __init__(self, nodes, h):
        self.nodes = np.asarray(nodes, dtype=float)
        self.h = h
        self.a = SUPPORT_FACTOR * h

    def evaluate(self, x, y):
        """Return nonzero RK shape indices, values, and x/y gradients."""
        dx = x - self.nodes[:, 0]
        dy = y - self.nodes[:, 1]
        distance = np.hypot(dx, dy)
        active = np.flatnonzero(distance <= self.a * (1.0 + 1e-12))
        if len(active) < 3:
            raise np.linalg.LinAlgError(f"insufficient RK support at ({x:g},{y:g}); only {len(active)} nodes")
        dxa, dya, ra = dx[active], dy[active], distance[active]
        q = ra / self.a
        weights = np.array([cubic_bspline(float(z))[0] for z in q])
        dwdq = np.array([cubic_bspline(float(z))[1] for z in q])
        safe_r = np.where(ra > 1e-14, ra, 1.0)
        grad_w = np.column_stack((dwdq * dxa / (self.a * safe_r),
                                  dwdq * dya / (self.a * safe_r)))
        grad_w[ra <= 1e-14] = 0.0
        H = np.column_stack((np.ones(len(active)), dxa, dya))
        dH_x = np.tile(np.array([0.0, 1.0, 0.0]), (len(active), 1))
        dH_y = np.tile(np.array([0.0, 0.0, 1.0]), (len(active), 1))
        M = (H.T * weights) @ H
        invM = np.linalg.inv(M)
        values = np.empty(len(active))
        gradients = np.empty((len(active), 2))
        qrow = np.array([1.0, 0.0, 0.0])
        for direction, dH, out_col in ((0, dH_x, 0), (1, dH_y, 1)):
            dM = (dH.T * weights) @ H + (H.T * weights) @ dH
            dM += (H.T * grad_w[:, direction]) @ H
            dInv = -invM @ dM @ invM
            for local in range(len(active)):
                b = H[local]
                db = dH[local]
                values[local] = qrow @ invM @ b * weights[local]
                gradients[local, out_col] = (
                    qrow @ dInv @ b * weights[local]
                    + qrow @ invM @ db * weights[local]
                    + qrow @ invM @ b * grad_w[local, direction]
                )
        return active, values, gradients


def make_nodes(n):
    h = 2.0 / n
    values = np.linspace(-1.0, 1.0, n + 1)
    nodes = [(x, y) for y in values for x in values
             if not (x > 1e-12 and y < -1e-12)]
    return np.asarray(nodes), h


def gauss_rule(order):
    points, weights = np.polynomial.legendre.leggauss(order)
    return points, weights


def domain_cells(n):
    h = 2.0 / n
    for j in range(n):
        for i in range(n):
            x0, y0 = -1.0 + i * h, -1.0 + j * h
            if x0 >= 0.0 and y0 < 0.0:
                continue
            yield x0, y0, h


def boundary_segments(n):
    h = 2.0 / n
    # (kind, x0, y0, dx, dy), each segment has length h.
    segments = []
    for j in range(n // 2):
        y = -1.0 + j * h
        segments.append(("g", 0.0, y, 0.0, h))             # x=0, lower notch
    for i in range(n // 2):
        x = i * h
        segments.append(("g", x, 0.0, h, 0.0))             # y=0, right notch
    for i in range(n // 2):
        x = -1.0 + i * h
        segments.append(("h_bottom", x, -1.0, h, 0.0))
    for j in range(n // 2, n):
        y = -1.0 + j * h
        segments.append(("h_right", 1.0, y, 0.0, h))
    for i in range(n):
        x = -1.0 + i * h
        segments.append(("h_top", x, 1.0, h, 0.0))
    for j in range(n):
        y = -1.0 + j * h
        segments.append(("h_left", -1.0, y, 0.0, h))
    return segments


def boundary_data(kind, x, y):
    if kind == "h_bottom":
        normal = np.array([0.0, -1.0])
    elif kind == "h_right":
        normal = np.array([1.0, 0.0])
    elif kind == "h_top":
        normal = np.array([0.0, 1.0])
    elif kind == "h_left":
        normal = np.array([-1.0, 0.0])
    else:
        normal = None
    value, gradient = exact_value_gradient(x, y)
    if normal is None:
        return value, None
    return value, float(gradient @ normal)


def add_outer(A, ix, iy, value):
    A[np.ix_(ix, iy)] += value


def assemble(n, domain_order=4, boundary_order=8):
    nodes, h = make_nodes(n)
    basis = RKBasis(nodes, h)
    size = len(nodes)
    Kbar = np.zeros((size, size))
    fbar = np.zeros(size)
    Kcouple = np.zeros(size)
    Khat = 0.0
    fhat = 0.0
    points, weights = gauss_rule(domain_order)
    for x0, y0, cell_h in domain_cells(n):
        for px, wx in zip(points, weights):
            x = x0 + 0.5 * cell_h * (px + 1.0)
            for py, wy in zip(points, weights):
                y = y0 + 0.5 * cell_h * (py + 1.0)
                active, _, gradients = basis.evaluate(x, y)
                jac = 0.25 * cell_h * cell_h * wx * wy
                Kbar[np.ix_(active, active)] += jac * (gradients @ gradients.T)

    bpoints, bweights = gauss_rule(boundary_order)
    beta = BETA_FACTOR / h
    for kind, x0, y0, dx, dy in boundary_segments(n):
        length = math.hypot(dx, dy)
        tangent_jac = 0.5 * length
        for p, w in zip(bpoints, bweights):
            x, y = x0 + 0.5 * (p + 1.0) * dx, y0 + 0.5 * (p + 1.0) * dy
            active, values, gradients = basis.evaluate(x, y)
            if kind == "g":
                # Outward normals point into the removed quadrant on notch edges.
                # On the removed-quadrant boundaries the outward normals point
                # into the omitted quadrant: +x on x=0 and -y on y=0.
                normal = np.array([1.0, 0.0]) if abs(dx) < 1e-15 else np.array([0.0, -1.0])
                g, _ = boundary_data(kind, x, y)
                dn = gradients @ normal
                jac = tangent_jac * w
                Kbar[np.ix_(active, active)] += jac * (
                    -np.outer(values, dn) - np.outer(dn, values)
                    + beta * np.outer(values, values))
                fbar[active] += jac * (beta * values - dn) * g
            else:
                g, flux = boundary_data(kind, x, y)
                value_f, grad_f = singular_value_gradient(x, y)
                # ISBFM: all terms involving F are boundary-only.
                normal = {
                    "h_bottom": np.array([0.0, -1.0]),
                    "h_right": np.array([1.0, 0.0]),
                    "h_top": np.array([0.0, 1.0]),
                    "h_left": np.array([-1.0, 0.0]),
                }[kind]
                dn_f = float(grad_f @ normal)
                jac = tangent_jac * w
                Kcouple[active] += jac * values * dn_f
                Khat += jac * value_f * dn_f
                fbar[active] += jac * values * flux
                fhat += jac * value_f * flux

    K = np.zeros((size + 1, size + 1))
    K[:size, :size] = Kbar
    K[:size, size] = Kcouple
    K[size, :size] = Kcouple
    K[size, size] = Khat
    rhs = np.r_[fbar, fhat]
    coefficients = np.linalg.solve(K, rhs)
    return nodes, basis, coefficients, h, np.linalg.cond(K)


def errors(n, basis, coefficients, error_order=8):
    points, weights = gauss_rule(error_order)
    e0 = h0 = h1 = 0.0
    for x0, y0, cell_h in domain_cells(n):
        for px, wx in zip(points, weights):
            x = x0 + 0.5 * cell_h * (px + 1.0)
            for py, wy in zip(points, weights):
                y = y0 + 0.5 * cell_h * (py + 1.0)
                active, values, gradients = basis.evaluate(x, y)
                uh = float(values @ coefficients[active]) + coefficients[-1] * singular_value_gradient(x, y)[0]
                grad_uh = gradients.T @ coefficients[active] + coefficients[-1] * singular_value_gradient(x, y)[1]
                exact, grad_exact = exact_value_gradient(x, y)
                jac = 0.25 * cell_h * cell_h * wx * wy
                e0 += jac * (uh - exact) ** 2
                h0 += jac * exact ** 2
                h1 += jac * ((uh - exact) ** 2 + np.dot(grad_uh - grad_exact, grad_uh - grad_exact))
    # Also provide the normalized gradient seminorm for diagnosis.
    return math.sqrt(e0 / h0), math.sqrt(h1 / (h0 + grad_norm_sq(n, error_order))), math.sqrt(grad_error_sq(n, basis, coefficients, error_order) / grad_norm_sq(n, error_order))


def grad_norm_sq(n, order):
    points, weights = gauss_rule(order)
    total = 0.0
    for x0, y0, cell_h in domain_cells(n):
        for px, wx in zip(points, weights):
            x = x0 + 0.5 * cell_h * (px + 1.0)
            for py, wy in zip(points, weights):
                y = y0 + 0.5 * cell_h * (py + 1.0)
                total += 0.25 * cell_h * cell_h * wx * wy * np.dot(exact_value_gradient(x, y)[1], exact_value_gradient(x, y)[1])
    return total


def grad_error_sq(n, basis, coefficients, order):
    points, weights = gauss_rule(order)
    total = 0.0
    for x0, y0, cell_h in domain_cells(n):
        for px, wx in zip(points, weights):
            x = x0 + 0.5 * cell_h * (px + 1.0)
            for py, wy in zip(points, weights):
                y = y0 + 0.5 * cell_h * (py + 1.0)
                active, _, gradients = basis.evaluate(x, y)
                grad_uh = gradients.T @ coefficients[active] + coefficients[-1] * singular_value_gradient(x, y)[1]
                diff = grad_uh - exact_value_gradient(x, y)[1]
                total += 0.25 * cell_h * cell_h * wx * wy * np.dot(diff, diff)
    return total


def save_fields(n, basis, coefficients, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    grid = np.linspace(-1.0, 1.0, 201)
    X, Y = np.meshgrid(grid, grid)
    uh = np.full_like(X, np.nan)
    exact = np.full_like(X, np.nan)
    for j, y in enumerate(grid):
        for i, x in enumerate(grid):
            if x > 0 and y < 0:
                continue
            active, values, _ = basis.evaluate(x, y)
            uh[j, i] = values @ coefficients[active] + coefficients[-1] * singular_value_gradient(x, y)[0]
            exact[j, i] = exact_value_gradient(x, y)[0]
    error = abs(uh - exact)
    output.parent.mkdir(parents=True, exist_ok=True)
    for data, title, name in ((uh, "RK + ISBFM solution", "solution"), (exact, "Manufactured exact solution", "exact"), (error, "Pointwise absolute error", "error")):
        fig, ax = plt.subplots(figsize=(5.2, 4.5))
        im = ax.pcolormesh(X, Y, data, shading="auto")
        ax.set_aspect("equal"); ax.set_title(title); fig.colorbar(im, ax=ax)
        fig.tight_layout(); fig.savefig(output.parent / f"{output.stem}_{name}.png", dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", nargs="+", type=int, default=list(DEFAULT_LEVELS))
    parser.add_argument("--output-dir", type=Path, default=Path("results/processed"))
    args = parser.parse_args()
    results = []
    for n in args.levels:
        print(f"Solving n={n} ...", flush=True)
        nodes, basis, coeff, h, condition = assemble(n)
        e0, e1, egrad = errors(n, basis, coeff)
        row = {"n": n, "h": h, "nodes": len(nodes), "coefficient_F1": coeff[-1], "coefficient_error": abs(coeff[-1] - 1.0), "normalized_L2": e0, "normalized_H1": e1, "normalized_H1_seminorm": egrad, "condition_number": condition}
        results.append(row)
        print(row)
        if n == max(args.levels):
            try:
                save_fields(n, basis, coeff, args.output_dir / "poisson_rk_isbfm")
            except ImportError:
                print("Plot skipped (matplotlib unavailable)")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "poisson_rk_isbfm_convergence.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0])); writer.writeheader(); writer.writerows(results)
    print("Convergence CSV:", args.output_dir / "poisson_rk_isbfm_convergence.csv")


if __name__ == "__main__":
    main()
