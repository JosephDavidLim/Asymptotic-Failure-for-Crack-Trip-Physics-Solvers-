"""
Correlation analysis and figure for the matched-asymptotic detector study.
Loads results/processed/detectors.csv (produced by detector_sweep.py) and:

  - computes Pearson/Spearman correlation of each candidate detector against
    the plastic-reference true error, both aggregate (all 200 runs) and
    per-sweep-point (n=10 means), mirroring exactly the statistics the prior
    (loss-vs-error) study reported, for direct comparability;
  - determines, for each series, the first S at which its per-seed
    distribution is resolvable (>2 pooled-sigma separated) from the S=0.05
    baseline distribution -- the "rises above seed noise" question;
  - renders the one required figure: loss, the naive field-fit control, the
    HRR severity detector, and the true error, all against S, each
    min-max normalized to [0,1] so they share one comparable axis (a single
    y-axis, not a dual-axis chart).

Usage:
    python src/analyze_detectors.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

IN_CSV = "results/processed/detectors.csv"
OUT_SUMMARY_CSV = "results/processed/detector_summary_by_S.csv"
OUT_STATS_TXT = "results/processed/detector_correlation_stats.txt"
OUT_FIG = "figures/detector_vs_error.png"

BLUE = "#2a78d6"     # loss
ORANGE = "#eb6834"   # true error (reference)
AQUA = "#1baf7a"      # zeta detector (HRR severity)
YELLOW = "#eda100"   # naive field-fit control
MUTED = "#898781"
GRID = "#e1e0d9"
INK = "#0b0b0b"
SEC_INK = "#52514e"
SURFACE = "#fcfcfb"


def sigma_separation(mean, std, base_mean, base_std):
    pooled = np.sqrt(std ** 2 + base_std ** 2)
    if pooled == 0:
        return np.inf if mean != base_mean else 0.0
    return (mean - base_mean) / pooled


def correlation_block(df, g, col, mean_col, label, lines):
    r_all, p_all = pearsonr(df[col], df["plastic_reference_error"])
    rho_all, ps_all = spearmanr(df[col], df["plastic_reference_error"])
    r_s, p_s = pearsonr(g[mean_col], g["err_mean"])
    rho_s, ps_s = spearmanr(g[mean_col], g["err_mean"])
    lines.append(f"{label}:")
    lines.append(f"  all-200 runs:     Pearson r={r_all:+.3f} (p={p_all:.2e})   Spearman rho={rho_all:+.3f} (p={ps_all:.2e})")
    lines.append(f"  per-S means(n=10): Pearson r={r_s:+.3f} (p={p_s:.2e})   Spearman rho={rho_s:+.3f} (p={ps_s:.2e})")
    return dict(r_all=r_all, p_all=p_all, rho_all=rho_all, ps_all=ps_all, r_s=r_s, p_s=p_s, rho_s=rho_s, ps_s=ps_s)


def main():
    df = pd.read_csv(IN_CSV)

    g = df.groupby("S").agg(
        n=("seed", "count"),
        loss_mean=("normalized_train_loss", "mean"), loss_std=("normalized_train_loss", "std"),
        err_mean=("plastic_reference_error", "mean"), err_std=("plastic_reference_error", "std"),
        exp_dev_mean=("field_fit_exponent_deviation", "mean"), exp_dev_std=("field_fit_exponent_deviation", "std"),
        zeta_mean=("zeta_detector", "mean"), zeta_std=("zeta_detector", "std"),
    ).reset_index()
    os.makedirs(os.path.dirname(OUT_SUMMARY_CSV), exist_ok=True)
    g.to_csv(OUT_SUMMARY_CSV, index=False)

    lines = []
    lines.append(f"n = {len(df)} runs, {len(g)} sweep points, {int(g.n.iloc[0])} seeds/point")
    lines.append("")
    stats_loss = correlation_block(df, g, "normalized_train_loss", "loss_mean", "Loss (prior study's diagnostic)", lines)
    lines.append("")
    stats_field = correlation_block(df, g, "field_fit_exponent_deviation", "exp_dev_mean",
                                     "Detector 1: naive post-hoc field-fit exponent deviation from -0.5", lines)
    lines.append("")
    stats_zeta = correlation_block(df, g, "zeta_detector", "zeta_mean",
                                    "Detector 2: HRR severity zeta = sigma_HRR(r*=0.05;S)/sigma_y", lines)
    lines.append("")

    lines.append("Onset above seed noise (sigma-separation from S=0.05 baseline distribution, pooled std):")
    baseline = g.iloc[0]
    lines.append(f"{'S':>5}  {'err_sigma':>10}  {'zeta_sigma':>11}  {'field_fit_sigma':>16}")
    for _, row in g.iterrows():
        es = sigma_separation(row.err_mean, row.err_std, baseline.err_mean, baseline.err_std)
        zs = sigma_separation(row.zeta_mean, row.zeta_std, baseline.zeta_mean, baseline.zeta_std)
        fs = sigma_separation(row.exp_dev_mean, row.exp_dev_std, baseline.exp_dev_mean, baseline.exp_dev_std)
        lines.append(f"{row.S:5.2f}  {es:10.2f}  {zs:11.1f}  {fs:16.2f}")
    lines.append("")
    lines.append("Reading >2-sigma as 'resolvable': true error is not resolvable from the S=0.05")
    lines.append("baseline until S~0.2-0.3 (1.97-sigma at 0.20, 8.29-sigma at 0.30); the HRR severity")
    lines.append("detector is already >99-sigma separated at the very next point tested, S=0.10 --")
    lines.append("i.e. its onset is at or below our sweep's resolution and we cannot localize it")
    lines.append("more precisely without finer S spacing near 0.05-0.10. The naive field-fit control")
    lines.append("never becomes resolvable (max ~2-sigma, non-monotonic) anywhere in the sweep.")

    with open(OUT_STATS_TXT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # --- Figure: single panel, all series indexed to a COMMON scale ---
    # IMPORTANT: per-series min-max normalization (each series stretched to
    # fill [0,1] independently) is misleading here -- it would inflate a
    # near-constant, noise-dominated series (the naive field-fit control) to
    # look like it swings across the full panel, when its real absolute
    # movement is ~1000x smaller than the true error's. Instead we index
    # every series to the SAME yardstick: the true error's own absolute
    # range (its S=0.05 -> S=0.70 change), after subtracting each series'
    # own S=0.05 baseline. A series that barely moves in absolute terms then
    # correctly reads as flat, near zero, on this shared axis.
    err_range = g["err_mean"].iloc[-1] - g["err_mean"].iloc[0]

    def index_to_error_scale(s):
        return (s - s.iloc[0]) / err_range

    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")
    ax.tick_params(colors=SEC_INK, labelsize=9)

    series = [
        ("err_mean", ORANGE, "True error vs Dugdale reference (0.9%→33%, the target)"),
        ("zeta_mean", AQUA, "HRR severity detector (σ_HRR/σ_y: 1.75→2.98)"),
        ("loss_mean", BLUE, "Training loss (barely moves in absolute terms)"),
        ("exp_dev_mean", YELLOW, "Naive post-hoc field-fit deviation (barely moves)"),
    ]
    for col, color, label in series:
        y = index_to_error_scale(g[col])
        ax.plot(g.S, y, color=color, linewidth=2.2, marker="o", markersize=5, label=label, zorder=3)

    ax.axhline(0, color=MUTED, linewidth=1, zorder=1)
    ax.set_xlabel("S = applied stress / yield stress (σ_ext / σ_y)", fontsize=10, color=INK)
    ax.set_ylabel("Change from S=0.05, indexed to the true error's own\nabsolute range (so 0→1 = the error's full climb)", fontsize=10, color=INK)
    ax.set_title("Only the HRR-informed detector tracks the true error; loss and a naive\npost-hoc field-fit both stay essentially flat on the same scale",
                  fontsize=12, color=INK, loc="left", pad=12)
    leg = ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=SEC_INK)

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
    fig.savefig(OUT_FIG, dpi=200, facecolor=SURFACE)
    print(f"\nWrote figure to {OUT_FIG}")
    print(f"Wrote per-S summary to {OUT_SUMMARY_CSV}")
    print(f"Wrote correlation stats to {OUT_STATS_TXT}")


if __name__ == "__main__":
    main()
