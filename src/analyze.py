"""
Loads results/processed/all_runs.csv, computes the per-S summary statistics
(seed spread, not best-run cherry-picking), correlation diagnostics between
loss and error, and renders the single required figure:
loss and error against the sweep parameter S = sigma_ext / sigma_y.

Usage:
    python src/analyze.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(__file__))

IN_CSV = "results/processed/all_runs.csv"
OUT_SUMMARY_CSV = "results/processed/summary_by_S.csv"
OUT_STATS_TXT = "results/processed/correlation_stats.txt"
OUT_FIG = "figures/loss_vs_error.png"

# Palette (see dataviz skill reference/palette.md)
BLUE = "#2a78d6"
ORANGE = "#eb6834"
MUTED = "#898781"
GRID = "#e1e0d9"
INK = "#0b0b0b"
SEC_INK = "#52514e"
SURFACE = "#fcfcfb"

REFERENCE_UNCERTAIN_S = 0.6  # reference_disagreement crosses ~20% here
NET_SECTION_YIELD_S = 0.75


def main():
    df = pd.read_csv(IN_CSV)

    g = df.groupby("S").agg(
        n_seeds=("seed", "count"),
        loss_mean=("normalized_train_loss", "mean"),
        loss_std=("normalized_train_loss", "std"),
        loss_median=("normalized_train_loss", "median"),
        lefm_err_mean=("lefm_verification_error", "mean"),
        lefm_err_std=("lefm_verification_error", "std"),
        lefm_err_max=("lefm_verification_error", "max"),
        plastic_err_mean=("plastic_reference_error", "mean"),
        plastic_err_std=("plastic_reference_error", "std"),
        plastic_err_min=("plastic_reference_error", "min"),
        plastic_err_max=("plastic_reference_error", "max"),
        K_I_mean=("K_I", "mean"),
        K_I_std=("K_I", "std"),
        K_ref_dugdale=("K_ref_dugdale", "first"),
        K_ref_irwin=("K_ref_irwin", "first"),
        reference_disagreement=("reference_disagreement", "first"),
        dugdale_zone_ratio=("dugdale_zone_ratio", "first"),
    ).reset_index()
    os.makedirs(os.path.dirname(OUT_SUMMARY_CSV), exist_ok=True)
    g.to_csv(OUT_SUMMARY_CSV, index=False)

    # --- Correlation diagnostics: does loss track error? ---
    lines = []
    r_all, p_all = pearsonr(df["normalized_train_loss"], df["plastic_reference_error"])
    rho_all, ps_all = spearmanr(df["normalized_train_loss"], df["plastic_reference_error"])
    lines.append(f"All {len(df)} runs, loss vs plastic-reference error:")
    lines.append(f"  Pearson r = {r_all:+.3f} (p={p_all:.2e})")
    lines.append(f"  Spearman rho = {rho_all:+.3f} (p={ps_all:.2e})")
    lines.append("")

    r_s, p_s = pearsonr(g["loss_mean"], g["plastic_err_mean"])
    rho_s, ps_s = spearmanr(g["loss_mean"], g["plastic_err_mean"])
    lines.append(f"Per-S means (n={len(g)} sweep points), loss vs plastic-reference error:")
    lines.append(f"  Pearson r = {r_s:+.3f} (p={p_s:.2e})")
    lines.append(f"  Spearman rho = {rho_s:+.3f} (p={ps_s:.2e})")
    lines.append("")

    lines.append("Within-S (seed-noise only, controlling for S) loss vs error, n=20 each:")
    within_rs = []
    for S, sub in df.groupby("S"):
        r, p = pearsonr(sub["normalized_train_loss"], sub["plastic_reference_error"])
        within_rs.append(r)
        lines.append(f"  S={S:.2f}: r={r:+.3f} (p={p:.2f})")
    lines.append(f"  mean within-S r = {np.mean(within_rs):+.3f}")
    lines.append("")

    lines.append(f"loss range across sweep: {df.normalized_train_loss.min():.2e} - {df.normalized_train_loss.max():.2e}  "
                  f"(x{df.normalized_train_loss.max()/df.normalized_train_loss.min():.1f})")
    lines.append(f"plastic_reference_error range across sweep: {df.plastic_reference_error.min():.1%} - {df.plastic_reference_error.max():.1%}  "
                  f"(x{df.plastic_reference_error.max()/df.plastic_reference_error.min():.1f})")
    lines.append(f"lefm_verification_error (undertraining control) range: {df.lefm_verification_error.min():.2%} - {df.lefm_verification_error.max():.2%}")

    with open(OUT_STATS_TXT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # --- Figure ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 8.5), sharex=True, facecolor=SURFACE)
    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#c3c2b7")
        ax.tick_params(colors=SEC_INK, labelsize=9)
        ax.axvspan(REFERENCE_UNCERTAIN_S, 0.78, color=MUTED, alpha=0.10, zorder=0)
        ax.axvline(NET_SECTION_YIELD_S, color=MUTED, linestyle=":", linewidth=1.2, zorder=1)

    rng = np.random.default_rng(0)
    jitter = lambda s: s + rng.uniform(-0.006, 0.006, size=len(df[df.S == s]))

    # Panel 1: normalized training loss
    for S in sorted(df.S.unique()):
        sub = df[df.S == S]
        ax1.scatter(jitter(S), sub["normalized_train_loss"], s=14, color=BLUE, alpha=0.30,
                    linewidths=0, zorder=2)
    ax1.plot(g.S, g.loss_mean, color=BLUE, linewidth=2, marker="o", markersize=5, zorder=3,
              label="mean over 20 seeds")
    ax1.fill_between(g.S, g.loss_mean - g.loss_std, g.loss_mean + g.loss_std, color=BLUE, alpha=0.15, zorder=1)
    ax1.set_yscale("log")
    ax1.set_ylabel("Normalized final training loss\n(raw loss / S²)", fontsize=10, color=INK)
    ax1.set_title("Loss does not track error: enriched elastic pihnn solver, straight-crack panel",
                   fontsize=12, color=INK, loc="left", pad=12)
    ax1.legend(frameon=False, fontsize=9, loc="upper right", labelcolor=SEC_INK)
    ax1.text(0.02, 0.04, "flat / no trend with S", transform=ax1.transAxes, fontsize=9,
              color=SEC_INK, style="italic")

    # Panel 2: plastic-reference error
    for S in sorted(df.S.unique()):
        sub = df[df.S == S]
        ax2.scatter(jitter(S), 100 * sub["plastic_reference_error"], s=14, color=ORANGE, alpha=0.30,
                     linewidths=0, zorder=2)
    ax2.plot(g.S, 100 * g.plastic_err_mean, color=ORANGE, linewidth=2, marker="o", markersize=5, zorder=3,
              label="mean over 20 seeds")
    ax2.fill_between(g.S, 100 * (g.plastic_err_mean - g.plastic_err_std),
                      100 * (g.plastic_err_mean + g.plastic_err_std), color=ORANGE, alpha=0.15, zorder=1)
    ax2.set_ylabel("True error in reported K_I\nvs Dugdale plastic reference (%)", fontsize=10, color=INK)
    ax2.set_xlabel("S = applied stress / yield stress (σ_ext / σ_y)", fontsize=10, color=INK)
    ax2.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=SEC_INK)

    ax2.annotate("reference model\nuncertain here", xy=(0.68, 25), fontsize=8, color=SEC_INK, ha="left")
    ax2.annotate("net-section\nyield", xy=(NET_SECTION_YIELD_S, 2), xytext=(NET_SECTION_YIELD_S + 0.005, 2),
                 fontsize=8, color=SEC_INK, ha="left")

    ax1.set_xlim(0.0, 0.78)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
    fig.savefig(OUT_FIG, dpi=200, facecolor=SURFACE)
    print(f"\nWrote figure to {OUT_FIG}")
    print(f"Wrote per-S summary to {OUT_SUMMARY_CSV}")
    print(f"Wrote correlation stats to {OUT_STATS_TXT}")


if __name__ == "__main__":
    main()
