# Asymptotic Failure for Crack-Tip Physics Solvers

Does the training loss of an **enriched, elastic** physics-informed crack
solver signal when its own near-tip asymptotic assumption (LEFM, `sigma ~
r^-1/2`) has become invalid because the crack-tip plastic zone is no longer
negligible?

**Finding: no.** See [`report/REPORT.md`](report/REPORT.md) for the full
writeup. Short version: across a load sweep from deeply-elastic to
near-limit-load, the true error in the solver's reported `K_I` grows ~37x
while its own training loss stays flat (if anything trends slightly down).
The solver converges cleanly and reports a confident, wrong answer, with
nothing in its own diagnostics giving that away. This is a silent failure,
and it's the condition under which the proposed matched-asymptotic
(elastic + HRR) detector study is justified to proceed.

**Follow-on: can a matched-asymptotic overlap signal detect this where the
loss can't?** See
[`report/REPORT_matched_asymptotics.md`](report/REPORT_matched_asymptotics.md).
Short version: the cheapest version of the idea (post-hoc HRR-form fit to
pihnn's already-converged fields, no retraining) does **not** work, for a
provable structural reason -- it's as blind as the loss. A construction that
actually injects the missing physical input (yield stress, hardening
exponent, the HRR field, calibrated via J-matching against pihnn's own
`K_I`) does work: Spearman ρ=+0.95 with the true error across 200 runs (vs
−0.41 for loss), and it becomes statistically resolvable from seed noise
*before* the true error itself does (S≈0.10 vs S≈0.2-0.3). Both the
successful and two rejected constructions are documented, since the reasons
the naive ones fail are as informative as the one that works.

## What's here

- `src/pihnn_wrapper.py` -- wraps (does not modify) the
  [pihnn](https://github.com/teocala/pihnn) library's straight-crack,
  Williams-enriched example, parametrizing the one thing this study sweeps:
  applied remote stress.
- `src/reference.py` -- the independent plastic-regime reference (Dugdale
  strip-yield model, cross-checked against a first-order Irwin correction).
  See the "reference problem" section in the report for why this was chosen
  over the GE/EPRI handbook method or a from-scratch FEM.
- `src/sweep.py` -- runs the load sweep (10 load-over-yield ratios x 20 seeds
  = 200 training runs). Idempotent: re-running skips any `(S, seed)` that
  already has output, so raw per-run data is never overwritten.
- `src/aggregate.py` -- scans `results/raw/*/summary.json` into
  `results/processed/all_runs.csv`. Pure function of the raw data; safe to
  re-run any time (e.g. after editing `reference.py`) without re-training.
- `src/analyze.py` -- per-sweep-point statistics, correlation diagnostics
  (loss vs error), and the one required figure
  (`figures/loss_vs_error.png`).
- `configs/sweep_config.json` -- sweep points, seed count, epoch count, and
  the calibration numbers that justified the epoch count.
- `results/raw/S{value}_seed{seed}/` -- one directory per run: `config` is
  implicit in the folder name, `loss.dat` (full training/test loss
  trajectory), `model.dict` (trained weights), `summary.json` (extracted
  K_I, K_II, loss, convergence diagnostics). Every number in the report
  traces back to one of these.
- `results/processed/` -- derived, regenerable: `all_runs.csv` (one row per
  run), `summary_by_S.csv` (aggregated per sweep point),
  `correlation_stats.txt`.
- `report/REPORT.md` -- the actual writeup: method, reference justification
  and limitations, convergence checks, results, interpretation.
- `src/hrr_reference.py` -- the HRR (Hutchinson-Rice-Rosengren) inner field
  and the matched-asymptotic detector, `zeta`. Cites the source of its
  tabulated constants (I_n, angular function) and documents two rejected
  constructions before landing on the one used.
- `src/field_fit.py` -- Detector 1, the naive post-hoc control: reloads a
  trained model (no retraining) and fits a free power-law exponent to its
  own near-tip field over a fixed radius window.
- `src/detector_sweep.py` -- runs both detectors over all 200 existing raw
  runs from the prior sweep (post-hoc only, ~1 second total, no training).
  Writes `results/processed/detectors.csv`.
- `src/analyze_detectors.py` -- correlation statistics (Pearson/Spearman,
  aggregate and per-sweep-point) for both detectors against the true error,
  the seed-noise onset analysis, and the required figure
  (`figures/detector_vs_error.png`).
- `report/REPORT_matched_asymptotics.md` -- the follow-on study: route
  justification, the HRR-constant citation, two rejected detector
  constructions and why they failed, results, limitations.

## Reproducing

Requires Python >= 3.10 (pihnn's constraint) and no GPU (everything here runs
CPU-only; the full 200-run sweep takes ~65 minutes on a laptop CPU).

```bash
python3.1x -m venv .venv
source .venv/bin/activate
pip install torch numpy matplotlib tqdm scipy pandas
pip install "git+https://github.com/teocala/pihnn.git"

python src/sweep.py configs/sweep_config.json      # ~65 min, 200 runs
python src/aggregate.py                            # seconds
python src/analyze.py                              # seconds, writes the figure
```

`pihnn`'s own `pyproject.toml` does not pin its dependencies (torch, numpy,
matplotlib, tqdm), which is why they're installed separately above rather than
pulled in automatically.

## On pihnn

pihnn is LGPL-2.1 licensed. This repository depends on it as an ordinary
(unmodified) library dependency -- installed via pip/git, imported, never
patched in place. Where its behavior needed adjusting for batch runs (e.g.
silencing its per-epoch progress bar across 200 sequential trainings), the
change is a runtime monkeypatch of the *imported reference* inside
`pihnn_wrapper.py`, not an edit to any file inside the package.

## Why this experiment is designed the way it is

pihnn's crack solver is purely elastic: nothing about a yield stress or a
plastic zone appears anywhere in its PDE residual or loss function. Directly
retraining it at a range of applied loads (rather than post-hoc rescaling a
single elastic solution) matters for two reasons: it's the literal
experiment the task specifies, and it exercises real seed-to-seed training
variance at every point rather than assuming linear-elasticity scaling holds
exactly (which we then also independently verify: raw loss scales with load
magnitude as `~S^2`, a scaling artifact of the loss being an unnormalized MSE
residual, unrelated to physical validity -- hence all "does loss track error"
comparisons in the report use loss normalized by `S^2`, not raw loss).

Two independent convergence checks run at every single one of the 200 runs,
not just spot-checked, to separate "the model is undertrained" from "the
model's assumption is invalid" (see the report for why this distinction
mattered for a prior conclusion on this project):

1. Numerical: loss-trajectory plateau ratio (last 5% of epochs vs the
   preceding 5%).
2. Physical: since pihnn is solving a linear-elastic BVP with a **known
   closed-form solution** at every load level (`K_el(S) = S * 1.852` for
   this geometry), we check the network's reported `K_I` against that exact
   target directly -- this catches undertraining that the plastic-regime
   reference error alone could not distinguish from genuine model invalidity.
