# Quantitative claim audit

Status: **incomplete because the paper inputs are missing**.

Neither `main.tex` nor `refs.bib` is present in this worktree, any local Git
history/ref, the parent project directory, or connected Google Drive search.
Consequently, this document cannot yet be the requested exhaustive audit of
every paper claim or bibliography entry. It is a source-first audit of the
experiment repository and of quantitative claims found in the two study
reports. It must be reconciled line-by-line against the actual paper once the
two missing files (and any included `.tex` files) are supplied.

No paper or experiment source file was changed.

## Discrepancies and qualifications already established

Here, “paper/report value” means the value currently stated in one or both
study reports. Whether the absent paper repeats it is unknown.

| Claim | Paper/report value | Source value recomputed from underlying artifacts | Source | Verdict |
|---|---:|---:|---|---|
| Growth in true error | “37×”, described as growth across the sweep and as `0.9% -> 33%` | Endpoint **means** are 2.0931078% at S=0.05 and 32.8718054% at S=0.70: **15.7048×**. The **37.5533×** number is instead the global maximum divided by the global minimum across individual runs: 33.1754954% at S=0.70, seed 3 divided by 0.8834237% at S=0.10, seed 5. | `results/raw/S0.050_seed*/summary.json`, `results/raw/S0.100_seed005/summary.json`, `results/raw/S0.700_seed003/summary.json`; Dugdale formula in `src/reference.py` | **Discrepancy.** “37× growth” is not an endpoint ratio. Replace with 15.7× for endpoint means, or explicitly call 37.6× the full per-run max/min range and name its S/seed endpoints. |
| Low and high “endpoint” errors | 0.9% and 33% | S=0.05 mean/range: 2.093% (1.352%–3.077%); S=0.70 mean/range: 32.872% (32.664%–33.175%). The 0.9% minimum occurs at S=0.10, not the first sweep point. | Same raw summaries and `src/reference.py` | **Discrepancy.** 0.9%→33% cannot describe the S=0.05→0.70 endpoint means. |
| Full-sweep timing | `200 runs x ~18s/run ~= 66 minutes` | Sum of the 200 per-run training timers: 3,927.810 s = **65.464 min**; sweep log elapsed time: **3,936 s = 65.600 min**; mean per-run training time: **19.639 s**; median: 18.086 s. Also, 200×18 s is 60 min, not 66 min. | all `results/raw/*/summary.json`; final line of `results/logs/sweep_run.log` | **Mixed.** The ≈66-minute total matches the log, but the per-run mean and displayed arithmetic do not. |
| Calibration at 1,000/2,000/4,000 epochs | 2.45% / 1.20% / 1.15% LEFM error | No calibration run outputs, log, model, or loss trajectories are present. The values occur only as prose in `configs/sweep_config.json`. | `configs/sweep_config.json` | **Untraceable.** Configuration prose is not underlying evidence. Supply the calibration artifacts or remove/qualify the numbers. |
| Reproduction of published benchmark | K_I=1.841 versus 1.852, 0.59%, 1,000 epochs, S=1 | No benchmark run output or log is present. | Claim only in `report/REPORT.md` | **Untraceable from supplied data.** |
| Numerical convergence / clean plateau behavior | Reports and README imply clean convergence and two convergence checks on every run | The plateau ratio is recomputable and stored correctly, but ranges from **0.0241 to 62.3088**. Only 125/200 lie in [0.5, 2], 73/200 in [0.8, 1.2], and 52/200 in [0.9, 1.1]. There is no declared pass threshold. Separately, the physical LEFM verification error is 0.0197%–2.9275%, so the `<3%` physical-control claim is correct. | every `loss.dat` and `summary.json`; formula in `src/pihnn_wrapper.py` | **Wording change required.** Do not say the plateau diagnostic establishes clean convergence for every run. The physical control supports accurate K_I despite erratic last-block loss ratios. |
| Loss spread at every S | “roughly 2 orders of magnitude at every S” | Per-S max/min ratios range from **5.22× (0.718 orders)** at S=0.60 to **64.57× (1.810 orders)** at S=0.05. No S reaches 100×. | raw final losses divided by S² | **Overstated.** Say roughly one to two orders at most points, with only 5.2× at S=0.60. |
| Field-fit exponent range | “0.484–0.490 at every S” | Per-S means are **0.488136–0.489553**. Individual runs span **0.479452–0.493171**. | recomputed from all 200 `model.dict` files over `R_WINDOW`; matches `results/processed/detectors.csv` | **Ambiguous/inaccurate.** Use the mean range if discussing means, or the full individual range if discussing all runs. |
| Field-fit movement relative to HRR | range “about 1000× smaller” than movement toward the HRR exponent | Per-S-mean span is 0.00141693. Distance from the typical fitted exponent to 1/(8.93+1)=0.100705 is about 0.38818, a ratio of **274×**, not 1000×. | raw model recomputation; `src/field_fit.py`; `src/hrr_reference.py` | **Discrepancy.** The effect is still very small, but the stated factor is too large. |
| Field-fit detector direction | `field_fit_exponent_deviation = p_fit - 0.5`, with code comment “>0 => softer/HRR-like decay” | HRR has p=1/(n+1)=0.100705, which is **below** 0.5. Movement toward HRR therefore makes `p_fit - 0.5` more negative, not positive. | `src/field_fit.py`; `src/hrr_reference.py` | **Method-label discrepancy.** The computed correlations are numerically reproducible, but positive movement of this column is movement toward the elastic exponent, not toward HRR. Rename/re-sign or describe neutrally. |
| Post-hoc detector runtime | “seconds per run” in the matched-asymptotics report; “~1 second total” in README | Independent recomputation of all 200 field fits took **1.83 s total** in the audit environment (timing is machine-dependent). No original detector-run log is stored. | all 200 `model.dict` files; `src/field_fit.py` | **Report wording is wrong; exact original time untraceable.** “Seconds total” is supported by reproduction, not “seconds per run.” |
| Sigma separation definition | “pooled-σ separation” | Code uses Δmean / sqrt(s₁²+s₀²), not the conventional equal-n pooled SD sqrt((s₁²+s₀²)/2), and not the standard error of a difference in means. The printed values exactly match the implemented root-sum-square definition. | raw runs; `src/analyze_detectors.py:sigma_separation` | **Terminology change required.** Call it root-sum-square SD separation/effect-size heuristic and state the formula. Do not present values such as 99.7σ as a p-value or conventional significance statistic. |
| First true-error onset above 2σ | “S≈0.2–0.3”; 1.97σ at 0.20 and 8.29σ at 0.30 | Exact implemented values: **1.97354** at 0.20 and **8.28570** at 0.30. | raw runs and implemented formula | **Values match.** Under a strict `>2` rule the first tested point is S=0.30; S=0.20 is just below threshold. |
| Naive-field-fit maximum separation | Report says ≈0.27σ; generated stats-file prose says “max ~2-sigma” | Exact maximum is **0.2735** (rounds to 0.27), at S=0.40. | raw model recomputation; `results/processed/detector_correlation_stats.txt`; `src/analyze_detectors.py` | **Stored prose discrepancy.** The report is correct; the stats-file summary sentence is not. |
| Detector described as a “genuine two-region construction” | Abstract wording in matched-asymptotics report | Detector 2 is a closed-form HRR severity evaluated post hoc at a fixed radius; the same report later calls it a practical stand-in and says neither detector is a genuine two-region trained solver. | `src/hrr_reference.py`; `report/REPORT_matched_asymptotics.md` | **Internal wording contradiction.** Use “HRR-informed post-hoc severity indicator,” not “genuine two-region construction.” |
| Reference-uncertain threshold | Report/plot use S≥0.60; `src/reference.py` prose says S≥0.65 | Recomputed Dugdale–Irwin disagreement is **20.8616% at S=0.60** and 24.9888% at S=0.65. | `src/reference.py`; raw S values | **Source inconsistency.** S≥0.60 is defensible if the intended threshold is about 20%; make all files agree. |
| Figure legend’s “true error (0.9%→33%)” while normalizing endpoint means | 0.9%→33% | Figure code defines its scale with the change in per-S means: **2.093%→32.872%**. | `src/analyze_detectors.py` and raw recomputation | **Internal figure discrepancy.** Legend and normalization refer to different ranges. |

## High-risk claims requested explicitly

### ζ scaling

From the implemented definition,

```text
ζ = C J^(1/(n+1)),    J ∝ K_I²,    K_I ∝ S
```

so

```text
ζ ∝ S^(2/(n+1)) = S^(2/9.93) = S^0.201409869.
```

An independent log-log regression of the ten per-S mean ζ values recomputed
from raw K_I gives exponent **0.202402867**. The endpoint-derived exponent is
**0.202622436**. Therefore “approximately S^0.2” is **accurate**. It is a
definition-driven scaling, with the roughly 0.001 excess over the ideal
exponent caused by trained K_I scaling as S^1.00493 rather than exactly S.

Source: all raw `summary.json` K_I values, constants and definition in
`src/hrr_reference.py`.

### “37×” true-error growth

The ratio refers to neither sweep endpoint mean nor matched endpoints:

- S=0.05 mean = 0.02093107797; S=0.70 mean = 0.32871805436;
  endpoint-mean ratio = **15.70478381**.
- Global minimum run = 0.00883423688 at S=0.10, seed 5.
- Global maximum run = 0.33175495353 at S=0.70, seed 3.
- Global max/min ratio = **37.55332328**.

Thus the defensible endpoint statement is “mean true error increased from
2.09% to 32.87% (15.7×) between S=0.05 and S=0.70.” If the full individual
range matters, state separately that observed errors ranged from 0.883% to
33.175% (37.6×), and do not call that endpoint growth.

## Sweep source table: all ten points

Every entry below was independently regenerated from raw `summary.json` and
the formulas in `src/reference.py`, rather than copied from an aggregate CSV.
The checked-in `results/processed/summary_by_S.csv` agrees to floating-point
roundoff.

| S | n | Plastic-reference error mean (min–max) | Normalized loss mean (min–max) | LEFM verification error mean | ζ mean |
|---:|---:|---:|---:|---:|---:|
| 0.05 | 20 | 2.0931% (1.3516%–3.0772%) | 1.23935e-3 (2.25617e-4–1.45682e-2) | 1.9419% | 1.747608 |
| 0.10 | 20 | 1.7868% (0.8834%–3.3324%) | 3.70742e-4 (1.15480e-4–2.27210e-3) | 1.1765% | 2.012585 |
| 0.15 | 20 | 2.1046% (1.6228%–2.6676%) | 2.08194e-4 (9.18662e-5–1.04375e-3) | 0.7234% | 2.185856 |
| 0.20 | 20 | 3.2107% (2.5455%–4.4847%) | 2.45920e-4 (8.27352e-5–1.07398e-3) | 0.7514% | 2.316117 |
| 0.30 | 20 | 6.1507% (5.7165%–6.7329%) | 1.73218e-4 (4.10033e-5–1.32045e-3) | 0.5761% | 2.514095 |
| 0.40 | 20 | 10.5630% (9.7362%–11.2543%) | 3.26159e-4 (5.13671e-5–2.91056e-3) | 0.6006% | 2.664128 |
| 0.50 | 20 | 16.3470% (15.8574%–16.9069%) | 2.91885e-4 (4.11713e-5–1.28271e-3) | 0.5255% | 2.786853 |
| 0.60 | 20 | 23.6714% (23.3616%–24.1954%) | 8.53035e-5 (3.26308e-5–1.70393e-4) | 0.4416% | 2.891547 |
| 0.65 | 20 | 28.0820% (27.7607%–28.4806%) | 2.35064e-4 (3.32303e-5–1.18303e-3) | 0.5064% | 2.938155 |
| 0.70 | 20 | 32.8718% (32.6644%–33.1755%) | 1.33482e-4 (2.57516e-5–5.53840e-4) | 0.3720% | 2.983151 |

The absent paper prevents identifying which two points it omits. If the
omitted points are 0.15 and 0.65, neither reverses the main true-error or ζ
picture. They are still worth mentioning: S=0.15 shows the low-S error
returning upward after its S=0.10 dip, and S=0.65 shows normalized loss
rebounding by 2.76× from S=0.60 while true error and ζ continue upward. The
broad conclusion remains “error/ζ rise while loss is non-monotonic,” but
omitting them visually smooths two local features.

## Independently recomputed correlations and p-values

All values below were recomputed with SciPy 1.18.0 from raw summaries and, for
the field fit, from all 200 raw model weights. Pearson and Spearman p-values
are paired with the correct statistics in the checked-in stats files.

| Series vs plastic-reference error | Population | Pearson r (p) | Spearman ρ (p) | Stored values |
|---|---|---:|---:|---|
| Normalized training loss | all 200 runs | -0.117479636 (0.0975718) | -0.412066302 (1.34033e-9) | Match |
| Normalized training loss | 10 per-S means | -0.432332004 (0.212099) | -0.636363636 (0.0479117) | Match |
| Field-fit exponent deviation | all 200 runs | +0.105708682 (0.136289) | +0.210684767 (0.00274857) | Match |
| Field-fit exponent deviation | 10 per-S means | +0.486219883 (0.154182) | +0.612121212 (0.0599721) | Match |
| ζ | all 200 runs | +0.878943718 (1.38570e-65) | +0.950428261 (2.11815e-102) | Match |
| ζ | 10 per-S means | +0.879385603 (7.98526e-4) | +0.987878788 (9.30746e-8) | Match |

The broad phrase “loss does not correlate with error” needs population-level
qualification. Within fixed S, Pearson correlations are significantly
positive at S=0.05 (r=0.5656, p=0.00935), S=0.10 (r=0.4481, p=0.0475),
S=0.20 (r=0.6149, p=0.00391), and S=0.60 (r=0.5041, p=0.0234). Spearman is
significantly positive at S=0.20 and S=0.60. The correct qualitative claim is
that loss fails to track the **load-driven across-S increase** in true error;
there are several positive within-S seed-noise correlations.

## Sigma-separation recomputation

Using the code's exact definition Δmean/sqrt(s_S²+s_0.05²):

| S | True error separation | ζ separation | Field-fit separation |
|---:|---:|---:|---:|
| 0.05 | 0.000 | 0.0 | 0.000 |
| 0.10 | -0.453 | 99.7 | -0.255 |
| 0.15 | 0.023 | 227.9 | -0.248 |
| 0.20 | 1.974 | 240.1 | -0.158 |
| 0.30 | 8.286 | 380.5 | 0.140 |
| 0.40 | 16.456 | 395.8 | 0.274 |
| 0.50 | 27.873 | 425.2 | 0.224 |
| 0.60 | 45.702 | 522.5 | 0.153 |
| 0.65 | 55.149 | 525.5 | 0.042 |
| 0.70 | 68.109 | 601.7 | 0.039 |

Conventional equal-n pooled-SD values would be sqrt(2) times these values.
Values based on standard errors of the means would answer a different
question and be larger again. The paper should include the formula to avoid
calling these interchangeable “sigma” statistics.

## Method facts checked against executable sources and artifacts

| Fact | Source value | Source | Verdict |
|---|---|---|---|
| Sweep points | 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.65, 0.70 | `configs/sweep_config.json`; raw directory names | Verified |
| Seeds | 0–19 at every S; 20 unique seeds/point | all raw directory names and summaries | Verified |
| Run total | 10×20 = 200; all 200 have `loss.dat`, `model.dict`, and `summary.json` | `results/raw/` | Verified |
| Epochs | 2,000 for every run; every loss file has exactly 2,000 rows | raw summaries and loss files | Verified |
| Optimizer / learning rate | Adam inside pihnn train; wrapper passes 1e-2 | `src/pihnn_wrapper.py` and installed `pihnn/utils.py` | Verified from code |
| Network units | [1,10,10,10,1] | `src/pihnn_wrapper.py` | Verified from code/model construction |
| Training/test boundary points | 1,000 / 40 | `src/pihnn_wrapper.py` | Verified from code |
| Geometry | l=4, h=6, a=1; W=8 | `src/reference.py`; `src/pihnn_wrapper.py` | Verified from executable code |
| Net-section limit | (W-2a)/W = 6/8 = 0.75 | geometry constants and formula | Verified algebraically |
| σ_y | 1.0, so S=σ_ext numerically | `src/reference.py`; wrapper boundary loads | Verified from code |
| Elastic baseline | K_el=1.852 S | `src/reference.py`; every raw summary recomputes exactly | Verified as the implemented reference; external origin still needs the paper/example source |
| Material constants | λ=μ=1 default, giving ν=0.25 and E=2.5 | installed pihnn 1.1 `nn.py`; `src/reference.py` | Verified |
| SIF extraction radius | 1.0 (default argument passed as `A_CRACK`) and 1,000 integration points (library default) | `src/pihnn_wrapper.py`; installed pihnn `utils.py` | Verified |
| Field-fit window | 15 log-spaced points on [0.01, 0.1] | `src/field_fit.py` | Verified; all 200 model recomputations exactly match stored CSV |
| ζ radius | r*=0.05 = 5% of a | `src/hrr_reference.py` | Verified |
| HRR constants used | n=8.93, I_n=4.61, stress angular factor=2.46, α=1 | `src/hrr_reference.py` | Verified as implementation values; numeric table still requires checking against the cited paper itself |
| pihnn version | 1.1 | installed package metadata | Verified |
| CPU-only run | sweep log prints CPU device throughout | `results/logs/sweep_run.log` | Verified |
| Raw-to-summary integrity | final train/test losses, normalization, plateau ratios, K_el, and LEFM errors all recompute for 200/200; K_I/K_II recomputed from every model weight file with zero difference | all raw artifacts | Verified |
| Processed-file integrity | `all_runs.csv` and `detectors.csv` match regenerated raw values to floating-point roundoff | raw artifacts and source formulas | Verified |

## Bibliography status

`refs.bib` is missing, so no entry-by-entry comparison, entry-type check, or
identification of the miscategorised book is possible.

The authoritative metadata located for references visible in the reports is:

- Matteo Calafà, Henrik Myhre Jensen, and Tito Andriollo, “Solving plane
  crack problems via enriched holomorphic neural networks,” *Engineering
  Fracture Mechanics* 322 (2025), article 111133,
  DOI 10.1016/j.engfracmech.2025.111133. This is an **article**.
- Matteo Calafà, Emil Hovad, Allan P. Engsig-Karup, and Tito Andriollo,
  “Physics-Informed Holomorphic Neural Networks (PIHNNs): Solving 2D linear
  elasticity problems,” *Computer Methods in Applied Mechanics and
  Engineering* 432 (2024), article 117406,
  DOI 10.1016/j.cma.2024.117406. This is an **article**.
- Andrzej Neimitz and Sebastian Lipiec, “Stress distribution in front of the
  crack – analytical solutions vs. numerical. Can the differences be
  minimized?”, *Journal of Theoretical and Applied Mechanics* 57(3) (2019),
  713–721, DOI 10.15632/jtam-pl/110216. This is an **article**.

These records do not substitute for checking the absent BibTeX fields. The
Neimitz–Lipiec bibliographic metadata is verified against the journal's
official author index, but the Table 3 numeric constants were not independently
read from an accessible publisher PDF during this audit and remain flagged.

## Still unverifiable / required to finish

1. Every number, table row, equation constant, caption, and factual claim in
   `main.tex`: the file is missing.
2. Which eight sweep rows the paper prints and which two it omits: `main.tex`
   is missing.
3. Every `refs.bib` field and the miscategorised book: `refs.bib` is missing.
4. The 1,000/2,000/4,000-epoch calibration numbers: no underlying run data.
5. The K_I=1.841 benchmark reproduction and its 0.59% error: no run artifact.
6. Original detector wall time: no detector timing log.
7. HRR Table 3 constants and the α=1 textual attribution against the article
   body: only bibliographic metadata was independently accessible.

## Wording changes needed even if rounded numbers are retained

1. Replace “37× growth” with an endpoint-mean statement (15.7×) or explicitly
   identify 37.6× as a global individual-run max/min range.
2. Qualify “loss does not correlate with error” as an across-load statement;
   several within-S correlations are positive and significant.
3. Do not call the plateau behavior clean for every run. Use the independently
   accurate `<3%` LEFM verification control as the main convergence evidence.
4. Describe ζ as an HRR-informed post-hoc severity indicator or stand-in, not
   a literal overlap residual or genuine trained two-region construction.
5. Define the sigma-separation formula and call it a root-sum-square SD
   separation heuristic, not generic “pooled σ” or statistical significance.
6. State that `p_fit - 0.5` becoming negative—not positive—would indicate
   movement toward the HRR exponent for n=8.93.
7. Align the reference-uncertainty threshold across prose, code comments, and
   figure shading.
