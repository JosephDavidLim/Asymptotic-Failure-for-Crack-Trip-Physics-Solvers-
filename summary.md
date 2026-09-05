# SIF self-consistency diagnostic summary

## Repository findings

This repository is a Python 3.10+ project using PyTorch through the external
`pihnn` v1.1 package. The local code wraps pihnn's enriched holomorphic neural
network solver; the underlying network, Williams enrichment, contour integral,
and Yau-Wang interaction method are implemented in the installed dependency.

There is no reduced-order model (ROM) in the repository. `src/sweep.py` is a
parameter sweep that trains an independent full solver for each `(S, seed)`
pair. There is no reduced basis, surrogate, latent model, or parameter-to-SIF
predictor. A ROM would need to be built separately.

Each trained run stores:

- `model.dict`: complete trained weights, including the enrichment coefficient
- `loss.dat`: training and test loss history
- `summary.json`: extracted SIFs and convergence diagnostics

Historical `summary.json` files contain only the integral-extracted SIF, but
the trainable enrichment coefficient remains recoverable from `model.dict`.

## SIF sources

### Enrichment coefficient, K_c

The trainable Williams coefficient is `model.sif`, created by
`pihnn.nn.enriched_PIHNN`. The wrapper creates the left crack tip first and the
right crack tip second, so for the current geometry:

```python
K_c_left = model.sif[0]
K_c_right = model.sif[1]
```

The coefficient is inserted into the Williams potentials with the conventional
`1/sqrt(2*pi)` normalization.

### Integral-extracted SIF, K_J

`pihnn.utils.yau_wang_method()` computes both SIF components using interaction
integrals constructed from `pihnn.utils.compute_J_integral()`.

Before this change, the repository's saved `K_I` and `K_II` fields already came
from this Yau-Wang calculation. They are therefore K_J, not K_c. Their existing
names and meanings have been preserved for compatibility.

The current contour settings are:

- Crack tip index: 0, the left tip
- Radius: `A_CRACK = 1.0`
- Integration points: 1,000

## Implemented change

`src/pihnn_wrapper.py` now provides:

```python
compute_sif_diagnostics(model, crack, tip_index=0, radius=1.0,
                        integration_points=1000)
```

It returns the JSON-safe fields:

- `K_c_I`, `K_c_II`
- `K_J_I`, `K_J_II`
- `g`
- `g_status`
- `sif_tip_index`
- `sif_contour_radius`
- `sif_integration_points`
- Complex-convention metadata

The diagnostic is calculated as:

```text
g = sqrt((K_c_I - K_J_I)^2 + (K_c_II - K_J_II)^2)
    / sqrt(K_J_I^2 + K_J_II^2)
```

If `|K_J|` is exactly zero, `g` is stored as `null` and `g_status` is
`undefined_zero_K_J`.

`run_training()` now accepts:

```python
include_sif_diagnostics=False
```

The option is disabled by default. When disabled, output and behavior are
unchanged. When enabled, the new fields are added to the returned dictionary
and `summary.json`. Existing `K_I` and `K_II` remain unchanged and continue to
contain the Yau-Wang result.

`configs/sweep_config.json` contains the corresponding default:

```json
"include_sif_diagnostics": false
```

Set it to `true` for future training sweeps that should save both K_c and K_J.
`src/aggregate.py` recognizes the new optional columns while continuing to
read all historical summaries.

The value of `g` is a computed diagnostic only. It is not connected to:

- Training or the loss function
- Convergence or stopping decisions
- Error bounds
- Gating or filtering
- Snapshot selection

## Sanity check

The read-only sanity check reloads an existing model without retraining or
modifying its saved results:

```bash
.venv/bin/python src/sanity_check_sif.py
```

An alternative run directory can be supplied as the first argument.

For `results/raw/S0.200_seed000`, it reports:

```text
K_c = 0.403319303802-0.00662443583801j  (K_I - i*K_II)
K_J = 0.368275353393+0.00132369927887j  (K_I - i*K_II)
g   = 0.0975730605585
contour: tip=0, radius=1.0, points=1000
```

Thus this existing example has a self-consistency discrepancy of approximately
9.76% under the current contour settings.

## Normalization and convention audit

No fixed factor-of-2 or factor-of-`sqrt(2*pi)` mismatch was found. A synthetic
pure Williams field with coefficient `1 + 0.2i` was recovered by the Yau-Wang
method as `K_I = 1` and `K_II = 0.2` across multiple contour radii.

Both paths use the plane-strain relation:

```text
C1 = (1 - nu^2) / E
```

For the current model, `nu = 0.25` and `E = 2.5`.

There is a complex-sign notation difference:

- pihnn stores and documents `K = K_I + i*K_II`.
- The requested reporting convention is `K = K_I - i*K_II`.

The implementation stores the real SIF components separately and records both
conventions as metadata. The sanity-check display converts both values to the
requested minus-sign convention. Conjugating both values does not change `g`.

The observed K_c/K_J discrepancy is therefore not explained by a constant
normalization factor. The integral result can also depend on contour radius for
an approximate trained field, which is why the selected radius and integration
point count are stored next to every enabled diagnostic.

## Verification

- All source files compile successfully.
- The new K_J fields exactly reproduce the legacy `K_I` and `K_II` values on
  the checked existing model.
- All 200 historical summaries still aggregate successfully, with blank cells
  for diagnostics they do not contain.
- No existing raw results were rewritten.

## Remaining limitation

No ROM snapshot pipeline exists to update. If a ROM is introduced later, each
training snapshot should retain `K_c_I`, `K_c_II`, `K_J_I`, `K_J_II`, and the
contour metadata rather than saving only `g` or a single SIF. This keeps the
diagnostic reconstructable and allows its definition to be revisited later.
