"""
Detector 1 (post-hoc, cheapest route): reload each already-converged pihnn
model from the prior sweep (no retraining) and fit a free-exponent power law
sigma_theta-theta(r) = A * r^(-p) to its own near-tip field, over a FIXED
radius window used identically at every S. This directly controls for the
"mismatch grows just because the fit window widened" artifact named in the
task: the window never changes, only S does.

If pihnn's field is (as argued in report/REPORT_matched_asymptotics.md) very
close to pure elastic at every S -- because pihnn has no plasticity model
and its enrichment only ever represents a K-dominant singular term -- then
p_fit should stay close to 0.5 at every S, and this detector should be as
blind to S as the training loss was. That is a real, checkable prediction,
not an assumption; this module exists to check it.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import pihnn.nn as nn  # noqa: E402
from pihnn_wrapper import build_boundary, UNITS  # noqa: E402

# Fixed radius window, identical for every S (controls for the widening-window artifact).
R_WINDOW = np.geomspace(0.01, 0.1, 15)


def load_trained_model(S: float, model_dict_path: str):
    boundary, crack = build_boundary(S)
    model = nn.enriched_PIHNN("km", UNITS, boundary)
    state = torch.load(model_dict_path, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, boundary, crack


def sample_hoop_stress(model, crack, r_values):
    """sigma_theta-theta(r, theta=0) ahead of the right crack tip (tip index 1,
    coords=+a); at theta=0 the local hoop stress equals the global sigma_yy."""
    tip = crack.tips[1]
    z = (tip.coords + torch.tensor(r_values, dtype=torch.complex128)).requires_grad_(True)
    sxx, syy, sxy, ux, uy = model(z, real_output=True, force_williams=True)
    return syy.detach().numpy()


def fit_power_law(r_values, sigma_values):
    """Fit log(sigma) = log(A) - p*log(r) by ordinary least squares.
    Returns (A, p, residual_rms_log)."""
    x = np.log(r_values)
    y = np.log(np.abs(sigma_values))
    p_neg, log_A = np.polyfit(x, y, 1)  # y = p_neg * x + log_A
    p = -p_neg
    fitted = log_A + p_neg * x
    resid_rms = float(np.sqrt(np.mean((y - fitted) ** 2)))
    return float(np.exp(log_A)), float(p), resid_rms


def naive_field_fit_detector(S: float, model_dict_path: str) -> dict:
    model, boundary, crack = load_trained_model(S, model_dict_path)
    sigma = sample_hoop_stress(model, crack, R_WINDOW)
    A_fit, p_fit, resid_rms = fit_power_law(R_WINDOW, sigma)
    return {
        "field_fit_amplitude": A_fit,
        "field_fit_exponent": p_fit,
        "field_fit_exponent_deviation": p_fit - 0.5,  # 0 => pure elastic, >0 => softer/HRR-like decay
        "field_fit_resid_rms_log": resid_rms,
    }
