"""
Thin wrapper around pihnn's straight-crack (Williams-enriched) example.

This module does not modify the installed pihnn package. It only imports its
public API and reproduces the geometry/hyperparameters of
examples/cracks/straight_crack_williams.py from github.com/teocala/pihnn,
parametrizing the one quantity this study sweeps: the remote applied stress
`sig_ext` (equivalently S = sig_ext / sigma_y, since sigma_y = 1 here).

pihnn is an ELASTIC-ONLY solver: nothing in this file, and nothing pihnn
itself does, ever sees a yield stress or a plasticity model. That is the
point of the experiment -- see reference.py and report/REPORT.md.
"""
import json
import math
import os
import time
import warnings

import numpy as np
import torch

import pihnn.nn as nn
import pihnn.utils as utils
import pihnn.geometries as geom
import pihnn.bc as bc

from reference import A_CRACK, L_HALF, H_HALF, K_EL_HAT

class _SilentPbar:
    """Drop-in replacement for tqdm that iterates silently but still
    tolerates pihnn's pbar.set_postfix_str(...) calls."""

    def __init__(self, iterable, **kwargs):
        self._iterable = iterable

    def __iter__(self):
        return iter(self._iterable)

    def set_postfix_str(self, *args, **kwargs):
        pass


# Silence pihnn's per-epoch tqdm bar (we run O(100) trainings sequentially).
# This patches the *imported reference* in our own process; it does not touch
# any file inside the installed pihnn package.
utils.tqdm = _SilentPbar
warnings.filterwarnings("ignore", message="Converting a tensor with requires_grad=True")

UNITS = [1, 10, 10, 10, 1]
NP_TRAIN = 1000
NP_TEST = 40
BETA = 0.5
GAUSS = 3
LEARN_RATE = 1e-2
SCHEDULER_APPLY = []
SIF_CONTOUR_RADIUS = A_CRACK
SIF_INTEGRATION_POINTS = 1000


def compute_sif_diagnostics(
    model,
    crack,
    tip_index: int = 0,
    radius: float = SIF_CONTOUR_RADIUS,
    integration_points: int = SIF_INTEGRATION_POINTS,
) -> dict:
    """Compute coefficient- and interaction-integral SIFs at one crack tip.

    pihnn stores its trainable complex coefficient as K_I + i*K_II.  The
    individual real components below are convention-independent; the metadata
    records the convention explicitly so downstream code can form either
    K_I+i*K_II or K_I-i*K_II without guessing.

    ``g`` is a reported diagnostic only.  It is not used by training,
    convergence checks, or snapshot selection.
    """
    if not 0 <= tip_index < len(crack.tips):
        raise IndexError(f"tip_index {tip_index} is outside [0, {len(crack.tips)})")

    # build_boundary() creates one crack and pihnn orders model.sif in the same
    # order as crack.tips. Read the coefficient before yau_wang_method
    # temporarily swaps model.sif while applying its auxiliary fields.
    K_c = model.sif[tip_index].detach()
    K_c_I = float(K_c.real)
    K_c_II = float(K_c.imag)

    K_J_I_tensor, K_J_II_tensor = utils.yau_wang_method(
        model,
        crack.tips[tip_index],
        radius,
        integration_points=integration_points,
    )
    K_J_I = float(K_J_I_tensor)
    K_J_II = float(K_J_II_tensor)

    K_J_abs = math.hypot(K_J_I, K_J_II)
    if K_J_abs == 0.0:
        g = None
        g_status = "undefined_zero_K_J"
    else:
        g = math.hypot(K_c_I - K_J_I, K_c_II - K_J_II) / K_J_abs
        g_status = "ok"

    return {
        "K_c_I": K_c_I,
        "K_c_II": K_c_II,
        "K_J_I": K_J_I,
        "K_J_II": K_J_II,
        "g": g,
        "g_status": g_status,
        "sif_tip_index": tip_index,
        "sif_contour_radius": float(radius),
        "sif_integration_points": int(integration_points),
        "sif_library_complex_convention": "K_I + i*K_II",
        "sif_report_complex_convention": "K_I - i*K_II",
    }


def build_boundary(sig_ext: float):
    l, h, a = L_HALF, H_HALF, A_CRACK
    line1 = geom.line(P1=[-l, -h], P2=[l, -h], bc_type=bc.stress_bc(), bc_value=-sig_ext * 1j)
    line2 = geom.line(P1=[l, -h], P2=[l, h], bc_type=bc.stress_bc(), bc_value=0)
    line3 = geom.line(P1=[l, h], P2=[-l, h], bc_type=bc.stress_bc(), bc_value=sig_ext * 1j)
    line4 = geom.line(P1=[-l, h], P2=[-l, -h], bc_type=bc.stress_bc(), bc_value=0)
    crack = geom.line(P1=[-a, 0], P2=[a, 0], bc_type=bc.stress_bc(), bc_value=0.0)
    interface = geom.line(P1=[a, 0], P2=[l, 0], bc_type=bc.interface_bc())
    interface2 = geom.line(P1=[-a, 0], P2=[-l, 0], bc_type=bc.interface_bc())

    def dd_partition(x, y):
        return torch.stack([(y >= -1e-10), (y <= 1e-10)])

    crack.add_crack_tip(tip_side=0)
    crack.add_crack_tip(tip_side=1)
    boundary = geom.boundary(
        [line1, line2, line3, line4, crack, interface, interface2],
        NP_TRAIN, NP_TEST, dd_partition, enrichment="williams",
    )
    return boundary, crack


def run_training(
    S: float,
    seed: int,
    n_epochs: int,
    out_dir: str,
    include_sif_diagnostics: bool = False,
) -> dict:
    """
    Train one enriched elastic pihnn model at applied stress sig_ext = S
    (sigma_y is fixed at 1, so S is directly the load-over-yield ratio),
    with a given random seed. Returns a summary dict and writes raw
    per-run outputs (config, loss trajectory, model weights, summary) to
    out_dir, which must not already contain a summary.json (idempotent:
    existing runs are never overwritten).
    """
    summary_path = os.path.join(out_dir, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            return json.load(f)

    os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    boundary, crack = build_boundary(S)
    model = nn.enriched_PIHNN("km", UNITS, boundary)
    model.initialize_weights("exp", BETA, boundary.extract_points(10 * NP_TRAIN)[0], GAUSS)

    t0 = time.time()
    loss_train, loss_test = utils.train(
        boundary, model, n_epochs, LEARN_RATE, SCHEDULER_APPLY, dir=out_dir
    )
    wall_time = time.time() - t0

    loss_train = np.array([float(x) for x in loss_train])
    loss_test = np.array([float(x) for x in loss_test])

    # NOTE: must NOT be wrapped in torch.no_grad() -- compute_J_integral
    # internally differentiates the network output w.r.t. its input points.
    sif_diagnostics = None
    if include_sif_diagnostics:
        sif_diagnostics = compute_sif_diagnostics(model, crack)
        K_I = sif_diagnostics["K_J_I"]
        K_II = sif_diagnostics["K_J_II"]
    else:
        sif_i, sif_ii = utils.yau_wang_method(model, crack.tips[0], A_CRACK)
        K_I = float(sif_i)
        K_II = float(sif_ii)

    # Convergence diagnostic 1: numerical convergence / plateau check.
    # Compare mean loss over the last 5% of epochs to the preceding 5%.
    n = len(loss_train)
    tail = max(10, n // 20)
    last_block = loss_train[-tail:]
    prev_block = loss_train[-2 * tail:-tail]
    plateau_ratio = float(np.mean(last_block) / max(np.mean(prev_block), 1e-300))

    final_train_loss = float(loss_train[-1])
    final_test_loss = float(loss_test[-1])
    normalized_train_loss = final_train_loss / max(S ** 2, 1e-300)
    normalized_test_loss = final_test_loss / max(S ** 2, 1e-300)

    # Convergence diagnostic 2: physical / LEFM verification error.
    # pihnn is solving an exact linear-elastic BVP whose analytical SIF is
    # known independently of any plasticity question: K_el(S) = S * K_EL_HAT.
    # If pihnn is well-trained, K_I should match this almost exactly at
    # EVERY S, regardless of whether LEFM is the *physically* appropriate
    # model at that S. This isolates "undertrained" from "assumption invalid".
    K_el_exact = S * K_EL_HAT
    lefm_verification_error = abs(K_I - K_el_exact) / abs(K_el_exact)

    summary = {
        "S": S,
        "seed": seed,
        "n_epochs": n_epochs,
        "wall_time_sec": wall_time,
        "K_I": K_I,
        "K_II": K_II,
        "final_train_loss": final_train_loss,
        "final_test_loss": final_test_loss,
        "normalized_train_loss": normalized_train_loss,
        "normalized_test_loss": normalized_test_loss,
        "plateau_ratio": plateau_ratio,
        "K_el_exact": K_el_exact,
        "lefm_verification_error": lefm_verification_error,
        "out_dir": out_dir,
    }
    if sif_diagnostics is not None:
        summary.update(sif_diagnostics)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary
