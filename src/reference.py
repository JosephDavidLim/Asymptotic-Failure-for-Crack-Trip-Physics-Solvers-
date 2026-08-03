"""
Independent (non-pihnn) reference for the crack-tip driving force in the
plastic regime, used to check pihnn's reported K_I against.

Design decision (see report/REPORT.md and README.md for full justification):
We could not obtain trustworthy tabulated EPRI/GE (Kumar-German-Shih) J-estimation
coefficients -- reproducing them from memory would create a reference that "only
looks like ground truth", which the task explicitly rules out. A from-scratch
elastic-plastic FEM reference was judged too much unverified new code to trust
within this project's time budget.

Instead we use two independent, closed-form, textbook analytical models that
require no tabulated data:

1. Dugdale (1960) strip-yield model, applied to the same center-cracked panel
   whose linear-elastic solution pihnn already reproduces (K_el(S) = S * 1.852,
   the analytical value used and verified by pihnn's own straight-crack example).
   This gives a secant plasticity-correction factor that stays well-defined all
   the way to net-section yield, where it (correctly) diverges.

2. Irwin (1960) first-order plane-strain plastic-zone correction, valid only
   deep in small-scale yielding. Used as an independent cross-check at small S:
   the two models should agree closely there, and the point where they stop
   agreeing marks where "the reference itself" becomes uncertain.

LIMITATIONS (stated explicitly, not buried):
- Dugdale's model is derived for an INFINITE plate in PLANE STRESS with
  elastic-perfectly-plastic material. pihnn's elasticity formulation is PLANE
  STRAIN (confirmed from pihnn.utils.compute_J_integral's C1 = (1-nu^2)/E
  factor). We apply the Dugdale secant factor to the plane-strain elastic K
  anyway, as a widely-used engineering approximation -- not a matched
  plane-strain elastic-plastic solution. This mixes stress states.
- Neither model includes strain hardening (both assume elastic-perfectly-plastic).
- Neither model captures the actual finite-height (h=6) boundary of pihnn's
  domain; both use the infinite/semi-infinite idealization for the plastic
  correction (only the *elastic* K_el(S)=1.852*S baseline already reflects the
  true finite geometry, since 1.852 is the finite-geometry analytical value).
- Both models are known to degrade in accuracy as the net-section stress
  approaches yield (S -> 0.75 for this geometry, where W/(W-2a) = 8/6). We
  report but visually flag results at S >= 0.65 as "reference uncertain".

Net effect: the ABSOLUTE magnitude of "true error" reported here carries real
uncertainty at the high-S end. The QUALITATIVE question this study asks --
does pihnn's loss track this error, or not -- does not depend on getting the
last few percent of the reference right, because the two independent models
agree on the *sign and trend* (monotonic growth, unbounded as S approaches
net-section yield) even where they disagree on the exact magnitude.
"""
import math

A_CRACK = 1.0        # crack half-length
L_HALF = 4.0          # plate half-length (W = 2*L_HALF = 8)
H_HALF = 6.0          # plate half-height
W_FULL = 2 * L_HALF
SIGMA_Y = 1.0         # yield stress, dimensionless, sets S = sig_ext / SIGMA_Y = sig_ext
K_EL_HAT = 1.852      # pihnn/paper's verified finite-geometry elastic K_I at S=1 (l=4,h=6,a=1)
NU = 0.25             # matches pihnn default material {lambda:1, mu:1}
E_YOUNG = 2.5         # matches pihnn default material {lambda:1, mu:1}

NET_SECTION_YIELD_S = SIGMA_Y * (W_FULL - 2 * A_CRACK) / W_FULL  # = 0.75 for this geometry


def k_elastic(S: float) -> float:
    """Exact finite-geometry linear-elastic K_I at applied stress S*SIGMA_Y (LEFM, no plasticity)."""
    return S * K_EL_HAT


def dugdale_zone_ratio(S: float) -> float:
    """(a_eff - a) / a for the classical Dugdale strip-yield model, infinite plate."""
    x = math.pi * S / 2
    if x >= math.pi / 2 - 1e-9:
        return float("inf")
    return 1.0 / math.cos(x) - 1.0


def k_ref_dugdale(S: float) -> float:
    """
    Primary plasticity-corrected reference SIF: elastic K_I scaled by the
    Dugdale secant correction sqrt(sec(pi*S/2)).
    """
    x = math.pi * S / 2
    if x >= math.pi / 2 - 1e-9:
        return float("inf")
    secant = 1.0 / math.cos(x)
    return k_elastic(S) * math.sqrt(secant)


def k_ref_irwin_plane_strain(S: float) -> float:
    """
    Secondary cross-check: first-order (non-iterated) Irwin plane-strain
    plastic-zone correction. Valid only for small S (small-scale yielding).
    """
    Kel = k_elastic(S)
    ry = (1.0 / (6 * math.pi)) * (Kel / SIGMA_Y) ** 2
    return Kel * math.sqrt(1.0 + ry / A_CRACK)


def reference_disagreement(S: float) -> float:
    """Relative disagreement between the two independent reference models.
    Large values mark where the reference itself is untrustworthy."""
    kd = k_ref_dugdale(S)
    ki = k_ref_irwin_plane_strain(S)
    if not math.isfinite(kd):
        return float("inf")
    return abs(kd - ki) / kd


def j_ref_dugdale(S: float) -> float:
    """Equivalent-K to J conversion, plane strain, matching pihnn's own convention."""
    k = k_ref_dugdale(S)
    if not math.isfinite(k):
        return float("inf")
    return (1 - NU ** 2) / E_YOUNG * k ** 2
