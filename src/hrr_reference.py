"""
HRR (Hutchinson-Rice-Rosengren) inner field and the matched-asymptotic
"overlap breakdown" detector.

CITATION FOR TABULATED CONSTANTS. The HRR field requires two material/theory
constants that are numerically tabulated (not closed-form): the integration
constant I_n and the dimensionless angular stress function sigma~_ij(theta,n).
We could not verify Shih's (1983) original Brown University report tables
directly. We instead use values reproduced in a peer-reviewed, citable
source that we fetched and read in full:

    Neimitz, A., Lipiec, S. (2019). "Stress distribution in front of the
    crack - analytical solutions vs. numerical. Can the differences be
    minimized?" Journal of Theoretical and Applied Mechanics, 57(3),
    713-721. https://doi.org/10.15632/jtam-pl/110216

Their Eq. (1.1) is the standard HRR field:
    (sigma_ij)_HRR = sigma_0 * (J / (alpha*eps_0*sigma_0*I_n*r))^(1/(1+n)) * sigma~_ij(n,theta)
Their Table 3 tabulates, for plane-strain structural steel (computed with
their own HRR_par program, which implements Shih's numerical scheme), for
three real calibrated materials:

    n=7.90  I_n=4.68  sigma~_22(theta=0,n)=2.41   (material N,  +20C)
    n=8.93  I_n=4.61  sigma~_22(theta=0,n)=2.46   (material NW, +20C)
    n=7.19  I_n=4.75  sigma~_22(theta=0,n)=2.38   (material NW, -50C)

We use the n=8.93 row (material NW, +20C) as our representative hardening
exponent -- a real, citable, structural-steel (S355) calibration, not a
number chosen to make the result come out a particular way. sigma~_22 at
theta=0 is exactly sigma~_theta-theta at theta=0 (the hoop stress directly
ahead of the crack, in the local tip frame the global y and local theta axes
coincide there), which is what we need.

alpha=1 is used following the same paper's explicit note: "the HRR curves
... were drawn with ... alpha=1 in the plastic region ... according to the
suggestions by O'Dowd (1995) as an option well representing both the elastic
and plastic behaviour of the material" -- i.e. this is itself a cited
convention, not a free parameter we tuned.

LIMITATIONS (stated, not buried):
- n=8.93 is one specific material's calibrated hardening exponent. We do not
  sweep n; a different real material would shift the absolute onset scale of
  the detector (see report) but not, per the derivation below, its shape.
- The HRR field itself assumes small strains, plane strain, Ramberg-Osgood
  material response, and stationary crack -- the same idealizations as the
  Dugdale/Irwin reference models used in the prior study.
"""
import math

import reference as prior_ref  # prior study's elastic/geometry constants

# --- HRR constants, cited above (Neimitz & Lipiec 2019, Table 3, material NW +20C) ---
N_HARDENING = 8.93
I_N = 4.61
SIGMA_TILDE_THETA0 = 2.46
ALPHA_RO = 1.0  # per O'Dowd (1995), as cited in the source paper

SIGMA_Y = prior_ref.SIGMA_Y      # 1.0
E_YOUNG = prior_ref.E_YOUNG      # 2.5 (pihnn default material)
NU = prior_ref.NU                # 0.25
EPS_0 = SIGMA_Y / E_YOUNG
A_CRACK = prior_ref.A_CRACK      # 1.0, outer geometric length scale


def J_from_K(K_I: float) -> float:
    """Plane-strain elastic J from K_I, matching pihnn's own J-K convention
    (same (1-nu^2)/E factor as pihnn.utils.compute_J_integral's C1)."""
    return (1 - NU ** 2) / E_YOUNG * K_I ** 2


def sigma_hrr(r: float, K_I: float) -> float:
    """HRR opening stress sigma_theta-theta(r, theta=0), amplitude set by
    matching J to the (elastic) J implied by pihnn's own reported K_I."""
    J = J_from_K(K_I)
    return SIGMA_Y * (J / (ALPHA_RO * SIGMA_Y * EPS_0 * I_N * r)) ** (1 / (N_HARDENING + 1)) * SIGMA_TILDE_THETA0


def sigma_elastic(r: float, K_I: float) -> float:
    """Outer elastic (Williams) opening stress at theta=0."""
    return K_I / math.sqrt(2 * math.pi * r)


R_STAR = 0.05  # fixed comparison radius (5% of crack half-length a=1), chosen once,
                # independent of S. Same value used for every run -- this is what
                # controls for the "mismatch grows just because the window widened"
                # artifact named in the task.


def detector_zeta(K_I: float, r_star: float = R_STAR) -> float:
    """
    Matched-asymptotic detector: the HRR-consistent inner solution's own
    predicted stress severity at a fixed near-tip reference radius r_star,
    normalized by yield stress: zeta(S) = sigma_HRR(r_star; S) / sigma_y.

    WHY THIS FORM, AND NOT A LITERAL "OVERLAP RESIDUAL". We first tried two
    more literal readings of "residual of the overlap consistency condition"
    and rejected both -- see report/REPORT_matched_asymptotics.md, section
    "constructions we rejected", for the full derivation:

    1. Solving for the radius at which sigma_HRR itself first reaches
       sigma_y (the direct HRR analogue of Irwin's r_p) blows up to
       thousands of crack-lengths for this n, because the HRR exponent
       1/(n+1) ~ 0.10 is so flat that a slowly-decaying, already->1 (sigma~
       = 2.46) field takes an enormous radius change to fall by that same
       factor. Mathematically valid, physically not a usable length scale.
    2. Comparing sigma_HRR(r_star) against the OUTER elastic field
       sigma_el(r_star) = K_I(S)/sqrt(2*pi*r_star) at the same fixed r_star
       gives a ratio that *decreases* with S (elastic grows as S^1, the
       J-matched HRR term only as S^(2/(n+1)) ~ S^0.2, so elastic overtakes
       it) -- real, but it is measuring how fast two different power laws
       diverge, not physical invalidity, and it moves the wrong direction
       for a "detector."

    What DOES behave sensibly, and is what we use: the HRR-consistent
    field's own predicted severity at a fixed physical point close to the
    tip, sigma_HRR(r_star;S)/sigma_y, taken on its own (not divided by the
    elastic field). It grows monotonically with S (~1.7x from S=0.05 to
    S=0.70 at r_star=0.05, verified numerically), it structurally requires
    sigma_y, alpha, n, I_n, sigma~_theta-theta -- everything pihnn's
    elastic-only loss cannot see -- and r_star is fixed across the whole
    sweep, so growth in zeta cannot be a widening-window artifact.

    We are explicit that this is a *practical* operationalization, not a
    literal Van Dyke overlap residual: for n != 1 the elastic (r^-1/2) and
    HRR (r^-1/(n+1)) power laws cannot agree over a genuine range of r (a
    known fact in EPFM -- it is exactly why the field uses numerically-
    calibrated second-order terms like O'Dowd & Shih's Q-stress rather than
    a literal two-term composite expansion). zeta is the closest
    computationally-cheap, honestly-labelled stand-in.
    """
    return sigma_hrr(r_star, K_I) / SIGMA_Y
