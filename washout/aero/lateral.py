"""Directional and roll stability: the axis nothing here has modelled.

Every gate in this program so far judges the aircraft in PITCH. The
vortex lattice trims it, the static margin keeps it from diverging, the
tip-stall margin decides which end gives up first -- all longitudinal.
A tailless swept wing's hardest problem is the OTHER two axes, and the
optimizer has been free to do whatever it liked there because nothing
was looking.

That freedom is not theoretical. Sweep is the only thing giving these
designs any yaw stiffness at all, and the search has been handed sweep
as three independent variables to minimise drag with. A flying wing
with too little Cn_beta does not crash dramatically; it wanders, it
will not hold a heading, and it drops a wing in a turn because yaw and
roll are coupled through sweep. That is a bad trainer.

WHAT IS DERIVED AND WHAT IS CALIBRATED, stated plainly, because mixing
the two silently is how a model earns unearned trust:

  * the fin/winglet term is GEOMETRIC -- side area and its arm are
    integrated from the actual lofted surface, no coefficient invented
  * the dihedral term is STRIP THEORY -- the standard sideslip argument
    (one panel's effective incidence goes up, the other's goes down)
    integrated over the real chord and dihedral distribution rather
    than collapsed to the textbook constant for a rectangular wing
  * the wing-sweep terms carry published DATCOM-shaped coefficients and
    are the weakest part; they are here because omitting them entirely
    would be worse, not because they are trustworthy to two figures

So: use this to REJECT designs that are clearly unstable, and to rank
one configuration against another. Do not read an absolute Dutch-roll
frequency off it.

Sign convention: body axes, positive Cn_beta is stable in yaw (the nose
swings into the relative wind), negative Cl_beta is stable in roll
(sideslip rolls the aircraft away from the slip). Derivatives are per
RADIAN of sideslip.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FIN_LIFT_SLOPE = 2.2
"""Lift-curve slope of an upturned tip, per radian.

Not 2*pi: a winglet is a very low aspect-ratio surface and its slope
falls off accordingly. 2.0-2.5 covers the aspect ratios these tips
actually reach."""

FIN_DYNAMIC_PRESSURE_RATIO = 0.95
"""A tip-mounted surface sits in clean air, unlike a fuselage-mounted
fin living in the wing's wake. This is close to 1 on purpose."""

def _wing_slope(ar: float) -> float:
    """3D lift-curve slope, per radian.

    The strip argument for dihedral effect wants the slope the WING
    actually has, not the section's 2*pi. Checked against the textbook
    closed form: with constant chord and constant dihedral the integral
    below collapses to -a*Gamma/4, and using 2*pi there puts Cl_beta
    about 30% above published values for the same geometry. The finite
    span is not optional.
    """
    return 2.0 * np.pi * ar / (ar + 2.0)


@dataclass(frozen=True)
class LateralState:
    cn_beta: float          # per rad, positive = directionally stable
    cl_beta: float          # per rad, negative = roll-stable in sideslip
    cn_beta_fin: float
    cn_beta_sweep: float
    cl_beta_dihedral: float
    cl_beta_sweep: float
    side_area_m2: float
    fin_arm_m: float

    @property
    def stable(self) -> bool:
        return self.cn_beta > 0.0 and self.cl_beta < 0.0

    @property
    def roll_yaw_ratio(self) -> float:
        """|Cl_beta / Cn_beta|. The handling number.

        Too low and the aircraft yaws without rolling -- it skids, and
        a tailless wing with no fin skids a lot. Too high and roll
        outruns yaw, which is Dutch roll: the wing rocks and the nose
        lags it. Conventional aircraft live around 1; the tolerable band
        is wide and the ends of it are genuinely unpleasant."""
        return abs(self.cl_beta) / max(abs(self.cn_beta), 1e-9)

    def report(self) -> str:
        v = "stable" if self.stable else "UNSTABLE"
        return "\n".join([
            f"lateral: Cn_beta {self.cn_beta:+.4f}/rad "
            f"(fin {self.cn_beta_fin:+.4f} + sweep {self.cn_beta_sweep:+.4f}) "
            f"-- {v}",
            f"         Cl_beta {self.cl_beta:+.4f}/rad "
            f"(dihedral {self.cl_beta_dihedral:+.4f} + "
            f"sweep {self.cl_beta_sweep:+.4f})",
            f"         roll/yaw ratio {self.roll_yaw_ratio:.2f} | "
            f"side area {self.side_area_m2*1e4:.1f} cm^2 at "
            f"{self.fin_arm_m*1000:.0f} mm arm",
        ])


def _spanwise(plan, n: int = 80):
    """Sample the half-wing: y, chord, dihedral angle, and the x of the
    section's quarter chord. One pass, reused by every term."""
    eta = np.linspace(0.0, 1.0, n)
    y = eta * plan.half_span_m
    chord = np.empty(n)
    z = np.empty(n)
    x_qc = np.empty(n)
    for i, e in enumerate(eta):
        st = plan.at(float(e))
        chord[i] = st.chord_m
        z[i] = st.z_le_m
        x_qc[i] = st.x_le_m + 0.25 * st.chord_m
    # local dihedral from the actual lofted leading edge
    dz_dy = np.gradient(z, y, edge_order=2)
    gamma = np.arctan(dz_dy)
    return y, chord, z, gamma, x_qc


def side_area(plan, x_cg_m: float, n: int = 80) -> tuple[float, float]:
    """Projected side area of the wing itself, and its x arm from the CG.

    A wing element of chord c spanning dy, tilted up by the local
    dihedral gamma, has surface area c*dy/cos(gamma) and throws a shadow
    of c*dy*tan(gamma) on the x-z plane. That shadow is what resists
    sideslip. Integrating it means a winglet, a raked tip and plain
    dihedral are all measured the same way -- there is no separate
    'winglet' object to special-case, because on this aircraft there
    isn't one: a winglet here IS the last segment's dihedral.

    Both halves count. They tilt the same way, so their side areas add.
    """
    y, chord, _z, gamma, x_qc = _spanwise(plan, n)
    strip = chord * np.abs(np.tan(gamma))
    area = 2.0 * float(np.trapezoid(strip, y))
    if area < 1e-9:
        return 0.0, 0.0
    x_bar = float(np.trapezoid(strip * x_qc, y) / np.trapezoid(strip, y))
    return area, x_bar - x_cg_m


def cn_beta(plan, x_cg_m: float, cl: float,
            n: int = 80) -> tuple[float, float, float]:
    """Directional stiffness: (total, fin part, sweep part)."""
    s_ref = plan.area_m2
    b = plan.span_m
    s_v, arm = side_area(plan, x_cg_m, n)
    # fin volume coefficient x slope. Arm positive = area BEHIND the CG,
    # which is the stabilising direction.
    cn_fin = (s_v * arm) / max(s_ref * b, 1e-12) \
        * FIN_LIFT_SLOPE * FIN_DYNAMIC_PRESSURE_RATIO

    # Wing sweep. The windward panel of a swept wing meets the flow at a
    # lower effective sweep, so it makes more lift AND more drag than the
    # leeward one; the drag difference is a yawing moment. It scales with
    # CL^2 because it is a drag asymmetry, not a lift one.
    ar = plan.aspect_ratio
    lam = np.radians(plan.sweep_quarter_chord_deg())
    cn_sweep = cl ** 2 * (1.0 / (4.0 * np.pi * ar)
                          - np.tan(lam) / (np.pi * ar * (ar + 4.0 * np.cos(lam)))
                          * (np.cos(lam) - 0.5 * ar
                             - ar ** 2 / (8.0 * np.cos(lam))))
    return float(cn_fin + cn_sweep), float(cn_fin), float(cn_sweep)


def cl_beta(plan, cl: float, n: int = 80) -> tuple[float, float, float]:
    """Roll-due-to-sideslip: (total, dihedral part, sweep part).

    The dihedral term is the textbook argument done on the real
    geometry. In sideslip the crossflow component adds incidence to the
    panel that is tilted into it and removes it from the other, by
    d(alpha) = gamma * beta locally. That incidence difference makes a
    lift difference, and the lift difference times its moment arm y is a
    rolling moment:

        Cl_beta = -(2 a0 / (S b)) * INTEGRAL gamma(y) c(y) y dy

    Collapsing gamma and c to constants recovers the familiar -a0*Gamma/4
    for a rectangular wing; doing the integral instead means polyhedral
    and a winglet are counted where they actually are, far outboard,
    where y is largest and they matter most.
    """
    y, chord, _z, gamma, _x = _spanwise(plan, n)
    integ = float(np.trapezoid(gamma * chord * y, y))
    a0 = _wing_slope(plan.aspect_ratio)
    cl_dih = -(2.0 * a0 / max(plan.area_m2 * plan.span_m, 1e-12)) * integ

    # Sweep also rolls the aircraft in sideslip, for the same reason it
    # yaws it: the windward panel is effectively less swept and lifts
    # more. Published fits put it near -CL*sin(2*Lambda)/8 for moderate
    # aspect ratios; this is the calibrated part of the model.
    lam = np.radians(plan.sweep_quarter_chord_deg())
    cl_sweep = -cl * np.sin(2.0 * lam) / 8.0
    return float(cl_dih + cl_sweep), float(cl_dih), float(cl_sweep)


def analyse(plan, x_cg_m: float, cl: float, n: int = 80) -> LateralState:
    cn, cn_f, cn_s = cn_beta(plan, x_cg_m, cl, n)
    clb, cl_d, cl_s = cl_beta(plan, cl, n)
    s_v, arm = side_area(plan, x_cg_m, n)
    return LateralState(cn_beta=cn, cl_beta=clb, cn_beta_fin=cn_f,
                        cn_beta_sweep=cn_s, cl_beta_dihedral=cl_d,
                        cl_beta_sweep=cl_s, side_area_m2=s_v, fin_arm_m=arm)
