"""Lateral-directional dynamics: does the Dutch roll actually die out?

The roll/yaw ratio gate is a PROXY. It compares two static stiffnesses
and says nothing about damping, which is the only thing that decides
whether a wing that has started to rock stops rocking. This module
answers the real question -- the eigenvalues of the linearised lateral
equations -- and it exists because the proxy turned out to be wrong in
both directions at once:

  * the strip-theory dihedral term in lateral.py runs ~65% high against
    the lattice on a plain 5 degree dihedral wing (-0.098 against -0.059
    per radian), so every roll/yaw ratio was inflated; and
  * the gen3 trainer passed that gate at 10.8 while its Dutch roll is
    DIVERGENT (zeta -0.057): in the lattice its yaw stiffness is
    essentially zero, and no static number can see damping.

The model, and what each part rests on:

  * four states in stability axes -- sideslip, roll rate, yaw rate, bank
  * wing derivatives from the same vortex lattice that trims the
    aircraft, with sideslip and body rates imposed as onset flow and the
    induced velocity included in the forces (VLM.lateral_derivatives)
  * tip-fin derivatives, drag and mass from aero/fins.py
  * profile-drag yaw damping by strip theory,
        Cn_r += -4 INT y^2 c cd dy / (S b^2)
    which is -cd/3 for a rectangular wing, the textbook value
  * inertias from the mass distribution: shell spread along the span by
    chord x perimeter, payload on the centreline, servos at the elevons,
    spar spread spanwise, fins at the tips

Validated: the assembly reproduces Nelson's Navion lateral example to
under 1% on the Dutch roll and the roll mode; roll damping on an
elliptic wing is within 15% of lifting-line theory; the stability-axis
derivatives are an exact rotation of the body-axis ones.

The axis bookkeeping has already cost one wrong answer. The first
prototype took derivatives and inertias in BODY axes and fed them to
stability-axis equations; at the trainer's 7.8 degree trim that leaked
roll stiffness into yaw and reported every winner as violently
divergent (zeta -0.20). Everything here is stability axes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

G = 9.80665
RHO_AIR = 1.225


@dataclass(frozen=True)
class LateralModes:
    zeta_dr: float | None       # None: the Dutch roll is not oscillatory
    wn_dr: float | None         # rad/s
    spiral: float               # eigenvalue, 1/s; positive diverges
    roll: float                 # eigenvalue, 1/s; the roll subsidence
    eigenvalues: np.ndarray
    derivatives: np.ndarray     # 3x3 total, stability axes
    inertia_kg_m2: tuple[float, float, float]     # Ixx, Izz, Ixz

    @property
    def spiral_t2_s(self) -> float:
        """Time for a spiral divergence to double; inf if it converges."""
        return float(np.log(2.0) / self.spiral) if self.spiral > 1e-9 else float("inf")

    @property
    def dutch_roll_period_s(self) -> float:
        if self.zeta_dr is None or abs(self.zeta_dr) >= 1.0:
            return float("inf")
        return float(2.0 * np.pi / (self.wn_dr * np.sqrt(1.0 - self.zeta_dr ** 2)))

    def report(self) -> str:
        D = self.derivatives
        if self.zeta_dr is None:
            dr = "Dutch roll not oscillatory"
        else:
            state = "DIVERGENT" if self.zeta_dr < 0.0 else "damped"
            dr = (f"Dutch roll zeta {self.zeta_dr:+.3f} ({state}), "
                  f"wn {self.wn_dr:.2f} rad/s, period {self.dutch_roll_period_s:.2f} s")
        sp = ("spiral convergent" if self.spiral <= 1e-9
              else f"spiral DIVERGES, doubles in {self.spiral_t2_s:.1f} s")
        roll = (f"roll time constant {-1.0/self.roll:.3f} s" if self.roll < 0.0
                else "roll mode not convergent")
        return "\n".join([
            f"dynamics: {dr}",
            f"          {sp} (root {self.spiral:+.3f}/s) | {roll}",
            f"          CY_b {D[0,0]:+.3f} Cl_b {D[1,0]:+.3f} Cn_b {D[2,0]:+.4f} | "
            f"Cl_p {D[1,1]:+.3f} Cn_p {D[2,1]:+.4f} | Cl_r {D[1,2]:+.3f} "
            f"Cn_r {D[2,2]:+.4f}  (stability axes, per rad)",
        ])


def state_matrix(y_beta: float, y_p: float, y_r: float,
                 l_beta: float, l_p: float, l_r: float,
                 n_beta: float, n_p: float, n_r: float,
                 v: float, g: float = G) -> np.ndarray:
    """Dimensional lateral matrix; states (beta, p, r, phi), stability axes.

    Y terms are side-force accelerations per unit state; L and N are
    angular accelerations, already solved through the inertia matrix --
    the form Nelson tabulates, which is what lets his Navion example test
    this assembly directly."""
    return np.array([
        [y_beta / v, y_p / v, y_r / v - 1.0, g / v],
        [l_beta, l_p, l_r, 0.0],
        [n_beta, n_p, n_r, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ])


def classify(eigs: np.ndarray) -> tuple[float | None, float | None, float, float]:
    """(zeta_dr, wn_dr, spiral, roll) from the four eigenvalues."""
    osc = [z for z in eigs if z.imag > 1e-6]
    real = [float(z.real) for z in eigs if abs(z.imag) <= 1e-6]
    zeta = wn = None
    if osc:
        z = max(osc, key=lambda c: c.imag)
        wn = float(abs(z))
        zeta = float(-z.real / wn)
    spiral = min(real, key=abs) if real else float("nan")
    roll = min(real) if real else float("nan")
    return zeta, wn, spiral, roll


def solve_modes(mass_kg: float, v_ms: float, s_ref_m2: float, span_m: float,
                ixx: float, izz: float, ixz: float,
                derivatives: np.ndarray) -> LateralModes:
    D = np.asarray(derivatives, dtype=float)
    q = 0.5 * RHO_AIR * v_ms ** 2
    k = span_m / (2.0 * v_ms)
    Y = q * s_ref_m2 / mass_kg * np.array([D[0, 0], D[0, 1] * k, D[0, 2] * k])
    L = q * s_ref_m2 * span_m * np.array([D[1, 0], D[1, 1] * k, D[1, 2] * k])
    N = q * s_ref_m2 * span_m * np.array([D[2, 0], D[2, 1] * k, D[2, 2] * k])
    Lp, Np = np.linalg.solve(np.array([[ixx, -ixz], [-ixz, izz]]), np.array([L, N]))
    eigs = np.linalg.eigvals(state_matrix(*Y, *Lp, *Np, v_ms))
    zeta, wn, spiral, roll = classify(eigs)
    return LateralModes(zeta, wn, spiral, roll, eigs, D, (ixx, izz, ixz))


def profile_yaw_damping(plan, cd0: float, n: int = 81) -> float:
    """Yaw damping from section drag, by strip theory, both halves.

    Yawing moves one half forward faster than the other; drag goes as the
    square of local speed, so the faster half drags more and the moment
    opposes the rate. -cd/3 for a rectangular wing."""
    y = np.linspace(0.0, plan.half_span_m, n)
    c = np.array([plan.at(float(v / plan.half_span_m)).chord_m for v in y])
    return float(-4.0 * 2.0 * np.trapezoid(y ** 2 * c * cd0, y)
                 / (plan.area_m2 * plan.span_m ** 2))


def inertia(plan, mass, alpha_deg: float, servo_eta: float, fins=None,
            n: int = 41) -> tuple[float, float, float, float]:
    """(Ixx, Izz, Ixz, z_cg) about the CG, in stability axes.

    MassBudget records where each item sits fore and aft, not where it
    sits along the span, so this places them: the printed shell along the
    span in proportion to chord x perimeter (what the vase-mode bead
    length actually follows), the spar spread along 92% of the span,
    servos at the middle of the elevon, fins at the tips, and everything
    else on the centreline. Roll inertia is dominated by the shell and
    the fins, which are the two placed most carefully."""
    half = plan.half_span_m
    stations = [plan.at(float(e)) for e in np.linspace(0.0, 1.0, n)]
    w = np.array([s.chord_m * s.airfoil.perimeter for s in stations])
    w = w / w.sum()
    pts = []
    for s, wi in zip(stations, w):
        for sg in (1.0, -1.0):
            pts.append((0.5 * mass.shell_kg * wi, s.x_le_m + 0.42 * s.chord_m,
                        sg * s.eta * half, s.z_le_m))
    y_servo = (servo_eta + 0.5 * (1.0 - servo_eta)) * half
    for it in mass.items:
        name = it.name.lower()
        if "spar" in name:
            for e in np.linspace(0.0, 0.92, 12):
                for sg in (1.0, -1.0):
                    pts.append((it.mass_kg / 24.0, it.x_m, sg * e * half, it.z_m))
        elif "servo" in name:
            for sg in (1.0, -1.0):
                pts.append((it.mass_kg / 2.0, it.x_m, sg * y_servo, it.z_m))
        elif "fin" in name and fins is not None:
            xf, zf = fins.centroid()
            for sg in (1.0, -1.0):
                pts.append((it.mass_kg / 2.0, xf, sg * fins.y_m, zf))
        else:
            pts.append((it.mass_kg, it.x_m, 0.0, it.z_m))
    P = np.array(pts)
    m = P[:, 0]
    total = m.sum()
    xc = (m * P[:, 1]).sum() / total
    zc = (m * P[:, 3]).sum() / total
    a = np.radians(alpha_deg)
    xb, zb = -(P[:, 1] - xc), -(P[:, 3] - zc)            # body: x fwd, z down
    xs, zs = xb * np.cos(a) + zb * np.sin(a), -xb * np.sin(a) + zb * np.cos(a)
    y = P[:, 2]
    return (float((m * (y ** 2 + zs ** 2)).sum()), float((m * (xs ** 2 + y ** 2)).sum()),
            float((m * xs * zs).sum()), float(zc))


def analyse(vlm, plan, mass, alpha_deg: float, x_cg_m: float, v_ms: float,
            cd0_wing: float, servo_eta: float, fins=None) -> LateralModes:
    """Everything above, for one trimmed design."""
    ixx, izz, ixz, z_cg = inertia(plan, mass, alpha_deg, servo_eta, fins)
    ref = np.array([x_cg_m, 0.0, z_cg])
    D = np.array(vlm.lateral_derivatives(alpha_deg, ref), dtype=float)
    D[2, 2] += profile_yaw_damping(plan, cd0_wing)
    if fins is not None:
        D = D + fins.derivatives(plan.area_m2, plan.span_m, x_cg_m, z_cg, alpha_deg)
        # fin skin friction acts at the tips, y = b/2, where the strip
        # formula above reduces to exactly -CD_fin
        D[2, 2] -= fins.cd0(v_ms, plan.area_m2)
    return solve_modes(mass.total_kg, v_ms, plan.area_m2, plan.span_m,
                       ixx, izz, ixz, D)
