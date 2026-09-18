"""The joints between panels, and between the two halves: bonded rings.

Panels butt together on the spar and are glued face to face. Until now
that joint did not exist in the model: the spar was sized for bending
and the bond carried whatever was left, unexamined. This module puts a
number on it.

At a joint station the bonded area is the ring of the single wall's
cross-section -- the section's perimeter times one bead -- because a
vase part's face IS one bead wide. The loads that ring has to carry
are the ones the spar does not:

  * the SHEAR outboard of the joint, which is the lift carried by
    everything beyond it at the limit load factor -- `structure.span_loads`
    already integrates it;
  * the TORQUE of that lift about the spar, from the offset between each
    strip's aerodynamic centre (a quarter chord) and the spar line, which
    the ring reacts as a Bredt shear flow around the enclosed area.

The section's own pitching moment is not included -- the lattice does not
resolve it per strip -- and the report says so. Both are compared with a
DECLARED allowable for adhesive on foamed PLA, deliberately far below
any published figure for the adhesive itself, because the foam is the
weak side and has not been tested here. The centre joint is the same
calculation at eta = 0, where the loads are largest and the ring is
biggest.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .printing.vase import Gate

BOND_ALLOWABLE_MPA = 1.0
"""Shear a CA or epoxy bond to foamed PLA is credited with. Declared."""
BOND_SAFETY = 2.0
"""Factor on the LIMIT loads (already n_limit g)."""


@dataclass(frozen=True)
class Joint:
    eta: float
    perimeter_mm: float
    enclosed_mm2: float
    ring_mm2: float
    shear_n: float
    torque_nmm: float
    tau_shear_mpa: float
    tau_torque_mpa: float

    @property
    def tau_mpa(self) -> float:
        return float(np.hypot(self.tau_shear_mpa, self.tau_torque_mpa))

    @property
    def margin(self) -> float:
        return BOND_ALLOWABLE_MPA / max(self.tau_mpa, 1e-12)


def _section(plan, eta: float, n: int = 121):
    st = plan.at(float(np.clip(eta, 0.0, 1.0)))
    pts = st.airfoil.coords(n) * st.chord_m * 1000.0
    q = np.roll(pts, -1, axis=0)
    per = float(np.linalg.norm(q - pts, axis=1).sum())
    area = 0.5 * abs(float((pts[:, 0] * q[:, 1] - q[:, 0] * pts[:, 1]).sum()))
    return per, area


def joint_at(plan, loads, eta: float, spar_x_frac: float,
             wall_mm: float) -> Joint:
    """One joint, from the span loads `structure.span_loads` computed."""
    y_j = eta * plan.half_span_m * 1000.0
    y, l = loads.y_mm, loads.lift_n_per_mm
    sel = y >= y_j
    shear = float(np.trapezoid(l[sel], y[sel])) if sel.sum() > 1 else 0.0
    # torque about the spar line from each strip's quarter-chord offset
    lever = np.array([(0.25 - spar_x_frac) * plan.at(float(e)).chord_m * 1000.0
                      for e in np.clip(y / max(y.max(), 1e-9), 0.0, 1.0)])
    torque = float(np.trapezoid((l * lever)[sel], y[sel])) if sel.sum() > 1 else 0.0
    per, area = _section(plan, eta)
    ring = per * wall_mm
    tau_s = shear / max(ring, 1e-9)
    tau_t = abs(torque) / (2.0 * max(area, 1e-9) * wall_mm)
    return Joint(float(eta), per, area, ring, shear, torque, tau_s, tau_t)


def gates(plan, loads, joint_etas, spar_x_frac: float,
          wall_mm: float) -> tuple[list[Joint], list[Gate]]:
    """Every joint the print has, root first. `joint_etas` are the panel
    spans; the interior boundaries and the centreline are the joints."""
    etas = [0.0] + [b for _, b in joint_etas[:-1]]
    joints, out = [], []
    for e in etas:
        j = joint_at(plan, loads, e, spar_x_frac, wall_mm)
        joints.append(j)
        out.append(Gate(f"joint bond at eta {e:.2f}",
                        j.margin >= BOND_SAFETY, j.margin, BOND_SAFETY, "x",
                        f"{j.ring_mm2:.0f} mm2 ring, {j.shear_n:.1f} N shear, "
                        f"{j.torque_nmm / 1000:.2f} N.m torque -> "
                        f"{j.tau_mpa:.3f} MPa vs {BOND_ALLOWABLE_MPA} MPa declared"))
    return joints, out


def report(joints) -> str:
    lines = ["joints: (bonded ring, one bead wide; section pitching moment "
             "not included)"]
    for j in joints:
        lines.append(f"  eta {j.eta:.2f}: ring {j.ring_mm2:5.0f} mm2 | shear "
                     f"{j.shear_n:5.1f} N | torque {j.torque_nmm / 1000:6.2f} N.m | "
                     f"{j.tau_mpa:.3f} MPa, margin {j.margin:.0f}x")
    return "\n".join(lines)
