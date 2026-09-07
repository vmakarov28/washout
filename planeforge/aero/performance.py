"""Mass, trim, stability and drag: does this shape actually fly?

A tailless aircraft has no second surface to argue with, so three numbers
decide whether a planform is an aeroplane or a frisbee, and all three are
computed here rather than assumed:

  trim      an angle of attack where Cm about the CG is zero, and it has
            to be an angle the wing can actually hold;
  stability the CG must sit ahead of the neutral point, by a margin;
  CL_trim   the lift needed to hold the aircraft up at that CG must be
            available before the sections stall.

The optimizer is allowed to fail all three. That is the point of
computing them -- an untrimmable design scores as untrimmable instead of
scoring well on L/D and killing itself on the first launch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..geom.planform import Planform
from .vlm import VLM, AeroPoint

RHO_AIR = 1.225        # kg/m^3, sea level
NU_AIR = 1.5e-5        # m^2/s
G = 9.80665


# ------------------------------------------------------------------- mass


@dataclass(frozen=True)
class PointMass:
    name: str
    mass_kg: float
    x_m: float          # aft of the aircraft nose datum (x=0 at root LE)
    z_m: float = 0.0


@dataclass
class MassBudget:
    """Shell mass comes from the vase-mode bead length -- a measured
    quantity, not a guess -- and everything else is a declared item."""

    shell_kg: float
    shell_x_m: float
    items: tuple[PointMass, ...] = ()

    @property
    def total_kg(self) -> float:
        return self.shell_kg + sum(i.mass_kg for i in self.items)

    @property
    def x_cg_m(self) -> float:
        m = self.shell_kg * self.shell_x_m + sum(i.mass_kg * i.x_m for i in self.items)
        return m / max(self.total_kg, 1e-9)

    def report(self) -> str:
        rows = [f"  {'shell (vase)':<16} {self.shell_kg*1000:7.0f} g  "
                f"x {self.shell_x_m*1000:6.1f} mm"]
        rows += [f"  {i.name:<16} {i.mass_kg*1000:7.0f} g  x {i.x_m*1000:6.1f} mm"
                 for i in self.items]
        rows.append(f"  {'TOTAL':<16} {self.total_kg*1000:7.0f} g  "
                    f"CG {self.x_cg_m*1000:6.1f} mm")
        return "mass budget:\n" + "\n".join(rows)


def shell_centroid_x(plan: Planform, n: int = 120) -> float:
    """Spanwise-weighted x of the skin, weighting each station by its
    perimeter -- where the printed mass actually sits, which for a swept
    BWB is well aft of the root and matters to the CG."""
    e = np.linspace(0.0, 1.0, n)
    w, x = [], []
    for v in e:
        st = plan.at(float(v))
        w.append(st.chord_m * st.airfoil.perimeter)
        x.append(st.x_le_m + 0.45 * st.chord_m)     # skin centroid ~0.45c
    w, x = np.array(w), np.array(x)
    y = e * plan.half_span_m
    return float(np.trapezoid(w * x, y) / np.trapezoid(w, y))


# ------------------------------------------------------- drag build-up


@dataclass(frozen=True)
class DragModel:
    """Profile drag by strip theory.

    Tier 0 (here) is a Prandtl-Schlichting flat plate with a thickness
    form factor and an explicit ROUGHNESS multiplier for the printed
    surface -- a vase-mode skin is a helix of 0.25 mm ridges running
    spanwise, and pretending it is polished aluminium would flatter every
    design by 20-30% of its profile drag.

    Tier 1 replaces cf_strip() wholesale with measured section polars
    from the LBM tunnel (see tunnel.py). The interface is one function so
    that swap is a one-line change, not a rewrite.
    """

    roughness_factor: float = 1.25
    transition_xc: float = 0.05     # printed LE trips the boundary layer early

    def cf(self, re: float) -> float:
        re = max(re, 1e4)
        return 0.455 / (np.log10(re) ** 2.58)       # Prandtl-Schlichting

    def form_factor(self, t_c: float) -> float:
        return 1.0 + 2.0 * t_c + 60.0 * t_c**4      # Hoerner

    def cd0(self, plan: Planform, v_ms: float, n: int = 40) -> float:
        e = np.linspace(0.0, 1.0, n)
        y = e * plan.half_span_m
        contrib = []
        for v in e:
            st = plan.at(float(v))
            re = st.chord_m * v_ms / NU_AIR
            s_wet = st.chord_m * st.airfoil.perimeter
            contrib.append(self.cf(re) * self.form_factor(st.airfoil.t_max) * s_wet)
        swet_cf = 2.0 * np.trapezoid(np.array(contrib), y)
        return float(self.roughness_factor * swet_cf / plan.area_m2)


# --------------------------------------------------------- trim & stability


@dataclass
class TrimState:
    trimmed: bool
    alpha_deg: float
    CL: float
    CD: float
    CDi: float
    CD0: float
    static_margin: float
    x_np_m: float
    x_cg_m: float
    v_ms: float
    cl_local_max: float
    reason: str = "ok"

    @property
    def LD(self) -> float:
        return self.CL / max(self.CD, 1e-9)


def neutral_point(vlm: VLM, plan: Planform, alphas=(0.0, 4.0)) -> float:
    """x of the neutral point: where dCm/dCL = 0.

    Taken from two VLM solves about a fixed datum -- the moment slope is
    linear in this inviscid model, so two points define it exactly and a
    finer sweep would only add cost."""
    a, b = vlm.solve(alphas[0], 0.0), vlm.solve(alphas[1], 0.0)
    dcl = b.CL - a.CL
    if abs(dcl) < 1e-9:
        return float("nan")
    # Cm about x=0 is -CL*x_cp/MAC; the NP is where the moment stops
    # changing with lift
    return float(-(b.Cm - a.Cm) / dcl * plan.mac_m)


def trim(
    plan: Planform,
    vlm: VLM,
    mass: MassBudget,
    drag: DragModel,
    v_ms: float,
    alpha_bounds: tuple[float, float] = (-6.0, 12.0),
    min_static_margin: float = 0.05,
    max_static_margin: float = 0.20,
    cl_max_section: float = 1.05,
) -> TrimState:
    """Find the level-flight trim point at a given airspeed, and judge it.

    Airspeed is an input, not an output: a design is asked to fly at the
    speed you want to fly it, and the honest failure mode is 'it cannot
    hold that CL', not a silently re-scaled cruise speed.
    """
    x_cg = mass.x_cg_m
    x_np = neutral_point(vlm, plan)
    sm = (x_np - x_cg) / plan.mac_m

    lo, hi = alpha_bounds
    f = lambda al: vlm.solve(al, x_cg).Cm
    f_lo, f_hi = f(lo), f(hi)
    if f_lo * f_hi > 0:
        return TrimState(False, float("nan"), 0, 0, 0, 0, sm, x_np, x_cg, v_ms,
                         0, "no trim angle in bounds (Cm never crosses zero)")
    for _ in range(40):                      # bisection: robust, and 40
        mid = 0.5 * (lo + hi)                # iterations is 1e-12 of the range
        if f(mid) * f_lo > 0:
            lo, f_lo = mid, f(mid)
        else:
            hi = mid
    alpha = 0.5 * (lo + hi)
    pt: AeroPoint = vlm.solve(alpha, x_cg)

    cd0 = drag.cd0(plan, v_ms)
    cd = cd0 + pt.CDi
    cl_needed = mass.total_kg * G / (0.5 * RHO_AIR * v_ms**2 * plan.area_m2)

    reason = "ok"
    if sm < min_static_margin:
        reason = f"static margin {sm:.3f} below {min_static_margin}"
    elif sm > max_static_margin:
        reason = f"static margin {sm:.3f} above {max_static_margin} (sluggish)"
    elif pt.CL < cl_needed - 1e-3:
        reason = (f"trims at CL {pt.CL:.3f} but needs {cl_needed:.3f} "
                  f"to hold {mass.total_kg*1000:.0f} g at {v_ms:.0f} m/s")
    elif float(np.nanmax(pt.cl_local)) > cl_max_section:
        reason = (f"section cl {float(np.nanmax(pt.cl_local)):.2f} exceeds "
                  f"cl_max {cl_max_section}")
    return TrimState(reason == "ok", float(alpha), float(pt.CL), float(cd),
                     float(pt.CDi), float(cd0), float(sm), float(x_np),
                     float(x_cg), v_ms, float(np.nanmax(pt.cl_local)), reason)


def stall_speed_ms(mass: MassBudget, plan: Planform, cl_max: float = 0.9) -> float:
    return float(np.sqrt(2 * mass.total_kg * G
                         / (RHO_AIR * plan.area_m2 * cl_max)))


def wing_loading_gm2(mass: MassBudget, plan: Planform) -> float:
    return mass.total_kg * 1000.0 / plan.area_m2
