"""The design vector, the mission it is judged against, and the score.

Span is a requirement, not a variable. You decide how big the aeroplane
is; the optimizer decides what shape it is at that size. Likewise the
payload: motor, battery and electronics are declared masses, so the
search cannot cheat by quietly building a lighter aircraft.

The physics of the score is worth stating plainly, because it drives
everything. A tailless wing with fixed elevons trims at ONE lift
coefficient, set by its camber, twist and CG -- not at whatever CL you
would like. So the trim solution FIXES CL, and the cruise speed follows
from wing loading:

    Cm(alpha) = 0  ->  CL_trim  ->  v = sqrt(2 W / (rho S CL_trim))

Cruise speed is therefore an output to be checked against the mission,
never an input to be assumed. A design that trims at CL 0.15 is not
"efficient", it is a 30 m/s aeroplane, and the mission band is what says
whether that is what you asked for.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from ..geom.cst import Airfoil, deflect_te, scale_camber
from ..geom.planform import Planform, bwb
from ..printing import vase
from ..aero import performance as perf
from ..aero.vlm import VLM


# ------------------------------------------------------------- the mission


@dataclass(frozen=True)
class Item:
    """A payload item positioned as a FRACTION of root chord.

    Fractions, not millimetres, because the optimizer resizes the body
    underneath these parts. A battery pinned to x = 75 mm is in the nose
    of a 340 mm chord and hanging off the front of a 210 mm one; a
    battery at 0.22c is in the same place on both."""

    name: str
    mass_kg: float
    x_frac: float
    box_mm: tuple[float, float, float] | None = None   # length, width, height

    def at(self, root_chord_m: float) -> perf.PointMass:
        return perf.PointMass(self.name, self.mass_kg,
                              self.x_frac * root_chord_m)


@dataclass(frozen=True)
class Bay:
    """A rigid box that must physically fit inside the shell.

    `x_frac` is a NOMINAL seat; the check sweeps chordwise to find the
    best station, because a pack can slide fore and aft. What it cannot
    do is get thinner."""

    name: str
    x_frac: float
    box_mm: tuple[float, float, float]        # length, width, height
    x_var: str | None = None
    """Design variable that sets this bay's seat, if it has one.

    The battery is both a mass at a position and a box that must fit --
    and they have to be the SAME position. Leaving the fit check free to
    look elsewhere let a search pass a design whose pack sat 19 mm aft of
    the root leading edge, in 5 mm of depth, hanging 19 mm off the nose,
    while the checker cheerfully measured a different station 61 mm back."""


def bay_fits(plan: Planform, bay: Bay, wall_mm: float,
             x_frac: float | None = None) -> tuple[bool, float, float]:
    """Does the box fit AT ITS SEAT? -> (ok, available_mm, needed_mm).

    Checked at one station -- the one the mass model uses -- and never
    swept for the most flattering one. A sweeping check answers "does
    some seat exist", which is not the question: the CG that trims the
    aircraft is computed from where the pack actually is. Sweeping passed
    a design whose battery sat 19 mm aft of the root leading edge, in
    5 mm of depth, hanging off the nose, while the checker measured a
    different station 61 mm back and reported 26.3 mm.

    The binding dimension is depth across the box's own WIDTH and along
    its full LENGTH. The centreline always flatters -- measuring there
    gave 23.2 mm for a bay that really had 17.7.
    """
    length, width, height = bay.box_mm
    root_c_mm = plan.stations[0].chord_m * 1000.0
    half_span_mm = plan.half_span_m * 1000.0
    eta_edge = min(0.5 * width / half_span_mm, 1.0)
    x_mm = (bay.x_frac if x_frac is None else x_frac) * root_c_mm

    worst = np.inf
    for eta in np.linspace(0.0, eta_edge, 5):
        st = plan.at(float(eta))
        c_mm = st.chord_m * 1000.0
        x0, x1 = (x_mm - 0.5 * length) / c_mm, (x_mm + 0.5 * length) / c_mm
        if x0 < 0.02 or x1 > 0.98:
            return False, 0.0, float(height)      # hangs off the nose or tail
        xs = np.linspace(x0, x1, 7)
        t = float(st.airfoil.thickness(xs).min()) * c_mm - 2.0 * wall_mm
        worst = min(worst, t)
    return bool(worst >= height), float(worst), float(height)


@dataclass(frozen=True)
class Mission:
    """What the aeroplane is for. Everything here is a constraint."""

    span_m: float = 1.0
    payload: tuple[Item, ...] = ()
    bays: tuple[Bay, ...] = ()
    cruise_band_ms: tuple[float, float] = (12.0, 20.0)
    min_static_margin: float = 0.05
    max_static_margin: float = 0.18
    cl_max_section: float = 1.05
    max_mass_kg: float = 0.90
    require_printable: bool = True

    @staticmethod
    def fpv_1m() -> "Mission":
        """A 1 m FPV flying wing: 2207 motor, 4S 1500, two elevon servos.

        Masses are the declared build; x positions are measured from the
        root leading edge and are where these parts have to go -- the
        battery forward for CG, the motor at the back for a pusher."""
        return Mission(
            span_m=1.0,
            payload=(
                # box_mm is what makes a payload REAL. Without it the
                # optimizer shrinks the centre body for free: a search
                # run produced a "feasible" 483 g aircraft whose battery
                # bay was 23.2 mm deep for a 26 mm pack. Mass without
                # volume is not a payload, it is a number.
                Item("motor+prop", 0.075, 0.98),   # pusher, at the TE
                Item("esc+wiring", 0.045, 0.50),
                Item("fc+rx+vtx", 0.055, 0.38),
                Item("servos x2", 0.024, 0.72),
                Item("spar+joints", 0.040, 0.30),
            ),
            bays=(Bay("battery 4S 1500", 0.269, (76.0, 35.0, 26.0),
                      x_var="batt_x"),
                  Bay("fc stack", 0.38, (40.0, 40.0, 18.0))),
            cruise_band_ms=(13.0, 22.0),
            max_mass_kg=0.90,
        )


# --------------------------------------------------------- the design vector


@dataclass(frozen=True)
class Bound:
    name: str
    lo: float
    hi: float
    unit: str = ""


PLANFORM_BOUNDS = (
    Bound("root_chord", 0.20, 0.42, "m"),
    Bound("kink_eta", 0.18, 0.50, ""),
    Bound("kink_chord_frac", 0.45, 0.85, ""),
    Bound("tip_chord_frac", 0.16, 0.45, ""),
    Bound("sweep_le", 20.0, 55.0, "deg"),
    Bound("kink_sweep_le", 10.0, 40.0, "deg"),
    Bound("dihedral", 0.0, 6.0, "deg"),
    Bound("twist_kink", -3.0, 1.0, "deg"),
    Bound("twist_tip", -7.0, 0.0, "deg"),
    Bound("body_thickness", 1.15, 2.10, "x"),
    # Where the battery sits is a DESIGN VARIABLE, not a constant. On a
    # tailless aircraft the CG is the single strongest lever on trim --
    # it decides the lift coefficient the wing settles at, and therefore
    # the cruise speed. Fixing it by hand and then optimising the wing
    # around it is solving the problem backwards; the pack is the one
    # part of the aircraft that is trivial to slide.
    Bound("batt_x", 0.06, 0.72, "c_root"),
)

BATTERY = Item("battery 4S", 0.190, 0.0)

# Section shape: two interpretable knobs, not six coefficient nudges.
# REFLEX is degrees of trailing-edge-up built into the moulded section --
# the thing that buys Cm0 and therefore lets a tailless wing trim at
# positive lift with the CG ahead of the neutral point. CAMBER_SCALE
# trades cruise lift against that moment. Both are monotone in the
# quantity they control, which raw CST coefficients emphatically are not.
SECTION_BOUNDS = (
    Bound("reflex_deg", -1.0, 8.0, "deg"),
    Bound("camber_scale", 0.4, 1.6, "x"),
)
BOUNDS = PLANFORM_BOUNDS + SECTION_BOUNDS
N_DIM = len(BOUNDS)


def unit_to_physical(u: np.ndarray) -> dict:
    u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
    return {b.name: b.lo + v * (b.hi - b.lo) for b, v in zip(BOUNDS, u)}


def physical_to_unit(p: dict) -> np.ndarray:
    return np.array([(p[b.name] - b.lo) / (b.hi - b.lo) for b in BOUNDS])


def build(u: np.ndarray, mission: Mission, base: Airfoil) -> Planform:
    """Design vector -> planform. Total function: every u in [0,1]^n
    produces geometry, valid or not. Validity is judged, not assumed."""
    p = unit_to_physical(u)
    tip = deflect_te(scale_camber(base, p["camber_scale"]), p["reflex_deg"])
    return bwb(
        half_span_m=0.5 * mission.span_m,
        root_chord_m=p["root_chord"],
        kink_eta=p["kink_eta"],
        kink_chord_frac=p["kink_chord_frac"],
        tip_chord_frac=p["tip_chord_frac"],
        sweep_le_deg=p["sweep_le"],
        kink_sweep_le_deg=p["kink_sweep_le"],
        dihedral_deg=p["dihedral"],
        twist_tip_deg=p["twist_tip"],
        twist_kink_deg=p["twist_kink"],
        root_airfoil=tip,
        tip_airfoil=tip,
        body_thickness_scale=p["body_thickness"],
        name="bwb",
    )


# The ONE lattice resolution, used by the search and by the final report
# alike. It is not a tuning knob: the tailless trim solution converges far
# more slowly than lift does, because trim is where a small MOMENT
# difference is driven to zero. Measured on a converged design, cruise
# speed reads 13.5 m/s at 12x4, 13.0 at 16x4, 11.3 at 24x6 and settles at
# 10.6 by 32x8 -- a 27% error at the resolution the search was originally
# using. Searching at one resolution and reporting at another let the
# optimizer spend 7384 evaluations perfecting a design that the final
# check then rejected. Same number everywhere, or the search is optimising
# a different aeroplane from the one being judged.
LATTICE_NS = 32
LATTICE_NC = 8


# ------------------------------------------------------------- evaluation


@dataclass
class Evaluation:
    ok: bool
    score: float
    ld: float = 0.0
    v_cruise: float = 0.0
    mass_kg: float = 0.0
    cl_trim: float = 0.0
    static_margin: float = 0.0
    reasons: tuple[str, ...] = ()
    plan: Planform | None = None
    trim: perf.TrimState | None = None
    mass: perf.MassBudget | None = None
    panels: list = field(default_factory=list)

    def line(self) -> str:
        if not self.ok:
            return f"  rejected: {'; '.join(self.reasons)}"
        return (f"  L/D {self.ld:5.2f} | v {self.v_cruise:5.1f} m/s | "
                f"{self.mass_kg*1000:4.0f} g | CL {self.cl_trim:.3f} | "
                f"SM {self.static_margin:+.3f}")


def evaluate(
    u: np.ndarray,
    mission: Mission,
    base: Airfoil,
    settings: vase.PrintSettings,
    drag: perf.DragModel | None = None,
    ns: int = LATTICE_NS,
    nc: int = LATTICE_NC,
    want_panels: bool = False,
    z_step_mm: float | None = None,
) -> Evaluation:
    """One design -> one verdict. Cheap checks first, on purpose: the
    geometry test costs microseconds and rejects most of a random
    population before any lattice is ever built."""
    drag = drag or perf.DragModel()
    reasons: list[str] = []
    penalty = 0.0
    p_vec = unit_to_physical(u)
    try:
        plan = build(u, mission, base)
    except Exception as e:
        return Evaluation(False, -1e6, reasons=(f"geometry: {e}",))

    good, why = plan.is_valid()
    if not good:
        return Evaluation(False, -1e6, reasons=(why,), plan=plan)

    # --- printable? the shell mass comes out of this, so it runs early ---
    panels = vase.build_panels(plan, settings, z_step_mm=z_step_mm)
    checks = [vase.check(p) for p in panels]
    shell_kg = sum(p.mass_g() for p in panels) * 2.0 / 1000.0
    print_fail = [f"{p.name}: {','.join(c.failures())}"
                  for p, c in zip(panels, checks) if not c.ok]
    if mission.require_printable and print_fail:
        reasons.extend(print_fail)

    root_c = plan.stations[0].chord_m
    items = tuple(i.at(root_c) for i in mission.payload)
    items += (perf.PointMass(BATTERY.name, BATTERY.mass_kg,
                             p_vec["batt_x"] * root_c),)
    mass = perf.MassBudget(shell_kg=shell_kg,
                           shell_x_m=perf.shell_centroid_x(plan),
                           items=items)
    if mass.total_kg > mission.max_mass_kg:
        reasons.append(f"mass {mass.total_kg*1000:.0f} g over "
                       f"{mission.max_mass_kg*1000:.0f} g")

    for bay in mission.bays:
        seat = p_vec[bay.x_var] if bay.x_var else None
        ok_bay, have, need = bay_fits(plan, bay, settings.extrusion_width_mm, seat)
        if not ok_bay:
            where = f" at {seat:.2f}c" if seat is not None else ""
            reasons.append(f"{bay.name} bay{where} {have:.1f} mm deep, "
                           f"needs {need:.0f}")
            penalty += 20.0 * (need - have) / need

    # --- trim fixes CL; CL fixes cruise speed ---
    vlm = VLM(plan, ns=ns, nc=nc)
    x_np = perf.neutral_point(vlm, plan)
    sm = (x_np - mass.x_cg_m) / plan.mac_m
    alpha = vlm.trim_alpha(mass.x_cg_m)
    if alpha is None:
        reasons.append("no trim angle in [-6, 12] deg")
        return Evaluation(False, -1e5 - 10 * len(reasons), reasons=tuple(reasons),
                          plan=plan, mass=mass, static_margin=sm)
    pt = vlm.solve(alpha, mass.x_cg_m)

    if pt.CL <= 0.02:
        reasons.append(f"trims at CL {pt.CL:.3f}: cannot support itself")
        return Evaluation(False, -1e5, reasons=tuple(reasons), plan=plan, mass=mass)

    v = float(np.sqrt(2 * mass.total_kg * perf.G
                      / (perf.RHO_AIR * plan.area_m2 * pt.CL)))
    cd0 = drag.cd0(plan, v)
    cd = cd0 + pt.CDi
    ld = pt.CL / cd

    # Violations are scored by HOW FAR out they are, not merely that they
    # happened. A design 0.5 m/s outside the cruise band must rank above
    # one 8 m/s outside it, or the optimizer sees a flat cliff instead of
    # a slope and never finds its way back into the feasible set.
    penalty = 0.0

    def band(value: float, lo: float, hi: float, scale: float) -> float:
        return max(lo - value, 0.0, value - hi) / scale

    if not (mission.min_static_margin <= sm <= mission.max_static_margin):
        reasons.append(f"static margin {sm:+.3f} outside "
                       f"[{mission.min_static_margin}, {mission.max_static_margin}]")
        penalty += 40.0 * band(sm, mission.min_static_margin,
                               mission.max_static_margin, 1.0)
    if not (mission.cruise_band_ms[0] <= v <= mission.cruise_band_ms[1]):
        reasons.append(f"cruise {v:.1f} m/s outside {mission.cruise_band_ms}")
        penalty += 1.5 * band(v, *mission.cruise_band_ms, 1.0)
    cl_pk = float(np.nanmax(pt.cl_local))
    if cl_pk > mission.cl_max_section:
        reasons.append(f"peak section cl {cl_pk:.2f} > {mission.cl_max_section}")
        penalty += 25.0 * (cl_pk - mission.cl_max_section)
    if mass.total_kg > mission.max_mass_kg:
        penalty += 30.0 * (mass.total_kg - mission.max_mass_kg) / mission.max_mass_kg

    # Printability failures are graded the same way, by the worst gate's
    # overshoot, so "3 mm too wide for the bed" beats "80 mm too wide".
    for p_stack, chk in zip(panels, checks):
        for g in chk.gates:
            if not g.passed and g.limit > 0:
                penalty += 8.0 * abs(g.value - g.limit) / g.limit

    trim_state = perf.TrimState(not reasons, alpha, pt.CL, cd, pt.CDi, cd0,
                                sm, x_np, mass.x_cg_m, v, cl_pk,
                                "; ".join(reasons) or "ok")

    # Feasibility DOMINATES (Deb's constraint-handling rule): any design
    # that meets the mission beats any design that does not, and among
    # the infeasible ones the ranking is by how badly they miss. A single
    # weighted sum does not work here and the failure is instructive --
    # with L/D and the penalty on one scale the optimizer discovered it
    # could park 1 m/s below the cruise band, pay 1.5 points, and buy
    # more than that back in L/D by flying slow. It was maximising the
    # score exactly as written; the score was wrong.
    score = ld if not reasons else -(100.0 + penalty)
    return Evaluation(
        ok=not reasons, score=float(score), ld=float(ld), v_cruise=v,
        mass_kg=mass.total_kg, cl_trim=float(pt.CL), static_margin=float(sm),
        reasons=tuple(reasons), plan=plan, trim=trim_state, mass=mass,
        panels=panels if want_panels else [],
    )
