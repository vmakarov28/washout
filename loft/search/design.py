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

from ..geom.cst import (Airfoil, deflect_te, flap_effectiveness,
                        scale_camber, scale_thickness_ratio,
                        set_thickness_peak)
from ..geom.planform import Planform, bwb
from ..printing import vase
from ..aero import performance as perf
from .. import propulsion as prop
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

    name: str = "bwb"
    span_m: float = 1.0
    span_free: bool = False
    """Let the optimizer choose the span. Used by the micro mission,
    where the whole question IS how small the aircraft can get around a
    fixed motor and receiver."""
    objective: str = "ld"
    """What 'best' means: 'ld' cruise efficiency, 'speed' outright
    velocity, 'small' the smallest flyable aeroplane. A trainer and a
    racer are not the same design scored differently -- they are
    different scores."""
    payload: tuple[Item, ...] = ()
    bays: tuple[Bay, ...] = ()
    cruise_band_ms: tuple[float, float] = (12.0, 20.0)
    min_static_margin: float = 0.05
    max_static_margin: float = 0.18
    cl_max_section: float = 1.05
    max_mass_kg: float = 0.90
    require_printable: bool = True
    max_wing_loading_gdm2: float = 1e9
    """Wing loading is the single number a beginner feels. Low loading
    means a low stall speed, a long time to react, and a survivable
    arrival; it is worth more to a first-time pilot than any amount of
    L/D."""
    battery_kg: float = 0.190
    """The pack's mass, which belongs to the MISSION, not to the module.

    It was a module-level constant, so a 900 mm trainer was silently
    charged for the 1 m FPV wing's 4S 1500 -- 190 g on a 340 g aircraft.
    Wing loading then failed every candidate and the mission looked
    infeasible when it was only mis-specified. The pack's box lives in
    `bays`; its mass lives here; both describe the same object."""
    tip_stall_margin: float = 0.0
    """How far the tip section must stay below the peak section cl, as a
    fraction. A wing whose TIPS stall first drops a wingtip and departs
    into a spin -- the classic way a beginner loses a flying wing on its
    first launch. Loading the root harder than the tip means the centre
    lets go first, the nose drops, and the aircraft recovers itself.
    This is the most important safety property in the whole file."""
    n_limit_g: float = 3.5
    spar_d_mm: float = 4.0
    """Spar the bore gate must clear. Belongs to the MISSION: micro's
    tip panel is a few millimetres thick and cannot swallow the 4 mm tube
    a 900 mm trainer wants, which is the single gate that rejected an
    otherwise complete 363 mm aeroplane."""
    powertrain: object | None = None
    min_thrust_weight: float = 0.0
    min_elevon_power: float = 0.004
    max_elevon_power: float = 0.030
    max_elevon_deflect_deg: float = 12.0
    """How far the pilot can pull the elevons before the flow gives up.
    Beyond about 15 degrees a plain hinged surface separates and the
    extra deflection buys moment it cannot cash."""
    max_overhang_deg: float | None = None
    """Per-mission override of the printing overhang limit. A racer is
    printed once, carefully, with good cooling; a trainer is printed by
    someone who has not tuned their machine."""
    """Static thrust over weight, for the hand launch. 0.5 will fly off a
    gentle throw; a racer wants 1.0 or better."""
    max_trim_alpha_deg: float = 90.0
    """Cap on the angle of attack the aircraft settles at, hands off.

    The lattice is INVISCID: its lift curve rises for ever and the
    tier-0 drag model has no alpha dependence, so trimming at a huge
    angle costs the optimizer nothing and it will duly do it -- the
    trainer search returned a design that cruises at +14.3 deg, which a
    12% section at Re 60k has long since stopped flying at.

    This is a design choice as much as a model patch: a trainer should
    cruise with real margin to the stall, so that the first time a
    beginner pulls back, the aeroplane still has somewhere to go."""

    # ---------------------------------------------------------- the fleet
    #
    # Every aircraft here flies the SAME 2205 2300 kv motor and the SAME
    # Spektrum AR630 receiver, and prints in Bambu PLA Aero on an X1C.
    # Those are not incidental: a 36 g motor and an 8 g receiver are a
    # fixed tax, and on the micro that tax IS the design problem.

    @staticmethod
    def _common(servo_g: float = 0.018, esc_g: float = 0.020,
                spar_g: float = 0.030) -> tuple:
        return (
            Item("2205 2300kv + prop", 0.036, 0.97),   # pusher, at the TE
            Item("AR630 rx", 0.008, 0.34),
            Item("esc + wiring", esc_g, 0.50),
            Item("servos x2", servo_g, 0.72),
            Item("spar + joiners", spar_g, 0.30),
        )

    @staticmethod
    def trainer_v3() -> "Mission":
        """Slow, self-righting, hard to hurt yourself with. Under 1 m.

        Judged against one question: what happens when a first-time pilot
        lets go of the sticks? It has to fly away straight and slow down.
        That needs a big static margin (nose-down when it speeds up), real
        dihedral (rolls level on its own), low wing loading (slow, and
        gentle when it lands), and a root that stalls before the tips.

        The 8 degree trim cap is MEASURED, not guessed: the tunnel polar
        of this section at Re 60k peaks in L/D at 8 degrees and loses 40%
        of it by 14, while lift is still rising. v2 was capped at 10 on a
        hunch and duly trimmed at 9.95, hard against the constraint."""
        return Mission(
            name="trainer_v3", span_m=0.90, objective="ld",
            payload=Mission._common(),
            bays=(Bay("3S 1300", 0.28, (72.0, 35.0, 24.0), x_var="batt_x"),
                  Bay("AR630 + esc", 0.42, (40.0, 34.0, 16.0))),
            battery_kg=0.110,
            cruise_band_ms=(7.0, 11.0),
            min_static_margin=0.15, max_static_margin=0.32,
            cl_max_section=0.85, max_mass_kg=0.50,
            max_wing_loading_gdm2=26.0, tip_stall_margin=0.12,
            n_limit_g=3.0, max_trim_alpha_deg=8.0,
            spar_d_mm=4.0, powertrain=prop.trainer_power(),
            min_thrust_weight=0.55,
            min_elevon_power=0.004, max_elevon_power=0.022,
        )

    @staticmethod
    def demon1() -> "Mission":
        """Everything for speed. Nothing for comfort.

        The objective is outright velocity, not efficiency, and that
        inverts most of the trainer's choices: minimal static margin
        (a stable aircraft wastes lift trimming itself), high wing
        loading (small wing, high speed), thin low-camber sections, and a
        trim angle held down near the drag bucket because at 40 m/s
        profile drag is the entire budget.

        The load factor is 6 g rather than 3: this thing gets pulled out
        of dives, and the spar is sized for it."""
        return Mission(
            name="demon1", span_m=0.80, objective="speed",
            payload=Mission._common(esc_g=0.026, spar_g=0.038),
            # A 4S 1300 is a big pack for an 800 mm racer: 76 mm long
            # needs a 463 mm root chord to sit at 0.10c without hanging
            # off the nose, and the bound is 460. It missed by 3 mm and
            # that one gate sank the whole search. A 4S 850 is the pack
            # this aircraft would actually fly.
            bays=(Bay("4S 850", 0.30, (65.0, 34.0, 24.0), x_var="batt_x"),
                  Bay("AR630 + esc", 0.44, (42.0, 34.0, 16.0))),
            battery_kg=0.105,
            cruise_band_ms=(20.0, 48.0),
            min_static_margin=0.04, max_static_margin=0.16,
            cl_max_section=1.00, max_mass_kg=0.85,
            max_overhang_deg=56.0,       # printed once, carefully
            max_wing_loading_gdm2=1e9,      # loading is the POINT here
            tip_stall_margin=0.04,
            n_limit_g=6.0, max_trim_alpha_deg=5.0,
            spar_d_mm=4.0, powertrain=prop.demon_power(),
            min_thrust_weight=1.00,      # it has to leave the hand hard
            min_elevon_power=0.003, max_elevon_power=0.016,
        )

    @staticmethod
    def micro() -> "Mission":
        """The smallest aeroplane that can carry the same motor and rx.

        Span is a DESIGN VARIABLE here, because the whole question is how
        small it can get. And the answer is dominated by a tax it cannot
        negotiate: 36 g of motor and 8 g of receiver, before any wing
        exists. Shrink the span and wing area falls as span squared while
        that 44 g does not move, so wing loading runs away and the stall
        speed with it. Somewhere there is a smallest span that still
        flies slowly enough to land, and finding it is the mission."""
        return Mission(
            name="micro", span_m=0.55, span_free=True, objective="small",
            payload=Mission._common(servo_g=0.010, esc_g=0.012, spar_g=0.014),
            bays=(Bay("2S 450", 0.30, (55.0, 30.0, 17.0), x_var="batt_x"),
                  Bay("AR630", 0.44, (30.0, 20.0, 12.0))),
            battery_kg=0.028,
            cruise_band_ms=(8.0, 17.0),
            min_static_margin=0.10, max_static_margin=0.26,
            cl_max_section=0.90, max_mass_kg=0.22,
            max_wing_loading_gdm2=42.0, tip_stall_margin=0.08,
            n_limit_g=4.0, max_trim_alpha_deg=9.0,
            # 2.5 mm carbon ROD, not a tube: micro's tip panel is a few
            # millimetres thick and the 4 mm tube a 900 mm trainer wants
            # is the single gate that rejected an otherwise complete
            # 363 mm aeroplane.
            spar_d_mm=2.5, powertrain=prop.micro_power(),
            min_thrust_weight=0.75,
            min_elevon_power=0.004, max_elevon_power=0.026,
        )

    @staticmethod
    def beginner_trainer() -> "Mission":
        return Mission.trainer_v3()

# --------------------------------------------------------- the design vector


@dataclass(frozen=True)
class Bound:
    name: str
    lo: float
    hi: float
    unit: str = ""


PLANFORM_BOUNDS = (
    Bound("span_m", 0.35, 1.10, "m"),
    Bound("root_chord", 0.13, 0.46, "m"),
    Bound("body_eta", 0.10, 0.34, ""),
    Bound("body_chord_frac", 0.70, 0.99, ""),
    # kink is placed as a FRACTION of the span left outboard of the body,
    # never as an absolute eta, so body_eta < kink_eta < 1 holds for every
    # point in the unit cube. Ordering constraints the optimizer can
    # violate become geometry exceptions it must be penalised for; this
    # way the constraint cannot be expressed at all.
    Bound("kink_gap", 0.18, 0.62, ""),
    Bound("kink_chord_frac", 0.34, 0.86, ""),
    Bound("tip_chord_frac", 0.16, 0.55, ""),
    Bound("sweep_body", 20.0, 66.0, "deg"),
    Bound("sweep_mid", 12.0, 50.0, "deg"),
    Bound("sweep_outer", 4.0, 42.0, "deg"),
    Bound("dihedral", 0.0, 10.0, "deg"),
    Bound("twist_body", -2.0, 3.0, "deg"),
    Bound("twist_kink", -4.0, 2.0, "deg"),
    Bound("twist_tip", -9.0, 1.0, "deg"),
    Bound("body_thickness", 1.10, 2.30, "x"),
    Bound("batt_x", 0.06, 0.72, "c_root"),
)

SECTION_BOUNDS = (
    Bound("reflex_deg", -1.0, 9.0, "deg"),
    Bound("camber_scale", 0.30, 1.70, "x"),
    # The SHAPE of the section, not just its camber. t/c trades drag
    # against internal volume and spar depth; the thickness peak trades a
    # gentle stall (forward) against lower drag and a deeper bay aft.
    Bound("t_over_c", 0.075, 0.165, ""),
    Bound("x_tmax", 0.20, 0.44, "c"),
    # Elevons. Checked as a CONSTRAINT rather than assumed: a 25% chord
    # elevon already recovers ~60% of the section lift slope, so a few
    # degrees moves trim a long way, and too much authority is twitchy
    # rather than safe.
    Bound("elevon_chord", 0.16, 0.34, "c"),
    Bound("elevon_eta", 0.30, 0.72, ""),
)
BATTERY_NAME = "battery"

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
    sec = scale_thickness_ratio(base, p["t_over_c"])
    sec = set_thickness_peak(sec, p["x_tmax"])
    tip = deflect_te(scale_camber(sec, p["camber_scale"]), p["reflex_deg"])
    span = p["span_m"] if mission.span_free else mission.span_m
    body_eta = p["body_eta"]
    kink_eta = body_eta + p["kink_gap"] * (1.0 - body_eta)
    return bwb(
        half_span_m=0.5 * span,
        root_chord_m=p["root_chord"],
        body_eta=body_eta,
        body_chord_frac=p["body_chord_frac"],
        kink_eta=kink_eta,
        kink_chord_frac=p["kink_chord_frac"],
        tip_chord_frac=p["tip_chord_frac"],
        sweep_body_deg=p["sweep_body"],
        sweep_mid_deg=p["sweep_mid"],
        sweep_outer_deg=p["sweep_outer"],
        dihedral_deg=p["dihedral"],
        twist_body_deg=p["twist_body"],
        twist_kink_deg=p["twist_kink"],
        twist_tip_deg=p["twist_tip"],
        root_airfoil=tip,
        tip_airfoil=tip,
        body_thickness_scale=p["body_thickness"],
        name=mission.name,
    )


# The ONE lattice resolution, used by the search and by the final report
# alike. It is not a tuning knob: the tailless trim solution converges far
# more slowly than lift does, because trim is where a small MOMENT
# difference is driven to zero. Measured on a converged design, cruise
# speed reads 13.5 m/s at 12x4, 13.0 at 16x4, 11.3 at 24x6 and settles at
# 10.6 by 32x8 -- a 27% error at the resolution the search first used.
# Searching at one resolution and reporting at another let an optimizer
# spend 7384 evaluations perfecting a design the final check rejected.
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
    items += (perf.PointMass(BATTERY_NAME, mission.battery_kg,
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
    # The bracket is a SOLVER detail, not a design constraint. A slow
    # heavily-cambered trainer legitimately trims near 10 deg, and the
    # old [-6, 12] window was quietly rejecting exactly the aircraft this
    # mission asks for -- 154 of 216 candidates in one scan. What makes a
    # high trim angle unacceptable is proximity to the stall, and that is
    # already gated properly by cl_max_section below.
    alpha = vlm.trim_alpha(mass.x_cg_m, bounds=(-10.0, 18.0))
    if alpha is None:
        reasons.append("no trim angle in [-10, 18] deg")
        return Evaluation(False, -1e5 - 10 * len(reasons), reasons=tuple(reasons),
                          plan=plan, mass=mass, static_margin=sm)
    pt = vlm.solve(alpha, mass.x_cg_m)

    if pt.CL <= 0.02:
        reasons.append(f"trims at CL {pt.CL:.3f}: cannot support itself")
        return Evaluation(False, -1e5, reasons=tuple(reasons), plan=plan, mass=mass)

    v = float(np.sqrt(2 * mass.total_kg * perf.G
                      / (perf.RHO_AIR * plan.area_m2 * pt.CL)))
    try:
        cd0 = drag.cd0(plan, v, aero_point=pt)     # measured: per-strip cl
    except TypeError:
        cd0 = drag.cd0(plan, v)                    # tier-0: flat
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

    if alpha > mission.max_trim_alpha_deg:
        reasons.append(f"trims at {alpha:.1f} deg, over "
                       f"{mission.max_trim_alpha_deg:.0f} deg")
        penalty += 6.0 * (alpha - mission.max_trim_alpha_deg)

    # --- can the powertrain actually hold this speed? ---
    if mission.powertrain is not None:
        drag_n = 0.5 * perf.RHO_AIR * v * v * plan.area_m2 * cd
        thrust_n = mission.powertrain.power_limited_thrust_n(v)
        if thrust_n < drag_n:
            reasons.append(
                f"needs {drag_n/perf.G*1000:.0f} gf at {v:.1f} m/s, "
                f"powertrain gives {thrust_n/perf.G*1000:.0f}")
            penalty += 25.0 * (drag_n - thrust_n) / max(drag_n, 1e-6)
        tw = (mission.powertrain.power_limited_thrust_n(0.0)
              / (mass.total_kg * perf.G))
        if tw < mission.min_thrust_weight:
            reasons.append(f"static thrust/weight {tw:.2f} below "
                           f"{mission.min_thrust_weight:.2f}")
            penalty += 20.0 * (mission.min_thrust_weight - tw)

    # --- elevon authority: enough to trim, not so much it is twitchy ---
    # Analytic rather than a second lattice solve: the flap-effectiveness
    # tau is a closed-form thin-aerofoil result and a VLM rebuild would
    # double the cost of every evaluation.
    tau = flap_effectiveness(p_vec["elevon_chord"])
    e_start = p_vec["elevon_eta"]
    ele_area = 0.0
    ele_arm = 0.0
    for e_lo, e_hi in zip(np.linspace(e_start, 1.0, 9)[:-1],
                          np.linspace(e_start, 1.0, 9)[1:]):
        st_e = plan.at(0.5 * (e_lo + e_hi))
        dA = st_e.chord_m * p_vec["elevon_chord"] * (e_hi - e_lo) * plan.half_span_m
        x_e = st_e.x_le_m + (1.0 - 0.5 * p_vec["elevon_chord"]) * st_e.chord_m
        ele_area += dA
        ele_arm += dA * (x_e - mass.x_cg_m)
    ele_arm = ele_arm / max(ele_area, 1e-9)
    # dCm per degree of elevon, both surfaces
    dcm_ddeg = (2.0 * ele_area / plan.area_m2) * (ele_arm / plan.mac_m)         * 2.0 * np.pi * tau * np.radians(1.0)
    if abs(dcm_ddeg) < mission.min_elevon_power:
        reasons.append(f"elevon dCm/ddeg {abs(dcm_ddeg):.4f} below "
                       f"{mission.min_elevon_power:.4f}")
        penalty += 40.0 * (mission.min_elevon_power - abs(dcm_ddeg))
    if abs(dcm_ddeg) > mission.max_elevon_power:
        reasons.append(f"elevon dCm/ddeg {abs(dcm_ddeg):.4f} above "
                       f"{mission.max_elevon_power:.4f} (twitchy)")
        penalty += 20.0 * (abs(dcm_ddeg) - mission.max_elevon_power)

    loading = mass.total_kg * 1000.0 / (plan.area_m2 * 100.0)     # g/dm^2
    if loading > mission.max_wing_loading_gdm2:
        reasons.append(f"wing loading {loading:.1f} > "
                       f"{mission.max_wing_loading_gdm2:.0f} g/dm2")
        penalty += 2.0 * (loading - mission.max_wing_loading_gdm2)

    # stall progression: the tip must be working LESS hard than the peak,
    # so the root gives up first and the nose drops instead of a wing.
    if mission.tip_stall_margin > 0.0:
        eta = np.abs(pt.y_strip) / plan.half_span_m
        outer = pt.cl_local[eta > 0.80]
        if len(outer) and cl_pk > 1e-6:
            tip_ratio = float(np.nanmax(outer)) / cl_pk
            if tip_ratio > 1.0 - mission.tip_stall_margin:
                reasons.append(
                    f"tip cl is {tip_ratio*100:.0f}% of peak -- tips stall "
                    f"first (need <= {(1-mission.tip_stall_margin)*100:.0f}%)")
                penalty += 30.0 * (tip_ratio - (1 - mission.tip_stall_margin))
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
    if mission.objective == "speed":
        # TOP speed, not hands-off trim speed. A tailless wing trims at
        # one CL with elevons neutral, but a pilot chasing speed holds
        # down-elevon, and the elevon moment shifts trim by
        #
        #     dCL = dCm / SM
        #
        # which follows from CL_trim = Cm0/SM. So the aircraft can be
        # pushed to a much lower CL than it settles at -- and then the
        # limit is the powertrain, not the wing. Scoring hands-off trim
        # speed asked the optimizer for a wing that is fast when nobody
        # is flying it, which is not what a racer is.
        dcl_max = abs(dcm_ddeg) * mission.max_elevon_deflect_deg / max(sm, 1e-3)
        cl_min = max(pt.CL - dcl_max, 0.02)
        v_elevon = float(np.sqrt(2 * mass.total_kg * perf.G
                                 / (perf.RHO_AIR * plan.area_m2 * cl_min)))
        if mission.powertrain is not None:
            def drag_at(vv):
                cl_v = (2 * mass.total_kg * perf.G
                        / (perf.RHO_AIR * vv * vv * plan.area_m2))
                try:
                    cd_v = drag.cd_at(cl_v, plan.mac_m * vv / perf.NU_AIR)
                except AttributeError:
                    cd_v = cd0
                cdi_v = cl_v**2 / (np.pi * plan.aspect_ratio * 0.85)
                return 0.5 * perf.RHO_AIR * vv * vv * plan.area_m2 * (cd_v + cdi_v)
            # bracket from the trim speed: the aircraft demonstrably
            # flies there, so excess thrust is positive by construction
            v_thrust = mission.powertrain.top_speed_ms(
                drag_at, v_max=70.0, v_lo=v)
        else:
            v_thrust = v_elevon
        v_top = min(v_elevon, v_thrust)
        merit = v_top
    elif mission.objective == "small":
        # smallest flyable: span dominates, mass breaks ties. Negated
        # because the optimizer maximises.
        merit = -(plan.span_m * 100.0 + mass.total_kg * 10.0)
    else:
        merit = ld
    score = merit if not reasons else -(1000.0 + penalty)
    return Evaluation(
        ok=not reasons, score=float(score), ld=float(ld), v_cruise=v,
        mass_kg=mass.total_kg, cl_trim=float(pt.CL), static_margin=float(sm),
        reasons=tuple(reasons), plan=plan, trim=trim_state, mass=mass,
        panels=panels if want_panels else [],
    )
