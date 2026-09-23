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
from ..geom.planform import Planform, Segment, bwb, faired, lofted
from ..geom import fairness as fz
from ..geom import interior as it
from ..printing import vase
from ..printing import detours as det
from ..printing import parts as pmod
from ..printing import elevons as elv
from ..printing import inserts as ins_
from ..aero import performance as perf
from ..aero import lateral
from ..aero import dynamics as dyn
from ..aero import fins as fn
from .. import propulsion as prop
from .. import spars as sp
from .. import linkage as lkg
from .. import aeroelastic as ael
from .. import structure as struct
from .. import joints as jnt
from collections import OrderedDict
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
    x_lo: float = 0.10
    x_hi: float = 0.90
    """The chordwise band a SOLVER may seat this bay in, used only when
    `x_var` is None.

    Two classes of bay, and the difference is physical. The battery is the
    strongest single lever on trim, so the OPTIMIZER owns its seat and
    pays for it in static margin. The receiver and ESC do not move trim
    enough to be worth a design dimension -- they just have to go
    somewhere that exists -- so their seat is SOLVED, the same way the
    spar's chordwise station is solved, against whatever is already in the
    wing."""
    eta_frac: float | None = None
    eta_lo: float = 0.0
    eta_hi: float = 0.0
    """Where along the span the bay sits, and the band a solver may use.

    None means the centreline, which is where payload in a blended body
    belongs. A SERVO does not belong there: it has to sit next to the
    surface it drives, so its station is out in the wing and is solved in
    `[eta_lo, eta_hi]` along with its chordwise seat. A zero-width band
    means "not spanwise-solved", which keeps every existing bay exactly
    where it was."""
    open_from: str = "upper"
    """Which skin the opening is cut in.

    Every bay in the fleet opens from the UPPER skin, and that is not a
    styling choice. In vase mode an opening is a recess: exterior space,
    sealed from the wing's interior and from every other recess by one
    bead. A wire cannot pass from the receiver's pocket to a servo's
    through the wing -- there is no through. It runs in a channel cut
    into the surface, and a channel lives on one skin, so every pocket
    it joins has to be on that skin too. The servo therefore sits in the
    top of the wing with its arm up through the opening, the horn stands
    on the elevon's upper surface, and the belly stays clean for the
    landing and the CG mark. The lower skin is still a legal choice for
    a bay that needs no wiring."""
    drives_elevon: bool = False
    """Whether this bay holds the servos for the control surface.

    If it does, its output shaft must lie outboard of the hinge
    station, and that is GATED rather than imposed. On every aircraft in
    the fleet the solver put the servos inboard of the elevon they drive
    -- the trainer's at eta 0.30 with the elevon starting at 0.49 -- and
    `linkage.for_station` then built a hinge and a horn at a station
    where the trailing edge is not cut, so the four-bar that proves the
    deflection demon1's whole speed objective is scored from was solved
    on a surface that does not exist there. The pushrod runs chordwise
    from the shaft to a horn on the elevon; both ends have to be at the
    same station.

    Clamping the seat instead was tried, and it is the wrong shape of
    fix: it moved the trainer's servos a fifth of the span outboard, into
    the thin outer panels and through the TE spar's corridor, and the CG,
    the trim, the spar seats and the rib corridors all moved with them --
    silently, because a solver that relocates a part reports nothing.
    Penalties rank the infeasible; they do not rearrange the aircraft."""
    lidded: bool = True
    """Whether the opening gets a ledge and a printed lid. Payload does;
    a servo pocket does not -- the servo's arm comes up through the
    opening and the servo is taped in."""
    holds: tuple[str, ...] = ()
    """Payload item names that physically live inside this bay.

    Without this the bay and its contents were declared in different
    places and nothing noticed: the trainer's "AR630 + esc" bay sat at
    0.42c while the masses it contains, `AR630 rx` and `esc + wiring`,
    sat at 0.34c and 0.50c. Three stations for two objects in one box.
    That is the same mistake `x_var` was added to fix for the battery,
    still live for everything else, and it matters because the CG that
    trims the aircraft is computed from where the masses are said to be.

    Items named here take their x from the bay's final seat, solved or
    declared, so there is exactly one answer to 'where is the ESC'."""


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


def _box_outside_mm(plan, v, margin: float = det.BAND_MARGIN,
                    n: int = 5) -> float:
    """How far, in mm, a volume's band pokes out of the section anywhere
    over its OWN span. 0.0 when the box is inside everywhere."""
    worst = 0.0
    for e in np.linspace(v.eta0, v.eta1, n):
        st = plan.at(float(e))
        c_mm = st.chord_m * 1000.0
        x0, x1 = v.band(plan, float(e))
        worst = max(worst, (margin - x0) * c_mm, (x1 - (1.0 - margin)) * c_mm)
    return float(worst)

def seat_bays(plan, mission, p_vec, wall_mm: float, joint_etas=(),
              n_x: int = 33, n_eta: int = 13) -> tuple[list, dict, dict]:
    """Where every bay actually goes.

    -> ([Volume, ...], {name: x_frac}, {name: eta_frac})

    Two passes, in the physical order of who gets to choose:

    1. Bays the OPTIMIZER owns (`x_var`) take their declared seat. The
       battery is here because the CG is the strongest lever on trim and
       the search has to pay for where it puts the pack.
    2. Bays a SOLVER owns are then seated in their own band, at the
       station that fits the depth and clears everything already placed,
       breaking ties toward the nominal seat so the declared intent still
       means something.

    Solving rather than sweeping-to-check is the important distinction.
    `bay_fits`'s docstring warns against a sweeping CHECK, and rightly:
    answering "does some seat exist" while the mass is modelled somewhere
    else passed a design whose pack hung off the nose. Here the solved
    seat IS the seat -- it drives the bay volume, the clash gates and the
    x of every item the bay `holds` -- so there is one station per object,
    which is what that warning asks for.

    A bay with no clear seat anywhere in its band is returned at the best
    station found, with the miss left for the gates to price. Returning
    nothing, or silently moving it outside its band, would hide a real
    conflict: on the trainer, the pack and the electronics were declared
    in the same 16 mm of space and no value of `batt_x` in its whole
    range could separate them.
    """
    placed: list = []
    seats: dict = {}
    eta_seats: dict = {}

    def volume(bay, x, eta=None):
        # A top-opening bay's box rests on a FLOOR that sits one box-depth
        # below the upper skin, so the box occupies the TOP of the section
        # and is anchored there. Anchoring every box to the lower skin --
        # the first version -- reserved the bottom of the section for
        # nothing, which is exactly where the spar seat solver wanted to
        # put the TE spar under the pack; it pushed the tube out to 0.61c
        # instead. A pocket that opens downward really does sit against
        # the lower skin.
        return it.bay_volume(bay.name, x, bay.box_mm, plan,
                             offset_mm=wall_mm,
                             anchor=(it.UPPER if bay.open_from == "upper"
                                     else it.LOWER),
                             eta_frac=bay.eta_frac if eta is None else eta)

    for bay in mission.bays:
        if bay.x_var is None:
            continue
        x = float(p_vec[bay.x_var])
        seats[bay.name] = x
        eta_seats[bay.name] = bay.eta_frac
        placed.append(volume(bay, x))

    for bay in mission.bays:
        if bay.x_var is not None:
            continue
        etas = (np.linspace(bay.eta_lo, bay.eta_hi, n_eta)
                if bay.eta_hi > bay.eta_lo else (bay.eta_frac,))
        best = None
        for e in etas:
            for x in np.linspace(bay.x_lo, bay.x_hi, n_x):
                vol = volume(bay, float(x), e)
                ok, spare = vol.fits(plan, wall_mm)
                clash = sum(it.overlap_mm(vol, other, plan, wall_mm)
                            for other in placed)
                # A seat that crosses a print joint is a part in two
                # separate shells, so it is counted as interference here
                # rather than only reported later: the solver has a whole
                # band to choose from and should not need telling twice.
                first, last = it.straddles(vol, joint_etas)
                split = 0.0 if last <= first else 20.0
                # The BOX's own span must be inside the section: a rigid
                # box at an absolute station on a swept body is overtaken
                # by the leading edge some way outboard, and the opening
                # may narrow while it closes, but not while the box is in
                # it. Millimetres of box outside the section count as
                # interference.
                room = _box_outside_mm(plan, vol)
                # NOT a term for how far the servo's shaft falls short of
                # the elevon. That was tried, in the key and as a clamp,
                # and both are the same mistake: the seat that satisfies
                # it is 20% of the span outboard, in thinner section and
                # through the TE spar's corridor, and the CG, the trim,
                # the spar seats, the rib corridors and one panel's wall
                # clearance all move with it -- sixteen tests failed and
                # none of them was about servos. The shortfall is a GATE,
                # reported in millimetres. What satisfies it is the
                # elevon's own station, which the search owns.
                # And two openings side by side must leave a WALL between
                # them, not a fin. The rendered root section showed the
                # electronics bay seated 5 mm from the pack: a 5 mm wide,
                # 30 mm tall, two-bead sliver of skin standing between two
                # holes. Closer than a real wall counts as interference.
                thin = 0.0
                for other in placed:
                    if other.eta1 < vol.eta0 or other.eta0 > vol.eta1:
                        continue
                    gap = it.gap_mm(vol, other, plan)
                    if 0.0 <= gap < MIN_BAY_WALL_MM:
                        thin += MIN_BAY_WALL_MM - gap
                # Total millimetres of interference, then nearest the seat
                # the mission declared. SUMMED, not ranked: a clash and a
                # depth shortfall are both "millimetres of something that
                # does not fit", and ranking clash above depth made the
                # solver accept a 6 mm depth miss to dodge a 0.1 mm graze.
                key = (clash + max(-spare, 0.0) + split + thin + room,
                       abs(float(x) - bay.x_frac))
                if best is None or key < best[0]:
                    best = (key, float(x), vol, e)
        seats[bay.name] = best[1]
        eta_seats[bay.name] = best[3]
        placed.append(best[2])

    return placed, seats, eta_seats


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
    spars: tuple = ()
    """Spanwise spars to fit. Empty means the old single-bore check."""
    min_spar_reach_frac: float = 0.55
    """How far along the half span the shortest spar must reach. The tip
    panel is allowed to outrun it -- insisting a spar reach the tip would
    force the whole wing thick to satisfy its thinnest tenth -- but the
    outer JOINT has to be carried."""
    motor: object | None = None
    battery: object | None = None
    spar_d_mm: float = 4.0
    """Spar the bore gate must clear. Belongs to the MISSION: micro's
    tip panel is a few millimetres thick and cannot swallow the 4 mm tube
    a 900 mm trainer wants, which is the single gate that rejected an
    otherwise complete 363 mm aeroplane."""
    powertrain: object | None = None
    min_thrust_weight: float = 0.0
    min_elevon_power: float = 0.004
    max_elevon_power: float = 0.030
    servo_arm_mm: float = 11.0
    horn_below_mm: float = 8.0
    servo_travel_deg: float = 60.0
    servo_stall_nmm: float = 176.0
    """The servo's stall torque, in newton-millimetres. DECLARED
    hardware, like the arm length beside it: a 9 g servo is about
    1.8 kg.cm. It is the load case for the control horn, because a servo
    that meets a jammed surface delivers its stall torque and something
    has to be the thing that gives."""
    servo_shaft_offset_mm: float = 11.35
    """How far the servo's output shaft sits from the middle of its body
    along the span: half of a 9 g servo's 22.7 mm. The servo is mounted
    with the shaft OUTBOARD, so the arm, the pushrod and the horn all
    live at the pocket's centre plus this, and the linkage is solved
    there."""
    """The mechanism, as declared hardware.

    A 9 g servo's outermost arm hole is about 11 mm from the shaft, a
    moulded control horn stands 8 mm off the surface, and a standard servo
    gives about 60 degrees each way before the arm binds. The horn's ARM
    is not declared -- it is the section's thickness at the hinge plus the
    protrusion, because the horn screws to the lower surface while the
    hinge is on the upper one, so the leverage changes along the span
    whether or not anyone models it."""
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
    min_cl_trim: float = 0.0
    min_cn_beta: float = 0.0
    """Directional stiffness floor, per radian of sideslip.

    Zero means 'not checked', which is what every design before this one
    got. A swept tailless wing with too little Cn_beta does not fail
    dramatically -- it will not hold a heading, and because sweep couples
    yaw into roll it drops a wing whenever it skids. Nothing in the
    longitudinal gates can see any of that, so the optimizer was free to
    spend sweep and dihedral purely on drag."""
    max_roll_yaw_ratio: float = 1e9
    """Ceiling on |Cl_beta / Cn_beta|.

    Deliberately LOOSE. The Cl_beta model is strip theory on real
    geometry and the Cn_beta sweep term carries a published coefficient,
    so the ratio is good for rejecting the pathological corner -- a huge
    winglet on a heavily dihedralled wing -- and is not a handling
    qualities prediction. Treated as a sanity bound, not a target."""
    fairness: fz.Limits = field(default_factory=fz.Limits)
    """How far the SHAPE may depart from one continuous surface. Checked
    before anything expensive: an unfair loft is rejected in ~50 ms and
    never pays for a lattice, panels or a print check."""
    min_dutch_roll_zeta: float | None = None
    """Floor on the Dutch-roll damping ratio (aero/dynamics.py); None means
    not checked. MIL-F-8785C puts Level 1 at 0.08. Negative is a wobble
    that grows. This is the gate the roll/yaw ratio was standing in for:
    the gen3 trainer passed the ratio at 10.8 with a DIVERGENT Dutch roll."""
    min_aeroelastic_margin: float = 0.0
    """Factor the divergence and reversal speeds must clear the design
    speed by. 0 means not checked, which is what every design before this
    one got -- and demon1, scored at 44 m/s on a single-wall foamed shell,
    is exactly the aircraft that needed it.

    The margin is deliberately modest because the model is a LOWER bound:
    Bredt-Batho on one closed cell ignores the extra cells the rib truss
    makes, so the real GJ is higher. A large factor on a conservative
    model is two safety margins stacked, which rejects designs for
    arithmetic rather than for physics."""
    min_hands_off_margin: float = 0.0
    """Floor on (hands-off trim speed) / (slowest trimmed speed); 0 is off.

    Sticks centred, the aircraft settles at its trim speed; the slowest it
    can be flown is where the first section stalls or the up-elevon runs
    out (`performance.slow_flight`). 1.3 is the classic approach-speed
    margin over the stall. A trainer that trims closer than that is one
    gust, or one nervous pull, from the stall."""
    stall_onset_ahead_of_cg: bool = False
    """The first section to stall must sit AHEAD of the centre of gravity.

    A section that stops lifting behind the CG pitches the nose UP, which
    deepens the stall and spreads it -- the swept wing's pitch-up. Ahead
    of the CG, the nose drops and the aircraft recovers on its own. The
    tip-stall gate cannot see this: on a swept wing the tips can be well
    unloaded and the peak can still sit at mid-span, 30-50 mm aft of the
    CG, which is where every micro_fpv design through gen8 put it."""
    min_spiral_t2_s: float = 0.0
    """Fastest acceptable spiral divergence, as time to double; 0 is off.
    Yaw stiffness -- fins especially -- pushes the spiral mode toward
    neutral, and a trainer that tightens into a spiral hands-off is not a
    trainer."""
    """Floor on the HANDS-OFF trim lift coefficient.

    The speed objective scores top speed with down-elevon held, which is
    correct and thrust-bounded -- but nothing stopped the optimizer from
    also placing the hands-off trim point at the top end. demon1 duly
    trimmed at CL 0.032, meaning that with the sticks centred it flies at
    41 m/s and the pilot must hold UP elevator to slow down. That is not
    a racer, it is a dart. A racer trims somewhere sane and is PUSHED
    fast; this floor says where sane is."""
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
    def _common(servo_g: float = 0.018, esc_g: float = 0.020) -> tuple:
        """The declared payload. Everything here is a mass someone weighed.

        The spar is NOT here any more. It used to be a flat "spar +
        joiners" item -- 30 g trainer, 38 g demon1, 14 g micro -- and it
        was the last invented number in the mass budget: `structure.select`
        sized a real tube, `spars.fit_all` fitted two corridors, and
        neither reached the budget. It is now computed from the tubes
        actually fitted, at the stations they were fitted to, by
        `structure.spar_masses`."""
        return (
            Item("2205 2300kv + prop", 0.036, 0.97),   # pusher, at the TE
            Item("AR630 rx", 0.008, 0.34),
            Item("esc + wiring", esc_g, 0.50),
            # 0.72c is a nominal only: this item now lives in the
            # "servos" bay and takes the station the bay is SOLVED to,
            # because a servo has to sit beside the surface it drives.
            Item("servos x2", servo_g, 0.72),
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
                  Bay("AR630 + esc", 0.42, (40.0, 34.0, 16.0),
                      x_lo=0.12, x_hi=0.86,
                      holds=("AR630 rx", "esc + wiring")),
                  # A 9 g servo (the 18 g declared for two), ON ITS SIDE:
                  # 22.5 x 11.8 x 22.7 mm standing up will not go into an
                  # outer panel 17 mm deep, so the long axis runs chordwise
                  # and the 12 mm dimension is the one through the section.
                  # The 22.5 is the BODY; the mounting ears take the
                  # chordwise length to 32, and a pocket cut to the body
                  # alone is a pocket the servo does not go into without
                  # trimming its ears off. Seated out in the wing near the
                  # surface it drives, and both sides' mass is carried at
                  # the same x because the aircraft is symmetric and only
                  # x enters the CG.
                  Bay("servos", 0.55, (32.0, 23.0, 12.0),
                      x_lo=0.20, x_hi=0.68,
                      eta_lo=0.30, eta_hi=0.80,
                      lidded=False, drives_elevon=True,
                      holds=("servos x2",))),
            battery_kg=0.110,
            cruise_band_ms=(7.0, 11.0),
            min_static_margin=0.15, max_static_margin=0.32,
            cl_max_section=0.85, max_mass_kg=0.50,
            max_wing_loading_gdm2=26.0, tip_stall_margin=0.12,
            n_limit_g=3.0, max_trim_alpha_deg=8.0,
            # Calibrated against 3000 random designs: Cn_beta runs a
            # median of +0.028 and a lower quartile of +0.016, so 0.025
            # asks for roughly the better half and is reachable without
            # a grotesque winglet. The roll/yaw ceiling sits above the
            # median on purpose -- it is there to catch the pathological
            # corner, not to steer the design.
            # Roll/yaw ceiling tightened from 11 to 8.5: the lever
            # study showed moving the up-turn to the tips reaches it for
            # nothing. Now a backstop -- the damping gate below is the
            # real test, and the ratio's dihedral term is known to run high.
            min_cn_beta=0.025, max_roll_yaw_ratio=8.5,
            min_aeroelastic_margin=1.5,
            min_dutch_roll_zeta=0.08, min_spiral_t2_s=20.0,
            fairness=fz.Limits(max_root_t_over_c=0.24,
                               max_tip_rise_frac=0.22),
            spar_d_mm=8.0, motor=prop.M2205, battery=prop.PACKS["3S 1300"],
            spars=(sp.SparSpec("LE spar", 8.0, 0.12, 0.30),
                   sp.SparSpec("TE spar", 8.0, 0.54, 0.74)),
            powertrain=prop.trainer_power(),
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
            payload=Mission._common(esc_g=0.026),
            # A 4S 1300 is a big pack for an 800 mm racer: 76 mm long
            # needs a 463 mm root chord to sit at 0.10c without hanging
            # off the nose, and the bound is 460. It missed by 3 mm and
            # that one gate sank the whole search. A 4S 850 is the pack
            # this aircraft would actually fly.
            bays=(Bay("4S 850", 0.30, (65.0, 34.0, 24.0), x_var="batt_x"),
                  Bay("AR630 + esc", 0.44, (42.0, 34.0, 16.0),
                      x_lo=0.14, x_hi=0.86,
                      holds=("AR630 rx", "esc + wiring")),
                  Bay("servos", 0.55, (32.0, 23.0, 12.0),
                      x_lo=0.20, x_hi=0.70,
                      eta_lo=0.25, eta_hi=0.75,
                      lidded=False, drives_elevon=True,
                      holds=("servos x2",))),
            battery_kg=0.105,
            # This band is the HANDS-OFF trim window, not the top end:
            # top speed is the objective and is scored separately with
            # down-elevon. Policing trim speed at 20-48 m/s let the
            # aircraft trim at 41 and call it cruise.
            cruise_band_ms=(15.0, 32.0),
            min_cl_trim=0.15,
            min_static_margin=0.04, max_static_margin=0.16,
            cl_max_section=1.00, max_mass_kg=0.85,
            max_overhang_deg=56.0,       # printed once, carefully
            max_wing_loading_gdm2=1e9,      # loading is the POINT here
            tip_stall_margin=0.04,
            n_limit_g=6.0, max_trim_alpha_deg=5.0,
            # A racer buys speed with drag it does not spend elsewhere,
            # and tolerates livelier handling than a trainer.
            min_cn_beta=0.020, max_roll_yaw_ratio=12.0,
            min_aeroelastic_margin=1.3,
            min_dutch_roll_zeta=0.08,
            fairness=fz.Limits(max_root_t_over_c=0.24,
                               max_tip_rise_frac=0.20),
            spar_d_mm=8.0, motor=prop.M2205, battery=prop.PACKS["4S 850"],
            spars=(sp.SparSpec("LE spar", 8.0, 0.12, 0.30),
                   sp.SparSpec("TE spar", 8.0, 0.54, 0.74)),
            powertrain=prop.demon_power(),
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
            payload=Mission._common(servo_g=0.010, esc_g=0.012),
            bays=(Bay("2S 450", 0.30, (55.0, 30.0, 17.0), x_var="batt_x"),
                  Bay("AR630", 0.44, (30.0, 20.0, 12.0),
                      x_lo=0.14, x_hi=0.88,
                      holds=("AR630 rx", "esc + wiring")),
                  # 5 g sub-micro, on its side: 20 x 8.6 x 20 mm becomes
                  # 20 chordwise x 20 spanwise x 9 through the section,
                  # and its ears take the chordwise length to 24.
                  Bay("servos", 0.55, (24.0, 20.0, 9.0),
                      x_lo=0.20, x_hi=0.70,
                      eta_lo=0.30, eta_hi=0.80,
                      lidded=False, drives_elevon=True,
                      holds=("servos x2",))),
            # sub-micro hardware to match the sub-micro servos
            servo_arm_mm=7.0, horn_below_mm=5.0, servo_shaft_offset_mm=10.0,
            servo_stall_nmm=78.0,       # 5 g sub-micro, about 0.8 kg.cm
            battery_kg=0.028,
            cruise_band_ms=(8.0, 17.0),
            min_static_margin=0.10, max_static_margin=0.26,
            cl_max_section=0.90, max_mass_kg=0.22,
            max_wing_loading_gdm2=42.0, tip_stall_margin=0.08,
            n_limit_g=4.0, max_trim_alpha_deg=9.0,
            # Loosest of the three: micro is already the most constrained
            # mission in the fleet and is flown close in, where a wander
            # is corrected before it matters.
            min_cn_beta=0.018, max_roll_yaw_ratio=12.0,
            min_aeroelastic_margin=1.5,
            min_dutch_roll_zeta=0.08,
            # micro gets the most winglet: at 350 mm the fin arm is
            # short, so side area is the only yaw stiffness on offer
            fairness=fz.Limits(max_root_t_over_c=0.26,
                               max_tip_rise_frac=0.28),
            # 2.5 mm carbon ROD, not a tube: micro's tip panel is a few
            # millimetres thick and the 4 mm tube a 900 mm trainer wants
            # is the single gate that rejected an otherwise complete
            # 363 mm aeroplane.
            spar_d_mm=8.0, motor=prop.M2205, battery=prop.PACKS["2S 450"],
            spars=(sp.SparSpec("main spar", 8.0, 0.18, 0.40),),
            powertrain=prop.micro_power(),
            min_spar_reach_frac=0.45,
            min_thrust_weight=0.75,
            min_elevon_power=0.004, max_elevon_power=0.026,
        )

    @staticmethod
    def micro_fpv() -> "Mission":
        """Sub-250 g, FPV, and as forgiving as the class allows.

        250 g is not a target, it is a legal ceiling -- the registration
        threshold in most places -- so the mission treats it as a gate and
        spends nothing to approach it. The OBJECTIVE is stall speed, and
        minimising it pushes mass down and area up together.

        ## Why the span is fixed at 480 mm

        Every other mission lets the search choose a span. This one
        cannot, because the span IS the part height: panels print
        root-down with span along Z, so a half wing taller than the 250 mm
        envelope has to be cut into pieces. 480 mm of span is 240 mm of
        half span, which fits in one piece with the 8 mm margin
        `panel_etas` keeps. Ask for a millimetre more and the aircraft
        arrives as twice as many parts.

        The elevon still forces a break at its own root -- a wing's
        trailing edge cannot vanish mid-panel, because a surface normal
        to the span is a roof and spiralize cannot build one -- so the
        half wing is two parts, not one. Two per side is the floor for
        anything with a moving surface and a pusher motor between them.

        ## What it carries that the others do not

        An FPV camera and video transmitter, 8 g declared, at the nose.
        They are NOT given a bay: nothing is cut through the skin any
        more, so a camera buried inside a closed shell would be looking at
        the inside of it. The declared mass sits at 0.06c and the hardware
        goes in a pod bonded to the nose, which is where a camera on a
        wing this size goes anyway.

        ## Where the gates come from

        The stability numbers are the TRAINER's, not micro's. micro's are
        deliberately the loosest in the fleet because it is flown close in
        and is a proof that a 363 mm aeroplane can exist at all; this one
        is meant to be pleasant, so it inherits the mission that was
        written around what happens when a beginner lets go of the sticks.
        Wing loading is the one number relaxed from the trainer's 26, to
        34: a 480 mm wing cannot reach 26 g/dm2 while carrying 102 g of
        payload, and pretending otherwise would make the mission
        unsatisfiable by construction rather than demanding."""
        return Mission(
            name="micro_fpv", span_m=0.48, objective="docile",
            payload=Mission._common(servo_g=0.010, esc_g=0.012) + (
                # An AIO camera-and-VTX and a dipole: 6 g and 2 g, the
                # mass of the class of part rather than one part someone
                # weighed. Declared, and stated as declared.
                Item("fpv cam + vtx", 0.008, 0.06),
            ),
            bays=(Bay("2S 450", 0.30, (55.0, 30.0, 17.0), x_var="batt_x"),
                  Bay("AR630", 0.44, (30.0, 20.0, 12.0),
                      x_lo=0.14, x_hi=0.88,
                      holds=("AR630 rx", "esc + wiring")),
                  Bay("servos", 0.55, (24.0, 20.0, 9.0),
                      x_lo=0.20, x_hi=0.70,
                      eta_lo=0.30, eta_hi=0.80,
                      lidded=False, drives_elevon=True,
                      holds=("servos x2",))),
            servo_arm_mm=7.0, horn_below_mm=5.0, servo_shaft_offset_mm=10.0,
            servo_stall_nmm=78.0,
            battery_kg=0.028,
            # Slow, and the band is wide because the objective is already
            # pushing on the bottom of it.
            cruise_band_ms=(6.0, 14.0),
            # Pitch stiffness, set for THIS aircraft rather than copied.
            #
            # The first draft took the trainer's [0.15, 0.32] on the
            # reasoning that more margin is more docile. That is not true
            # past a point: a tailless wing at 20% and up is nose-heavy,
            # needs more up-reflex to trim, and spends elevon authority
            # holding it there. The handling sweet spot for the type is
            # 8-15%.
            #
            # The floor also has to be reachable at this span. The 900 mm
            # trainer makes +0.209 comfortably and the 352 mm micro only
            # +0.035, because the fixed 102 g of payload dictates the CG
            # at small scale; four 480 mm searches wandered between -0.061
            # and +0.140 against a 0.15 floor. The seeded design makes
            # +0.267, so the band was not unreachable -- it was simply the
            # wrong band, aimed at an aeroplane twice the size.
            min_static_margin=0.12, max_static_margin=0.28,
            cl_max_section=0.85, max_mass_kg=0.25,
            max_wing_loading_gdm2=34.0, tip_stall_margin=0.12,
            n_limit_g=3.0, max_trim_alpha_deg=8.0,
            min_cn_beta=0.025, max_roll_yaw_ratio=8.5,
            min_aeroelastic_margin=1.5,
            min_dutch_roll_zeta=0.08, min_spiral_t2_s=20.0,
            # For a first-time pilot, and measured on the aircraft as
            # flown (performance.slow_flight): hands-off at least 1.3x
            # the slowest trimmed speed, and a stall that drops the nose.
            min_hands_off_margin=1.3, stall_onset_ahead_of_cg=True,
            fairness=fz.Limits(max_root_t_over_c=0.26,
                               max_tip_rise_frac=0.28),
            # 6 mm, not micro's 8. The bore gate rejected two of the
            # four first searches on the outboard panel, which on a 480 mm
            # wing is a few millimetres of section -- and an 8 mm tube is
            # sized for a 900 mm trainer pulling 3 g at 500 g. The
            # buckling and aeroelastic gates size what is actually needed;
            # this is only the floor they start from.
            spar_d_mm=6.0, motor=prop.M2205, battery=prop.PACKS["2S 450"],
            spars=(sp.SparSpec("main spar", 6.0, 0.18, 0.40),),
            powertrain=prop.micro_power(),
            min_spar_reach_frac=0.45,
            # Relaxed from micro's 0.75. That number is for an aeroplane
            # that has to climb away from a hand launch briskly; this one
            # is slow by construction, and 0.6 static is enough to
            # accelerate a 7 m/s wing off a throw. It is still a gate.
            min_thrust_weight=0.60,
            min_elevon_power=0.004, max_elevon_power=0.026,
        )

    @staticmethod
    def micro_fpv_winglet() -> "Mission":
        """micro_fpv with its wing kept nearly flat, so that yaw stiffness
        has to come from REAL winglets -- the tip fins.

        micro_fpv's gen7 winner curled the outer 27% of its semi-span up
        through 54 degrees of cant (its cap is 28%) and carried no fins,
        which looks extreme and raised the obvious question: is the curl a
        winglet, or just what the search found cheapest? It was an uneven
        contest at the time: the curl was in the vortex lattice and the
        fins were not, so a fin earned nothing for tip losses. With the
        fins in the lattice, this mission asks the same question the other
        way round -- same gates, same objective, same aircraft -- with the
        tip rise capped at 12% of the semi-span. Whichever finds the
        better feasible design answers it on the numbers.

        A variant, not a fifth aircraft: see MISSION_VARIANTS. The cap is
        part of the design -> planform map (`build` clamps the rise to it),
        so a design found here is only this aircraft under this mission."""
        m = Mission.micro_fpv()
        return replace(m, name="micro_fpv_winglet",
                       fairness=replace(m.fairness, max_tip_rise_frac=0.12))

    @staticmethod
    def beginner_trainer() -> "Mission":
        return Mission.trainer_v3()


MISSIONS = ("trainer_v3", "demon1", "micro", "micro_fpv")
"""Every mission the CLI may be asked for, declared beside the factories.

run.py's argparse used to carry its own hardcoded list, and the list had
drifted: `--mission fpv_1m` was offered by `--help` and by tab completion
and crashed with `AttributeError: type object 'Mission' has no attribute
'fpv_1m'` on EVERY command, because no such factory exists. It failed
after the whole command line had been typed and, on a search, after the
print settings had been built.

A CLI that can name a mission the program cannot build is the same class
of mistake as a design vector that can express an invalid planform: the
fix is to make it unrepresentable rather than to correct the one instance.
`beginner_trainer` is deliberately absent -- it is an alias for
trainer_v3, not a fourth aircraft."""

MISSION_VARIANTS = {"micro_fpv_winglet": "micro_fpv"}
"""Search-space variants of a fleet mission: variant -> parent.

Same aircraft, same gates, same objective; a narrower design space. The
CLI offers them beside MISSIONS, but the fleet -- the tuple the tests walk
and results/fleet/index.json answers for -- is MISSIONS alone, so a
variant never has to have a tracked winner to exist."""

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
    # CHORD AS A RUNNING PRODUCT, not four independent fractions. Each
    # station's chord is the previous one's times a taper ratio of at
    # most one, so chord cannot grow outboard anywhere in the unit cube
    # -- the gen2 generator could draw a 37 mm outer station ahead of a
    # 47 mm tip, and random draws did. The lower bounds stop any single
    # segment collapsing the chord in one step, which is what hooked the
    # trailing edges.
    Bound("kink_taper", 0.52, 0.97, ""),
    # Fifth station, still placed by a gap fraction so body < kink <
    # outer < tip holds identically. Narrower than gen2: a gap of 0.78
    # left a tip segment 9% of the span long carrying a 60 degree
    # winglet, which is a stub, not a panel.
    Bound("outer_gap", 0.25, 0.65, ""),
    Bound("outer_taper", 0.62, 1.00, ""),
    Bound("tip_taper", 0.60, 1.00, ""),
    # SWEEP AS INCREMENTS. Adjacent gen2 segments could disagree by 40
    # degrees; demon1 went 43 -> 19 -> 29, and steep-flat-steep is a
    # wave no loft can hide. The body sets the sweep and each segment
    # outboard may relax it by a bounded amount or add only a little
    # back, which is the shape of every blended wing body that works.
    Bound("sweep_body", 20.0, 62.0, "deg"),
    Bound("sweep_mid_delta", -20.0, 4.0, "deg"),
    # Outboard of the mid segment sweep may only relax (the tip may add
    # back less than the 2 degree reversal amplitude). With +4 and +6
    # allowed, a third of random draws swept up, down and up again, and
    # the fairness gate rejected them all. Now the sweep distribution
    # has one hump by construction -- the nose rising to the body or mid
    # sweep -- and the gate never sees a leading-edge wave to reject.
    Bound("sweep_outer_delta", -12.0, 0.0, "deg"),
    Bound("sweep_tip_delta", -8.0, 1.5, "deg"),
    # DIHEDRAL THAT TURNS UP IN STAGES. Inboard dihedral, a final cant,
    # and a blend that puts the outer segment's dihedral BETWEEN them,
    # so the wing always turns up through two steps and never snaps from
    # 5 to 60 degrees at a single station. That is the difference
    # between a blended winglet and a bracket bolted on.
    Bound("dihedral", 0.0, 8.0, "deg"),
    Bound("winglet_cant", 0.0, 60.0, "deg"),
    Bound("winglet_blend", 0.35, 0.65, ""),
    # TWIST AS ONE SMOOTH FUNCTION: root incidence, total washout, and
    # where along the span it is spent --
    #     twist(eta) = twist_root - washout * eta ** washout_exp
    # Monotone by construction. gen2's four independent twists zig-zagged
    # (+0.4, -4.7, -0.9) and every zig showed up in the span loading.
    Bound("twist_root", -2.0, 3.0, "deg"),
    Bound("washout", -1.0, 8.0, "deg"),
    Bound("washout_exp", 0.6, 3.0, ""),
    Bound("body_thickness", 1.05, 2.00, "x"),
    Bound("batt_x", 0.06, 0.72, "c_root"),
)

SECTION_BOUNDS = (
    # ROOT section camber and reflex. Until now there was one of each for
    # the whole aircraft, and the pipeline passed the same airfoil object
    # as both root and tip -- so the blend was a no-op and camber measured
    # identical to five decimals at every station. A blended wing body
    # trims itself with reflex, and reflex is cheapest where the chord is
    # longest and the arm shortest: at the BODY. Lift is wanted where the
    # arm is longest: OUTBOARD. One knob could not ask for both.
    Bound("reflex_deg", -1.0, 9.0, "deg"),
    Bound("camber_scale", 0.30, 1.70, "x"),
    Bound("tip_reflex_deg", -3.0, 7.0, "deg"),
    Bound("tip_camber_scale", 0.10, 1.60, "x"),
    # How fast the section morphs from root shape to tip shape along the
    # span: blend(eta) = eta ** blend_exp. Above 1 the body holds its own
    # shape well outboard and the change happens late, which is what a
    # blended wing body looks like; below 1 it changes immediately.
    # Narrowed from 0.5-3.2: at 3.2 half the section change happens in
    # the last fifth of the span, so the tip panel visibly changes shape
    # within a few centimetres.
    Bound("blend_exp", 0.70, 2.20, ""),
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
    # The propeller. demon1's top speed is thrust-limited, so pitch is
    # worth more to it than any change to the wing -- leaving it hardcoded
    # meant the optimizer could not buy the one thing it most needed.
    Bound("prop_diam_in", 4.0, 7.0, "in"),
    Bound("prop_pitch_in", 2.5, 6.5, "in"),
    # Vertical tip fins (aero/fins.py): flat plates printed on the bed and
    # glued on, because vase mode cannot print a surface normal to the
    # span. Area is ONE fin as a fraction of wing area; below
    # FIN_MIN_AREA_FRAC there are no fins at all, so "none" is a real
    # region of the search rather than a token pair of tabs.
    Bound("fin_area_frac", 0.0, 0.06, "S"),
    Bound("fin_aspect", 0.8, 2.0, ""),
    Bound("fin_below", 0.0, 0.4, ""),
)
BATTERY_NAME = "battery"
PROP_CLEARANCE_MM = 10.0
"""Clearance the propeller disc keeps from the trailing edge.

A DECLARED process limit, in the same category as `max_overhang_deg`: a
motor mount bonded to foamed PLA flexes under thrust and gyroscopic
load, the prop itself is not perfectly true, and a strike at 30 000 rpm
destroys both. Not validated against a measurement."""

RAMP_MARGIN = 0.85
"""Fraction of the measured overhang budget a bay's ramp may use.

MEASURED, in the style `build_stack` measures the rib truss's factor. The
budget is taken on the bare skin, but a cut's corners sit off that skin
and, rotated by twist and judged against the nearest previous-layer
vertex, move up to 7% faster between layers than the skin at the same
station -- on the trainer's servo pocket 1.048 mm/mm was allowed and
0.977 was available, 51.3 degrees against a 50 degree limit, on both
sides of the p0/p1 joint. 0.85 keeps every measured case under the limit
with margin; a larger factor is a longer ramp, which is the right way to
be wrong."""
MIN_BAY_WALL_STRUCTURAL_MM = 10.0
"""Narrowest wall two openings may leave between them, for its own sake.

A declared print-process minimum, in the same category as the overhang
limit: a two-bead skin 5 mm wide and 30 mm tall between two holes is a
fin, and one that is 10 mm wide is a wall. Not derived -- there is no
buckling model for a free-standing sliver here -- and stated as such."""

MIN_BAY_WALL_MM = MIN_BAY_WALL_STRUCTURAL_MM
"""...and nothing has to CROSS it any more.

This was the larger of two reasons: the structural 10 mm, and 11 mm for
the pack's XT30 to cross the wall between the root bays in a channel.
The channels are gone -- a wire cannot be routed on a surface that is
one continuous bead -- so only the structural reason is left. The alias
is kept rather than folded away because the gate reads better naming
the wall than naming the sliver."""

BOUNDS = PLANFORM_BOUNDS + SECTION_BOUNDS
N_DIM = len(BOUNDS)
FIN_MIN_AREA_FRAC = 0.005


def unit_to_physical(u: np.ndarray) -> dict:
    u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
    return {b.name: b.lo + v * (b.hi - b.lo) for b, v in zip(BOUNDS, u)}


def physical_to_unit(p: dict) -> np.ndarray:
    return np.array([(p[b.name] - b.lo) / (b.hi - b.lo) for b in BOUNDS])


def build(u: np.ndarray, mission: Mission, base: Airfoil) -> Planform:
    """Design vector -> planform. Total function: every u in [0,1]^n
    produces geometry, valid or not. Validity is judged, not assumed.

    Five stations now, and a real spanwise section family: the root and
    the tip are DIFFERENT airfoils, lerped in CST coefficient space along
    the span. That is what `blend` in each Segment selects.
    """
    p = unit_to_physical(u)
    # shared shape: t/c and the thickness peak are properties of the
    # whole aircraft's structure and print, not of one station
    shell = scale_thickness_ratio(base, p["t_over_c"])
    shell = set_thickness_peak(shell, p["x_tmax"])
    root_af = deflect_te(scale_camber(shell, p["camber_scale"]),
                         p["reflex_deg"])
    tip_af = deflect_te(scale_camber(shell, p["tip_camber_scale"]),
                        p["tip_reflex_deg"])

    span = p["span_m"] if mission.span_free else mission.span_m
    body_eta = p["body_eta"]
    kink_eta = body_eta + p["kink_gap"] * (1.0 - body_eta)
    outer_eta = kink_eta + p["outer_gap"] * (1.0 - kink_eta)

    # chord: running product, monotone non-increasing by construction
    body_c = p["body_chord_frac"]
    kink_c = body_c * p["kink_taper"]
    outer_c = kink_c * p["outer_taper"]
    tip_c = outer_c * p["tip_taper"]

    # sweep: body sets it, each segment outboard adjusts by a bounded step
    sw_body = p["sweep_body"]
    sw_mid = float(np.clip(sw_body + p["sweep_mid_delta"], 0.0, 66.0))
    sw_outer = float(np.clip(sw_mid + p["sweep_outer_delta"], 0.0, 66.0))
    sw_tip = float(np.clip(sw_outer + p["sweep_tip_delta"], 0.0, 66.0))

    # dihedral: the outer segment always sits between inboard and cant
    d_in = p["dihedral"]
    d_tip = p["winglet_cant"]
    d_out = d_in + p["winglet_blend"] * (d_tip - d_in)

    def twist_at(eta: float) -> float:
        return float(p["twist_root"] - p["washout"] * eta ** p["washout_exp"])

    # Root t/c is capped by the mission's fairness limit; clamping here
    # makes a potato root unrepresentable instead of merely penalised.
    bt = p["body_thickness"]
    t_cap = 0.98 * mission.fairness.max_root_t_over_c
    if root_af.t_max * bt > t_cap:
        bt = max(t_cap / max(root_af.t_max, 1e-6), 1.0)

    # And the body may not fall away faster than the fairness limit. The
    # pod gate rejected up to half of all random draws, most of them on
    # micro, where a body_eta of 0.10 is 17 mm of span in which to lose a
    # quarter of the thickness -- the same fraction is 45 mm on trainer.
    # Estimated on the control stations with a 0.6 margin for the ~1.5x
    # peak-to-mean of a smooth ramp from zero slope; the gate on the
    # real loft stays as the backstop.
    half_mm = 0.5 * span * 1000.0
    c_root_mm = p["root_chord"] * 1000.0
    t_sec = root_af.t_max
    slope_cap = 0.6 * np.tan(np.radians(mission.fairness.max_thickness_slope_deg))

    def body_slope(b: float) -> float:
        ts = (b, 1.0 + 0.75 * (b - 1.0), 1.0 + 0.30 * (b - 1.0))
        cs = (1.0, body_c, kink_c)
        ys = (0.0, body_eta * half_mm, kink_eta * half_mm)
        h = [0.5 * t_sec * t_ * c_ * c_root_mm for t_, c_ in zip(ts, cs)]
        return max((h[i] - h[i + 1]) / max(ys[i + 1] - ys[i], 1e-6)
                    for i in range(2))

    if body_slope(bt) > slope_cap:
        lo_b, hi_b = 1.0, bt
        for _ in range(24):
            mid_b = 0.5 * (lo_b + hi_b)
            if body_slope(mid_b) > slope_cap:
                hi_b = mid_b
            else:
                lo_b = mid_b
        bt = lo_b
    exp = p["blend_exp"]

    def blend_at(eta: float) -> float:
        return float(np.clip(eta, 0.0, 1.0) ** exp)

    segs = (
        Segment(body_eta, body_c, sw_body, d_in, twist_at(body_eta),
                blend_at(body_eta), 1.0 + 0.75 * (bt - 1.0)),
        Segment(kink_eta, kink_c, sw_mid, d_in, twist_at(kink_eta),
                blend_at(kink_eta), 1.0 + 0.30 * (bt - 1.0)),
        Segment(outer_eta, outer_c, sw_outer, d_out, twist_at(outer_eta),
                blend_at(outer_eta), 1.0),
        Segment(1.0, tip_c, sw_tip, d_tip, twist_at(1.0), 1.0, 1.0),
    )
    return faired(
        half_span_m=0.5 * span,
        root_chord_m=p["root_chord"],
        root_twist_deg=p["twist_root"],
        root_airfoil=root_af,
        tip_airfoil=tip_af,
        segments=segs,
        root_thickness_scale=bt,
        max_tip_rise_frac=0.98 * mission.fairness.max_tip_rise_frac,
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


# ------------------------------------------------------------- lattice cache

_VLM_CACHE: "OrderedDict[tuple, VLM]" = OrderedDict()
_VLM_CACHE_SIZE = 2
_BUILD_CACHE: "OrderedDict[tuple, Planform]" = OrderedDict()


def _cached_build(u, mission: Mission, base: Airfoil) -> Planform:
    """The planform for this design, built at most once per evaluation.

    `evaluate` scores the bare shell and then the ribbed one, and each pass
    rebuilt an identical Planform from the same vector -- which threw away
    the station memo `Planform.at` keeps, so every station the second pass
    asked for was lofted twice. Keyed on everything `build` reads,
    including the mission's NAME, which becomes the planform's and so every
    part's; a hit is the same geometry by construction. A Planform is
    frozen, so sharing one is safe."""
    key = (np.asarray(u, dtype=float).tobytes(), mission.span_m,
           mission.span_free, mission.fairness, mission.name,
           base.au.tobytes(), base.al.tobytes(), float(base.te_gap),
           float(base.te_camber))
    plan = _BUILD_CACHE.get(key)
    if plan is None:
        plan = build(u, mission, base)
        _BUILD_CACHE[key] = plan
        while len(_BUILD_CACHE) > _VLM_CACHE_SIZE:
            _BUILD_CACHE.popitem(last=False)
    else:
        _BUILD_CACHE.move_to_end(key)
    return plan


def _cached_vlm(u, mission: Mission, base: Airfoil, plan: Planform,
                ns: int, nc: int, fins=None) -> VLM:
    """The lattice for this design, built at most once.

    Scoring the aircraft as built takes a bare-shell pass, a structure
    sizing and a ribbed pass, and each used to build its own lattice from
    identical geometry -- the ribs change the mass, never the shape. Timed
    on the gen5 seed designs, a built-aircraft evaluation cost 2.3-2.8x a
    bare one, and a large share of that was the same 512-panel influence
    matrix inverted three times.

    Keyed on everything build() reads -- the design vector, the span, the
    mission's fairness clamps, the base airfoil -- plus the lattice size,
    so a hit is the same geometry by construction. Two entries is enough:
    one evaluation at a time uses one."""
    key = (np.asarray(u, dtype=float).tobytes(), mission.span_m,
           mission.span_free, mission.fairness, base.au.tobytes(),
           base.al.tobytes(), float(base.te_gap), float(base.te_camber),
           int(ns), int(nc), fins)
    vlm = _VLM_CACHE.get(key)
    if vlm is None:
        # The fins are panels of the lattice: their end-plate effect on
        # the tip loading and the wake is part of the solve, where it used
        # to be left unclaimed -- which scored a real winglet at zero
        # induced-drag benefit and a curled wing tip at its full one.
        vlm = VLM(plan, ns=ns, nc=nc, fins=fins)
        _VLM_CACHE[key] = vlm
        while len(_VLM_CACHE) > _VLM_CACHE_SIZE:
            _VLM_CACHE.popitem(last=False)
    else:
        _VLM_CACHE.move_to_end(key)
    return vlm


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
    elevons: list = field(default_factory=list)
    horn_eta: float = 0.0
    """Span station of the control horn: the servo's shaft, or the
    elevon's root when the shaft falls short of it."""
    """The control surfaces as they were SCORED, sockets cut. They used
    to be built once here for the mass and the gates and again in
    `do_export` for the STLs, which is two places building one part --
    the failure this project keeps finding. The export uses these."""
    spar_fits: list = field(default_factory=list)
    lateral: object | None = None
    fairness: object | None = None
    fairness_limits: object | None = None
    fins: object | None = None
    slow: object | None = None
    """performance.SlowFlight: the slowest trimmed flight, and what limits it."""
    powertrain: object | None = None
    """The powertrain this design was SCORED with -- the mission's motor
    and pack, with the design's own propeller. The mission's powertrain
    is a default the design vector overrides, and the export built the
    motor mount from that default: gen9's build sheet checked a 5.04 in
    disc the search never chose, and failed prop clearance by 0.17 mm."""
    inserts: list = field(default_factory=list)
    """The joint wedge inserts (printing/inserts.py): the loft a turning
    joint leaves between two panel faces, as its own printed part.
    Charged in the mass budget, gated for print, exported."""
    dynamics: object | None = None
    structure: object | None = None
    linkage: object | None = None
    aeroelastic: object | None = None
    joints: tuple = ()
    """Every bonded joint the print has, with the shear and torque it
    carries. Panels used to butt together on the spar and nothing else:
    the tube was sized for bending and the glue carried whatever was
    left, unexamined."""
    print_settings: object | None = None
    max_elevon_deflect_deg: float = 12.0
    """The mission's deflection limit, carried out so the exporter sizes
    the hinge bevel from the same number the score used."""
    sm_band: tuple = (0.0, 1.0)
    cruise_band: tuple = (0.0, 0.0)
    max_loading_gdm2: float = 1e9
    """The mission's own limits, carried out with the verdict.

    The build sheet turns them into things a builder can act on -- a
    static-margin band becomes a CG window in millimetres, a cruise band
    becomes a margin to report -- and it should not have to be handed the
    Mission to do it. An Evaluation that cannot say what it was judged
    against is an Evaluation you have to keep a second object beside."""
    """Where the spanwise tubes ended up. Computed during the gates
    and carried out so the export can SHOW the placement -- the first
    version fitted them, gated on them, and then threw the geometry
    away, so the log could only say a spar existed."""

    def line(self) -> str:
        if not self.ok:
            return f"  rejected: {'; '.join(self.reasons)}"
        return (f"  L/D {self.ld:5.2f} | v {self.v_cruise:5.1f} m/s | "
                f"{self.mass_kg*1000:4.0f} g | CL {self.cl_trim:.3f} | "
                f"SM {self.static_margin:+.3f}")


def _evaluate_once(
    u: np.ndarray,
    mission: Mission,
    base: Airfoil,
    settings: vase.PrintSettings,
    drag: perf.DragModel | None = None,
    ns: int = LATTICE_NS,
    nc: int = LATTICE_NC,
    want_panels: bool = False,
    z_step_mm: float | None = None,
    spar_od_mm: float | None = None,
) -> Evaluation:
    """One design -> one verdict. Cheap checks first, on purpose: the
    geometry test costs microseconds and rejects most of a random
    population before any lattice is ever built.

    `spar_od_mm` is the diameter the fitted tubes are weighed at. None
    means the mission's declared hardware diameter, which is what the bore
    gate and the rib corridors were cut for; the second pass of
    `evaluate()` passes the diameter the load case actually demanded."""
    drag = drag or perf.DragModel()
    reasons: list[str] = []
    penalty = 0.0
    p_vec = unit_to_physical(u)
    if mission.motor is not None and mission.battery is not None:
        mission = replace(mission, powertrain=prop.Powertrain(
            mission.motor,
            prop.Propeller(p_vec["prop_diam_in"], p_vec["prop_pitch_in"]),
            mission.battery))
        if mission.powertrain.tip_speed_ms() > prop.MAX_TIP_SPEED_MS:
            reasons.append(
                f"prop tip {mission.powertrain.tip_speed_ms():.0f} m/s over "
                f"{prop.MAX_TIP_SPEED_MS:.0f}")
            penalty += 15.0
    try:
        plan = _cached_build(u, mission, base)
    except Exception as e:
        return Evaluation(False, -1e6, reasons=(f"geometry: {e}",))

    good, why = plan.is_valid()
    if not good:
        return Evaluation(False, -1e6, reasons=(why,), plan=plan)

    # --- fair? one continuous shape, before anything expensive ---
    # Fairness DOMINATES feasibility the way feasibility dominates merit:
    # every fair design outranks every unfair one, and unfair designs are
    # ranked among themselves by how far they miss. Without that ordering
    # the optimizer would happily trade a hooked trailing edge for a
    # tenth of L/D, because nothing else in the score can see a hook.
    fair = fz.measure(plan)
    unfair = fair.violations(mission.fairness)
    if unfair:
        pen = sum(40.0 * max(sev, 0.05) for _, sev in unfair)
        return Evaluation(False, -(5000.0 + pen),
                          reasons=tuple(r for r, _ in unfair), plan=plan,
                          fairness=fair, fairness_limits=mission.fairness)

    fins = (fn.tip_fins(plan, p_vec["fin_area_frac"], p_vec["fin_aspect"],
                        p_vec["fin_below"])
            if p_vec["fin_area_frac"] >= FIN_MIN_AREA_FRAC else None)

    # Spar corridors first: the ribs have to be told where the spars go
    # BEFORE the panels are built, because a chordwise web at a spar's
    # station is a web through the spar.
    # The control surface the score is computed from has to reach the
    # geometry that gets printed. `elevon_chord` and `elevon_eta` are
    # design variables; until now they existed only in the scoring and the
    # exported shell had its trailing edge attached, so the aircraft that
    # flew was not the aircraft that was scored -- the oldest failure mode
    # in the project, one more time.
    settings = replace(settings,
                       elevon_chord=float(p_vec["elevon_chord"]),
                       elevon_eta=float(p_vec["elevon_eta"]))
    joint_etas = vase.panel_etas(plan, settings)
    wall = settings.extrusion_width_mm

    # The payload's reserved volumes, built BEFORE the spars are fitted so
    # the fit can see them. The order matters and it is the physical order:
    # the electronics decide where they can go from the section they are
    # given, and the spar then takes the best remaining seat. Fitting the
    # tube first and checking the pack afterwards is what put a tube
    # through every pack in the fleet.
    bay_vols, bay_seats, bay_etas = seat_bays(
        plan, mission, p_vec, wall, joint_etas)

    # The spars must avoid the payload boxes: a tube seated where the
    # pack sits is a tube through the pack. Nothing else reserves space
    # any more -- the openings that used to reach deeper than their own
    # boxes are gone, so the box IS the reservation.
    spar_fits = []
    if mission.spars:
        spar_fits = sp.fit_all(plan, mission.spars, wall, joint_etas,
                               reserved=tuple(bay_vols),
                               min_reach=mission.min_spar_reach_frac)
        # The tubes as the straight lines they are. The panels keep their
        # truss out of the chord band each line sweeps through inside
        # them, and the bore is gated at the line's own centre in every
        # layer it reaches -- not at a fixed chord fraction, which is the
        # tube that bent with the wing (ROADMAP-CAD.md section 0.1).
        def line(f):
            d = max(f.spec.d_mm, settings.spar_d_mm)
            return (f.root_xz_mm[0], f.root_xz_mm[1], f.slope[0], f.slope[1],
                    f.reach_y_mm, 0.5 * d + 0.5 * wall + f.spec.clearance_mm, d)
        settings = replace(settings, spar_avoid=(), spar_corridors=(),
                           spar_lines=tuple(line(f) for f in spar_fits))

    # --- printable? the shell mass comes out of this, so it runs early ---
    panels = vase.build_panels(plan, settings, z_step_mm=z_step_mm)
    # --- the mechanism ---
    #
    # The horn goes where the pushrod can reach it: at the servo's output
    # shaft, or at the elevon's own root when the shaft falls short of
    # it. The shortfall is gated below rather than papered over.
    #
    # It used to be socketed into the elevon, in a pocket cut by the same
    # machinery as a bay. That pocket landed in the thin tail -- micro
    # had no section to take it at all -- and it is gone with the rest of
    # the cutting.
    link = None
    horn_at = None
    if elv.has_elevon(settings) and "servos" in bay_seats:
        servo_bay = next(b for b in mission.bays if b.name == "servos")
        half_mm = plan.half_span_m * 1000.0
        e_servo = bay_etas.get("servos") or 0.5 * (p_vec["elevon_eta"] + 1.0)
        e_shaft = e_servo + mission.servo_shaft_offset_mm / max(half_mm, 1e-9)
        e_horn = max(e_shaft, float(settings.elevon_eta))
        link = lkg.for_station(
            plan, e_horn, elv.hinge_x(settings), bay_seats["servos"],
            mission.servo_arm_mm, mission.horn_below_mm,
            mission.servo_travel_deg, wall,
            side=+1.0 if servo_bay.open_from == "upper" else -1.0,
            hinge_gap_mm=settings.hinge_gap_mm,
            deflect_deg=mission.max_elevon_deflect_deg)
        rod_n = mission.servo_stall_nmm / max(mission.servo_arm_mm, 1e-9)
        applied = rod_n * link.horn_arm_mm
        horn_at = (e_horn, e_shaft, applied)

    elevon_parts = elv.build_elevons(
        plan, settings, joint_etas,
        mission.max_elevon_deflect_deg + settings.hinge_margin_deg,
        z_step_mm=z_step_mm)
    checks = [vase.check(p) for p in panels + elevon_parts]
    # The elevons are shell too. Splitting the trailing edge off into its
    # own part does not make it weightless, and it adds two walls at the
    # cut -- so the total goes UP slightly, which is the honest direction.
    shell_kg = sum(p.mass_g() for p in panels + elevon_parts) * 2.0 / 1000.0
    print_fail = [f"{p.name}: {','.join(c.failures())}"
                  for p, c in zip(panels + elevon_parts, checks) if not c.ok]
    # --- the wedges the turning joints leave, as parts ---
    #
    # A joint whose dihedral turns leaves a wedge of the loft that neither
    # panel contains -- 6 mm open at micro_fpv's lower skin. It used to be
    # "fill with glue" on the build sheet and missing from CAD, and it was
    # never weighed. It is a printed part now, and a bigger turn is a
    # heavier part, which is the pressure on the search that belongs here.
    try:
        inserts = ins_.joint_inserts(plan, settings, panels)
    except (ValueError, IndexError, np.linalg.LinAlgError) as e:
        # a geometry the insert builder cannot parse is a rejection with
        # a reason, never a crash in a search worker
        inserts = []
        reasons.append(f"joint insert: {e}")
        penalty += 5.0
    for ins in inserts:
        chk = vase.check_insert(ins, settings)
        if not chk.ok:
            print_fail.append(f"{ins.name}: {','.join(chk.failures())}")
    if mission.require_printable and print_fail:
        reasons.extend(print_fail)

    # --- is the truss the buckling gate sized actually there? ---
    #
    # `structure.max_rib_pitch_mm` sizes the rib pitch from plate
    # buckling of the compression skin, and `ribs_that_fit` will return
    # ZERO for a panel whose chord is mostly openings -- correctly, since
    # webs squeezed into what is left print at 83 degrees. Nothing then
    # noticed that the pitch the structure had asked for was not
    # delivered: the trainer's centre body carried no truss at all while
    # the report went on quoting 26 mm, and the only symptom was 12 g of
    # mass that quietly went away.
    if settings.ribs:
        bare_of_ribs = [p_.name for p_ in panels if not p_.has_ribs]
        if bare_of_ribs:
            reasons.append(
                f"{', '.join(bare_of_ribs)} carr"
                f"{'ies' if len(bare_of_ribs) == 1 else 'y'} no rib truss: "
                f"the openings leave no chord for one, and the skin needs "
                f"support every {settings.rib_pitch_mm:.0f} mm")
            penalty += 6.0 * len(bare_of_ribs)

    root_c = plan.stations[0].chord_m
    # An item inside a bay sits where the bay sits. The trainer's
    # "AR630 + esc" bay was declared at 0.42c while the two masses it
    # contains were declared at 0.34c and 0.50c -- three stations for two
    # objects in one box, and the CG was computed from the wrong two.
    # ... and it sits at the bay's ABSOLUTE station. A servo's seat is a
    # fraction of the chord at its own eta, which is what the linkage
    # wants; read as a root-chord fraction it put 18 g of servos 30 mm
    # aft of where they are on the trainer.
    in_bay = {n: bay.name for bay in mission.bays for n in bay.holds}
    bay_x_abs = {v.name: v.x_abs_mm / (root_c * 1000.0) for v in bay_vols}
    items = tuple(
        replace(i, x_frac=bay_x_abs[in_bay[i.name]]).at(root_c)
        if i.name in in_bay else i.at(root_c)
        for i in mission.payload)
    items += (perf.PointMass(BATTERY_NAME, mission.battery_kg,
                             p_vec["batt_x"] * root_c),)
    # The tubes that were actually fitted, at the stations the fit solved
    # for. Not a flat allowance: see structure.spar_masses.
    for s_name, s_kg, s_x in struct.spar_masses(
            spar_fits, spar_od_mm if spar_od_mm is not None else mission.spar_d_mm,
            root_chord_mm=root_c * 1000.0):
        items += (perf.PointMass(f"spar {s_name}", s_kg, s_x * root_c),)
    if fins is not None:
        # at the tips and aft: they move the CG back and add roll inertia,
        # and both of those are part of what they cost
        xf, zf = fins.centroid()
        items += (perf.PointMass("tip fins",
                                 fins.mass_kg(settings.filament_density_gcc * 1000.0),
                                 xf, zf),)
    for ins in inserts:
        # one each side, at the joint
        c = ins.centroid_flight_mm() / 1000.0
        items += (perf.PointMass(f"joint {'fill' if ins.glue_fill else 'insert'} {ins.joint}",
                                 2.0 * ins.mass_g(settings.filament_density_gcc) / 1000.0,
                                 float(c[0]), float(c[2]), y_m=float(c[1])),)
    mass = perf.MassBudget(shell_kg=shell_kg,
                           shell_x_m=perf.shell_centroid_x(plan),
                           items=items)
    if mass.total_kg > mission.max_mass_kg:
        reasons.append(f"mass {mass.total_kg*1000:.0f} g over "
                       f"{mission.max_mass_kg*1000:.0f} g")

    # --- does the payload fit, and is anything else already there? ---
    #
    # Three gates, and the last two did not exist before. bay_fits asked
    # only "is the section deep enough here", which is necessary and
    # nowhere near sufficient: it passed every aircraft in the fleet while
    # an 8 mm carbon tube ran through the battery of all three.
    for bay in mission.bays:
        vol = next(v for v in bay_vols if v.name == bay.name)
        seat = bay_seats[bay.name]
        ok_bay, spare = vol.fits(plan, wall)
        if not ok_bay:
            where = f" at {seat:.2f}c"
            need = bay.box_mm[2]
            reasons.append(f"{bay.name} bay{where} {need + spare:.1f} mm deep, "
                           f"needs {need:.0f}")
            penalty += 20.0 * (-spare) / max(need, 1e-6)

        # straddling a print joint: the part would be in two shells
        first, last = it.straddles(vol, joint_etas)
        if last > first >= 0:
            reasons.append(f"{bay.name} spans print joints p{first}-p{last} "
                           f"(reaches eta {vol.eta1:.3f})")
            penalty += 25.0


    for a_name, b_name, mm in it.clashes(
            bay_vols + [it.spar_volume(f, wall, plan) for f in spar_fits],
            plan, wall):
        reasons.append(f"{a_name} and {b_name} share {mm:.1f} mm of space")
        # per millimetre of interpenetration: a 1 mm graze must rank above
        # a 12 mm tube through the pack, or the optimizer sees a cliff
        penalty += 8.0 * mm

    # The gates that are GEOMETRY, asked before trim is attempted. None of
    # them depends on the flight condition, and a design that fails to trim
    # returns early -- so asked after trim they went unreported on exactly
    # the designs whose report matters most: micro's servo shaft sat 81 mm
    # inboard of its elevon and nothing said so once it stopped trimming.
    # --- spar gates ---
    #
    # A spar aft of the hinge line is a spar inside the control surface:
    # the elevon is a separate printed part now, so a tube there has
    # nothing to run through and the surface cannot move. Newly askable --
    # before the hinge line reached the geometry there was no line to be
    # aft of.
    #
    # Asked along the tube, at every station it occupies outboard of the
    # hinge station: a straight tube's chord fraction drifts as the wing
    # sweeps and tapers under it, so being forward of the hinge at the
    # root says nothing about the tip.
    x_hinge = elv.hinge_x(settings) if elv.has_elevon(settings) else 1.0
    for f in spar_fits:
        if f.reach_eta <= settings.elevon_eta:
            continue
        over, at = -np.inf, 0.0
        for e in np.linspace(settings.elevon_eta, f.reach_eta, 9):
            c_mm = plan.at(float(e)).chord_m * 1000.0
            hx = 0.5 * f.spec.d_mm * np.sqrt(1.0 + f.slope[0] ** 2)
            o = (f.x_frac_at(plan, float(e)) - x_hinge) * c_mm + hx
            if o > over:
                over, at = o, float(e)
        if over > 0.0:
            reasons.append(f"{f.spec.name} is {over:.1f} mm aft of the hinge "
                           f"line ({x_hinge:.2f}c) at eta {at:.2f}")
            penalty += 10.0 * over

    for f in spar_fits:
        if f.joints_blocked:
            reasons.append(f"{f.spec.name}: a straight tube cannot reach the "
                           f"joint{'s' if len(f.joints_blocked) > 1 else ''} at eta "
                           + ",".join(f"{e:.2f}" for e in f.joints_blocked)
                           + f" (reaches {f.reach_eta:.2f})")
            penalty += 30.0 * len(f.joints_blocked)
        if f.reach_eta < mission.min_spar_reach_frac:
            reasons.append(f"{f.spec.name} reaches only eta {f.reach_eta:.2f} "
                           f"(need {mission.min_spar_reach_frac:.2f})")
            penalty += 40.0 * (mission.min_spar_reach_frac - f.reach_eta)

    # --- can the propeller swing without striking the wing? ---
    #
    # A pusher at the trailing edge of a SWEPT wing loses its clearance
    # outboard: the trailing edge runs aft as the disc runs out, so it is
    # the blade tips that are in danger and not the root. The motor was a
    # point mass at 0.97c and nothing else until the mount was generated,
    # and two of the three aircraft turned out to swing their declared
    # five-inch prop within a couple of millimetres of their own trailing
    # edge -- micro 2.0 mm, demon1 0.3 mm, against the 10 mm a mount
    # bonded to foam should keep.
    if mission.powertrain is not None:
        clear = pmod.prop_clearance_mm(plan, p_vec["prop_diam_in"])
        if clear < PROP_CLEARANCE_MM:
            reasons.append(
                f"prop disc clears the trailing edge by {clear:.1f} mm, "
                f"needs {PROP_CLEARANCE_MM:.0f}")
            penalty += 2.0 * (PROP_CLEARANCE_MM - clear)

    # --- and can the MECHANISM deliver it? ---
    #
    # dcm_ddeg, below, says how much moment a degree of elevon buys, and
    # the speed objective spends `max_elevon_deflect_deg` of it. Nothing
    # until now asked whether the servo, its arm, the pushrod and the horn
    # can reach that angle: demon1's headline speed was computed from a
    # deflection the aircraft had never been shown able to make.
    #
    # Solved as a four-bar rather than by the ratio r_servo / r_horn,
    # which is the small-angle limit of a parallel linkage and is wrong in
    # the two ways that matter -- it is linear, so it cannot show a
    # mechanism running out of travel, and it is symmetric, so it cannot
    # show the differential a real linkage has.
    # `link` was solved above, with the mechanism.
    if link is not None and horn_at is not None:
        _, e_shaft, applied = horn_at
        e_horn = max(e_shaft, float(settings.elevon_eta))
        servo_bay = next(b for b in mission.bays if b.name == "servos")
        half_mm = plan.half_span_m * 1000.0
        if servo_bay.drives_elevon and e_shaft < settings.elevon_eta - 1e-9:
            short = (settings.elevon_eta - e_shaft) * half_mm
            reasons.append(
                f"servo shaft at eta {e_shaft:.2f} is {short:.0f} mm inboard "
                f"of the elevon it drives (starts at eta "
                f"{settings.elevon_eta:.2f})")
            penalty += 0.25 * short
        # A lock inside the servo's travel is geometry, not a fault, so
        # long as it comes well AFTER the deflection the score spends: a
        # 17 mm horn on an 11 mm arm always locks before 54 degrees of
        # servo, and the transmitter's endpoints are what stop the servo
        # short of it. A lock BEFORE the wanted deflection is the fault.
        want = mission.max_elevon_deflect_deg
        ok_link, why = lkg.delivers(link, want)
        if not ok_link:
            thr_down, thr_up, _ = lkg.sweep(link)
            got = min(thr_down, -thr_up)
            reasons.append(why)
            penalty += 8.0 * max(want - got, 1.0)

    # --- trim fixes CL; CL fixes cruise speed ---
    vlm = _cached_vlm(u, mission, base, plan, ns, nc, fins)
    x_np = perf.neutral_point(vlm, plan)
    sm = (x_np - mass.x_cg_m) / plan.mac_m
    # The bracket is a SOLVER detail, not a design constraint. A slow
    # heavily-cambered trainer legitimately trims near 10 deg, and the
    # old [-6, 12] window was quietly rejecting exactly the aircraft this
    # mission asks for -- 154 of 216 candidates in one scan. What makes a
    # high trim angle unacceptable is proximity to the stall, and that is
    # already gated properly by cl_max_section below.
    alpha = vlm.trim_alpha(mass.x_cg_m, bounds=(-10.0, 18.0))

    def untrimmed(score: float) -> Evaluation:
        # Rejected at trim, but everything BEFORE trim was computed and is
        # true: the parts, the settings they were cut with, the tubes, the
        # mechanism, the mass. They used to be dropped, so exporting a
        # design that does not trim -- which is how you find out WHY --
        # crashed on settings that were never handed back, and every
        # check of the parts silently skipped it.
        return Evaluation(
            False, score, reasons=tuple(reasons), plan=plan, mass=mass,
            static_margin=sm, mass_kg=mass.total_kg,
            panels=panels if want_panels else [],
            elevons=elevon_parts if want_panels else [],
            horn_eta=float(horn_at[0]) if horn_at else 0.0,
            spar_fits=spar_fits, fairness=fair,
            fairness_limits=mission.fairness, fins=fins, inserts=inserts,
            max_elevon_deflect_deg=mission.max_elevon_deflect_deg,
            linkage=link,
            sm_band=(mission.min_static_margin, mission.max_static_margin),
            cruise_band=tuple(mission.cruise_band_ms),
            max_loading_gdm2=mission.max_wing_loading_gdm2,
            print_settings=settings)

    if alpha is None:
        reasons.append("no trim angle in [-10, 18] deg")
        return untrimmed(-1e5 - 10 * len(reasons))
    pt = vlm.solve(alpha, mass.x_cg_m)

    if pt.CL <= 0.02:
        reasons.append(f"trims at CL {pt.CL:.3f}: cannot support itself")
        return untrimmed(-1e5)

    v = float(np.sqrt(2 * mass.total_kg * perf.G
                      / (perf.RHO_AIR * plan.area_m2 * pt.CL)))
    try:
        cd0 = drag.cd0(plan, v, aero_point=pt)     # measured: per-strip cl
    except TypeError:
        cd0 = drag.cd0(plan, v)                    # tier-0: flat
    cd0_wing = cd0
    if fins is not None:
        cd0 = cd0 + fins.cd0(v, plan.area_m2)
    cd = cd0 + pt.CDi
    ld = pt.CL / cd

    # Violations are scored by HOW FAR out they are, not merely that they
    # happened. A design 0.5 m/s outside the cruise band must rank above
    # one 8 m/s outside it, or the optimizer sees a flat cliff instead of
    # a slope and never finds its way back into the feasible set.
    #
    # `penalty` is NOT reset here. It was, for the whole life of the
    # interior gates: every millimetre of clash, depth shortfall, joint
    # straddle and closure failure priced above this line was thrown
    # away before the score, so a 1 mm graze and a 12 mm tube through the
    # pack scored the same -- exactly the cliff this comment warns about.

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
    lat = lateral.analyse(plan, mass.x_cg_m, float(pt.CL))
    if fins is not None:
        # the static gates must see the fins too, or a fin could never
        # help a design past them
        _, _, _, z_cg = dyn.inertia(plan, mass, alpha, p_vec["elevon_eta"], fins)
        d_fin = fins.derivatives(plan.area_m2, plan.span_m, mass.x_cg_m, z_cg, alpha)
        lat = replace(lat, cn_beta=lat.cn_beta + d_fin[2, 0],
                      cl_beta=lat.cl_beta + d_fin[1, 0],
                      cn_beta_fin=lat.cn_beta_fin + d_fin[2, 0])
    if mission.min_cn_beta > 0.0:
        if lat.cn_beta < mission.min_cn_beta:
            reasons.append(f"Cn_beta {lat.cn_beta:+.4f} below "
                           f"{mission.min_cn_beta:.4f} (wanders in yaw)")
            penalty += 300.0 * (mission.min_cn_beta - lat.cn_beta)
        if lat.cl_beta > 0.0:
            reasons.append(f"Cl_beta {lat.cl_beta:+.4f} positive "
                           f"(rolls INTO the sideslip)")
            penalty += 300.0 * lat.cl_beta
        if lat.roll_yaw_ratio > mission.max_roll_yaw_ratio:
            reasons.append(f"roll/yaw ratio {lat.roll_yaw_ratio:.1f} over "
                           f"{mission.max_roll_yaw_ratio:.1f} (dutch roll)")
            penalty += 4.0 * (lat.roll_yaw_ratio - mission.max_roll_yaw_ratio)
    # --- does the Dutch roll die out? the real question, not the proxy ---
    modes = dyn.analyse(vlm, plan, mass, alpha, mass.x_cg_m, v, cd0_wing,
                        p_vec["elevon_eta"], fins)
    if mission.min_dutch_roll_zeta is not None:
        # an overdamped (non-oscillatory) Dutch roll is as good as it gets
        z_dr = modes.zeta_dr if modes.zeta_dr is not None else 1.0
        if z_dr < mission.min_dutch_roll_zeta:
            reasons.append(f"Dutch roll zeta {z_dr:+.3f} below "
                           f"{mission.min_dutch_roll_zeta:.2f}"
                           + (" (grows)" if z_dr < 0.0 else ""))
            penalty += 200.0 * (mission.min_dutch_roll_zeta - z_dr)
    if mission.min_spiral_t2_s > 0.0 and modes.spiral_t2_s < mission.min_spiral_t2_s:
        reasons.append(f"spiral doubles in {modes.spiral_t2_s:.1f} s "
                       f"(need >= {mission.min_spiral_t2_s:.0f})")
        penalty += 20.0 * (mission.min_spiral_t2_s - modes.spiral_t2_s) / mission.min_spiral_t2_s
    if pt.CL < mission.min_cl_trim:
        reasons.append(f"trims hands-off at CL {pt.CL:.3f}, below "
                       f"{mission.min_cl_trim:.2f}")
        penalty += 60.0 * (mission.min_cl_trim - pt.CL)

    # --- and do the BONDED joints carry what the spar does not? ---
    #
    # Panels butt together on the spar and are glued face to face. The
    # tube is sized for bending; the ring of single wall at each joint
    # carries the shear outboard of it and the torque of that lift about
    # the spar line, and until now nothing asked whether it could. The
    # allowable is declared, deliberately far below any figure for the
    # adhesive itself, because the foam is the weak side.
    fleet_joints = ()
    if spar_fits and len(joint_etas) > 1:
        x_axis = float(np.mean([f.x_frac for f in spar_fits]))
        loads_j = struct.span_loads(plan, pt, mass.total_kg, mission.n_limit_g)
        fleet_joints, j_gates = jnt.gates(plan, loads_j, joint_etas,
                                          x_axis, wall)
        for g in j_gates:
            if not g.passed:
                reasons.append(f"{g.name} carries {g.value:.1f}x its declared "
                               f"allowable, needs {g.limit:.0f}x")
                penalty += 10.0 * max(g.limit - g.value, 0.0)

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

    # --- the slowest it can be FLOWN, and how it stalls there ---
    # The elevon is a second right-hand side of the same lattice, so this
    # is three triangular solves, not a rebuild. On the micro_fpv winner
    # the minimum speed moves 0.4% from nc 8 to nc 32 (test_validation).
    slow = None
    if (mission.objective == "docile" or mission.min_hands_off_margin > 0.0
            or mission.stall_onset_ahead_of_cg):
        dn = vlm.elevon_dn(p_vec["elevon_eta"], p_vec["elevon_chord"])
        slow = perf.slow_flight(vlm, mass.x_cg_m, alpha, dn, p_vec["elevon_eta"],
                                mission.cl_max_section,
                                mission.max_elevon_deflect_deg, mass.total_kg)
        if mission.min_hands_off_margin > 0.0:
            ratio = v / slow.v_min_ms
            if ratio < mission.min_hands_off_margin:
                reasons.append(
                    f"hands-off {v:.1f} m/s is only {ratio:.2f}x the slowest "
                    f"trimmed {slow.v_min_ms:.1f} m/s "
                    f"(need {mission.min_hands_off_margin:.2f}x)")
                penalty += 30.0 * (mission.min_hands_off_margin - ratio)
        # Checked even when the elevon runs out first: that stops the
        # pilot holding the wing stalled, not a gust or a pull-out
        # taking it there, and where it lets go is the same place.
        if mission.stall_onset_ahead_of_cg:
            st_c = plan.at(slow.eta_critical)
            aft = (st_c.x_le_m + 0.25 * st_c.chord_m - mass.x_cg_m) / plan.mac_m
            if aft > 0.0:
                reasons.append(
                    f"stall starts at eta {slow.eta_critical:.2f}, "
                    f"{aft:.2f} MAC behind the CG -- pitches UP at the stall")
                penalty += 20.0 * aft

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
    for p_stack, chk in zip(panels + elevon_parts, checks):
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
                    cd_v = cd0_wing
                cdi_v = cl_v**2 / (np.pi * plan.aspect_ratio * 0.85)
                if fins is not None:
                    cd_v = cd_v + fins.cd0(vv, plan.area_m2)
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
    elif mission.objective == "docile":
        # SLOW is the objective, and stall speed is the one number that
        # says it. Everything a nervous pilot complains about scales with
        # it: the speed the aeroplane arrives at the ground, how far it
        # travels while they think, and how much energy a mistake has.
        #
        # It is not a weighted sum of the stability derivatives, and that
        # is deliberate -- those are GATES, and feasibility dominates the
        # score, so a design that misses Dutch roll damping cannot buy its
        # way back with a slower stall. What the merit does is pick, among
        # designs that are already stable enough, the one that flies
        # slowest.
        #
        # v_stall = sqrt(2W / rho S CLmax), so minimising it minimises
        # wing loading: it pushes mass DOWN and area UP at the same time,
        # which is why the 250 g ceiling is a gate rather than a target.
        # A design has no reason to spend mass it does not need.
        #
        # It is the TRIMMED minimum speed: the first section to reach
        # cl_max, or the up-elevon running out, whichever comes first. The
        # flat version -- every section at cl_max at once, nothing spent
        # on trim -- reported the gen8 winner at 6.40 m/s against 7.38 for
        # the aircraft as flown, and it could not tell a wing whose lift
        # was well spread from one whose peak stalled early.
        merit = -slow.v_min_ms
    else:
        merit = ld

    # --- does the wing twist itself apart, or reverse its own elevons? ---
    #
    # Placed HERE, after the merit, because the speed that matters is the
    # speed the design is SCORED at: top speed for a racer, cruise for a
    # trainer. The objective is what pushes the aircraft toward the
    # failure, so it is what the margin has to be measured against.
    #
    # The lift slope comes from the lattice rather than from 2*pi/AR
    # algebra: it is already built and factorised, so a second right-hand
    # side is nearly free.
    aero_e = None
    if mission.min_aeroelastic_margin > 0.0:
        pt2 = vlm.solve(alpha + 2.0, mass.x_cg_m)
        cl_a = (pt2.CL - pt.CL) / np.radians(2.0)
        v_design = v_top if mission.objective == "speed" else v
        aero_e = ael.analyse(
            plan, spar_fits, cl_a, p_vec["elevon_chord"], p_vec["elevon_eta"],
            v_design, settings.extrusion_width_mm,
            min_margin=mission.min_aeroelastic_margin,
            n_ribs=settings.rib_count if settings.ribs else 0)
        if not aero_e.ok:
            worst = min(aero_e.v_div_ms, aero_e.v_rev_ms)
            which = ("reversal" if aero_e.v_rev_ms <= aero_e.v_div_ms
                     else "divergence")
            reasons.append(
                f"{which} at {worst:.0f} m/s, only {aero_e.margin:.2f}x the "
                f"{v_design:.0f} m/s this design is scored at")
            penalty += 30.0 * (mission.min_aeroelastic_margin - aero_e.margin)

    score = merit if not reasons else -(1000.0 + penalty)
    return Evaluation(
        ok=not reasons, score=float(score), ld=float(ld), v_cruise=v,
        mass_kg=mass.total_kg, cl_trim=float(pt.CL), static_margin=float(sm),
        reasons=tuple(reasons), plan=plan, trim=trim_state, mass=mass,
        panels=panels if want_panels else [],
        elevons=elevon_parts if want_panels else [],
        horn_eta=float(horn_at[0]) if horn_at else 0.0,
        spar_fits=spar_fits, lateral=lat,
        fairness=fair, fairness_limits=mission.fairness,
        fins=fins, slow=slow, powertrain=mission.powertrain,
        inserts=inserts, dynamics=modes,
        max_elevon_deflect_deg=mission.max_elevon_deflect_deg,
        linkage=link,
        aeroelastic=aero_e,
        joints=tuple(fleet_joints),
        sm_band=(mission.min_static_margin, mission.max_static_margin),
        cruise_band=tuple(mission.cruise_band_ms),
        max_loading_gdm2=mission.max_wing_loading_gdm2,
        # The settings this verdict was actually reached with, including
        # the elevon station and the spar corridors set above. They used
        # to be a local variable, so `evaluate` handed `choose_structure`
        # the CALLER's settings and the exporter got those back -- with
        # spar_avoid empty and the elevon chord at zero. The search kept
        # the rib truss clear of the spar corridors and cut the trailing
        # edge off; the export did neither, and printed ribs straight
        # through both tubes and a wing with its elevons still attached.
        print_settings=settings,
    )


# ------------------------------------------------------------------ structure


def choose_structure(ev: Evaluation, mission: Mission,
                     settings: vase.PrintSettings,
                     ns: int = LATTICE_NS, nc: int = LATTICE_NC, vlm=None):
    """Size the spar and the rib pitch from the trimmed flight condition,
    and hand back the print settings that follow from them.

    It lives here, and not in run.py, because the search has to score the
    aircraft it is going to export. It used to run only in `run.py export`,
    after the search had finished -- so every search in four generations
    judged a bare vase shell. The buckling-driven ribs it adds are 8 to 22 g,
    mostly aft of the CG; on the gen4 trainer they pushed trim past 8
    degrees and cut Dutch-roll damping from +0.083 to +0.058. The search
    called that design feasible and the built aircraft was not."""
    pt = (vlm or VLM(ev.plan, ns, nc, fins=ev.fins)).solve(ev.trim.alpha_deg, ev.trim.x_cg_m)
    st = struct.select(ev.plan, pt, ev.mass_kg,
                       skin_t_mm=settings.extrusion_width_mm,
                       n_limit=mission.n_limit_g,
                       min_od_mm=mission.spar_d_mm)
    tuned = replace(settings, spar_d_mm=st.spar.od_mm, ribs=True,
                    rib_pitch_mm=st.rib_pitch_mm)
    return st, tuned


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
    size_structure: bool = True,
) -> Evaluation:
    """One design -> one verdict, for the aircraft as it will be BUILT.

    Two passes whenever the design trims. The first, as a bare shell, finds
    the trimmed loads the spar and ribs are sized from; the second puts that
    spar and those ribs into the print settings and judges everything again
    -- shell mass, CG, trim, damping and every gate -- on the ribbed
    aircraft. That is exactly what `run.py export` does, so the search and
    the export now reach the same verdict by construction.

    Designs rejected before trim never pay for the second pass. Pass
    size_structure=False for a bare-shell verdict on purpose."""
    # Parts are built on both passes; asking for them on the first costs
    # only the keeping. It matters when the first pass is the LAST one --
    # a design that does not trim never gets a second -- and its parts
    # are what someone exporting it to see why needs.
    first = _evaluate_once(u, mission, base, settings, drag=drag, ns=ns, nc=nc,
                           want_panels=want_panels, z_step_mm=z_step_mm)
    if not size_structure or first.trim is None or first.plan is None:
        return first
    st, tuned = choose_structure(first, mission, first.print_settings or settings,
                                 ns, nc,
                                 vlm=_cached_vlm(u, mission, base, first.plan, ns, nc,
                                                  first.fins))
    ev = _evaluate_once(u, mission, base, tuned, drag=drag, ns=ns, nc=nc,
                        want_panels=want_panels, z_step_mm=z_step_mm,
                        spar_od_mm=st.spar.od_mm)
    # `ev.print_settings` is the second pass's OWN settings, not `tuned`:
    # the pass adds the elevon station and re-solves the spar corridors
    # for the sized tube, and those are the settings the export has to
    # use. Overwriting them with `tuned` is what silently dropped both.
    ev = replace(ev, structure=st)
    if not st.ok:
        why = "; ".join(st.notes) or "spar overstressed or too flexible"
        ev = replace(ev, ok=False, reasons=ev.reasons + (f"structure: {why}",),
                     score=min(ev.score, -1000.0) - 25.0)
    return ev
