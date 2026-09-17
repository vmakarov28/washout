"""Nothing may share space with the battery.

Three checks used to ask three separate questions -- `bay_fits` "is the
section deep enough here", `spars.fit_all` "can a tube reach that far",
`vase.panel_etas` "does this panel fit the envelope" -- and none of them
knew the other two existed. So nothing ever asked whether the tube and
the pack wanted the same place, and on every aircraft in the fleet they
did:

    trainer   pack 0.291c-0.586c    TE spar   at 0.540c
    demon1    pack 0.197c-0.570c    LE spar   at 0.210c
    micro     pack 0.239c-0.582c    main spar at 0.363c, and the rx too

It was not detectable, and not for want of looking: `spars.place` solved
a chordwise station and nothing else, so the spar had no vertical
coordinate at all. A question about two objects overlapping cannot be put
to an object with two coordinates and a shrug.

Each test here pins one part of the fix.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from washout.geom import cst
from washout.geom import interior as it
from washout.printing import vase
from washout import spars as sp
from washout.search.design import (MISSIONS, Mission, N_DIM, build, evaluate,
                                   seat_bays, unit_to_physical)

ASSETS = Path(__file__).resolve().parent.parent / "assets"
RESULTS = Path(__file__).resolve().parent.parent / "results" / "fleet"
WALL = 0.45


def _fleet(name):
    index = json.loads((RESULTS / "index.json").read_text(encoding="utf-8"))
    d = json.loads((RESULTS / index[name] / "design.json").read_text(encoding="utf-8"))
    base = cst.load_selig(ASSETS / "mh45.dat")
    mission = getattr(Mission, name)()
    return np.array(d["u"]), mission, base, build(np.array(d["u"]), mission, base)


# ------------------------------------------------------ the frame itself

def test_depth_is_measured_from_the_lower_inner_skin():
    """z = 0 is the surface a part rests on, and [0, depth] is the room."""
    _, _, _, plan = _fleet("trainer_v3")
    d = it.depth_mm(plan, 0.0, 0.30, WALL)
    box = it.Volume("box", 0.28, 0.32, 0.0, 0.05, height_mm=10.0,
                    anchor=it.LOWER, offset_mm=WALL)
    lo, hi = box.z_interval(plan, 0.0, 0.30, WALL)
    assert lo == pytest.approx(WALL)
    assert hi == pytest.approx(WALL + 10.0)
    assert hi < d, "a part that fits must end below the upper inner skin"


def test_an_upper_anchored_part_follows_the_skin_outboard():
    """A fixed height above the lower skin is through the upper one at the
    tip. Anchoring is what keeps a part on the surface it rests on as the
    section thins."""
    _, _, _, plan = _fleet("trainer_v3")
    v = it.Volume("v", 0.30, 0.34, 0.0, 0.9, height_mm=6.0, anchor=it.UPPER)
    root_lo, _ = v.z_interval(plan, 0.0, 0.32, WALL)
    mid_lo, _ = v.z_interval(plan, 0.5, 0.32, WALL)
    assert root_lo > mid_lo, "the upper skin is lower outboard, so the seat is"
    d_root = it.depth_mm(plan, 0.0, 0.32, WALL)
    _, root_hi = v.z_interval(plan, 0.0, 0.32, WALL)
    assert root_hi == pytest.approx(d_root), "an upper part touches the skin"


def test_a_section_too_shallow_reports_nothing_there():
    """An empty interval, not a clipped one: a caller asking about a
    station outboard of where the part fits must get 'nothing here'
    rather than a silently squashed answer that would test as clear."""
    _, _, _, plan = _fleet("micro")
    v = it.Volume("fat", 0.30, 0.34, 0.0, 1.0, height_mm=500.0)
    lo, hi = v.z_interval(plan, 0.9, 0.32, WALL)
    assert hi < lo


def test_two_parts_stacked_in_the_depth_do_not_clash():
    """Overlapping in chord and span is not enough to collide, and this is
    exactly how the trainer's TE spar and its pack coexist: the tube takes
    the upper skin, the pack sits on the lower one, with under 2 mm to
    spare. Before the vertical coordinate existed that could not be said
    either way."""
    _, _, _, plan = _fleet("trainer_v3")
    # Sized against the THINNEST station in the band, not the thickest:
    # the trainer's centre body runs 47.7 mm deep at 0.30c and 34.1 mm at
    # 0.60c, so two parts that stack clear at max thickness can still meet
    # where the section has fallen away. That is a real collision and the
    # check is right to report it -- which is why the heights here are
    # chosen against 32.5 mm rather than 44.2.
    thin = min(it.depth_mm(plan, e, x, WALL)
               for e in (0.0, 0.08) for x in (0.30, 0.45, 0.60))
    h = 0.4 * thin
    low = it.Volume("low", 0.30, 0.60, 0.0, 0.08, height_mm=h, anchor=it.LOWER)
    high = it.Volume("high", 0.30, 0.60, 0.0, 0.08, height_mm=h, anchor=it.UPPER)
    assert it.overlap_mm(low, high, plan, WALL) == 0.0
    # the same two parts, both on the floor, certainly do
    also_low = it.Volume("also", 0.30, 0.60, 0.0, 0.08,
                         height_mm=h, anchor=it.LOWER)
    assert it.overlap_mm(low, also_low, plan, WALL) > 0.0


def test_parts_that_stack_at_the_root_can_still_meet_outboard():
    """Sized against max thickness, two stacked parts collide where the
    section has thinned. The check must see that, and it is the reason a
    Volume carries an anchor rather than an absolute z."""
    _, _, _, plan = _fleet("trainer_v3")
    h = 0.45 * it.depth_mm(plan, 0.0, 0.45, WALL)     # fits at 0.45c...
    low = it.Volume("low", 0.30, 0.60, 0.0, 0.08, height_mm=h, anchor=it.LOWER)
    high = it.Volume("high", 0.30, 0.60, 0.0, 0.08, height_mm=h, anchor=it.UPPER)
    assert it.overlap_mm(low, high, plan, WALL) > 0.0  # ...but not at 0.60c


def test_parts_at_different_chord_stations_never_clash():
    _, _, _, plan = _fleet("trainer_v3")
    fore = it.Volume("fore", 0.10, 0.20, 0.0, 0.08, height_mm=10.0)
    aft = it.Volume("aft", 0.50, 0.60, 0.0, 0.08, height_mm=10.0)
    assert it.overlap_mm(fore, aft, plan, WALL) == 0.0


def test_overlap_is_graded_by_how_far_in_it_reaches():
    """A 1 mm graze must rank above a 12 mm tube through the pack, or the
    optimizer sees a cliff instead of a slope back into the feasible set."""
    _, _, _, plan = _fleet("trainer_v3")
    # One part on the floor, and two intruders coming DOWN from the upper
    # skin by different amounts. Graded by the vertical overlap, which is
    # the axis that actually differs -- overlap_mm returns the minimum
    # separating translation, so a part wholly enclosed in another reports
    # its own height whatever the enclosure's size.
    d = min(it.depth_mm(plan, e, x, WALL)
            for e in (0.0, 0.08) for x in (0.30, 0.60))
    floor = it.Volume("floor", 0.30, 0.60, 0.0, 0.08,
                      height_mm=0.6 * d, anchor=it.LOWER)
    deep = it.Volume("deep", 0.30, 0.60, 0.0, 0.08,
                     height_mm=0.7 * d, anchor=it.UPPER)
    graze = it.Volume("graze", 0.30, 0.60, 0.0, 0.08,
                      height_mm=0.45 * d, anchor=it.UPPER)
    assert (it.overlap_mm(floor, deep, plan, WALL)
            > it.overlap_mm(floor, graze, plan, WALL) > 0.0)


# ---------------------------------------------- the spar has a seat now

def test_the_spar_is_given_a_vertical_seat():
    """`place` used to solve a chordwise station and nothing else."""
    _, mission, _, plan = _fleet("trainer_v3")
    fits = sp.fit_all(plan, mission.spars, WALL)
    assert fits, "the trainer declares spars"
    for f in fits:
        assert f.anchor in it.ANCHORS, f"{f.spec.name} has no seat"


def test_the_spar_seat_is_solved_against_what_is_already_there():
    """A tube offered a corridor through the battery must take the other
    skin. This is the fix: not a new penalty, a seat that avoids it."""
    _, mission, _, plan = _fleet("trainer_v3")
    pack = next(b for b in mission.bays if b.x_var)
    # a pack sitting on the floor across the whole chord band the TE spar
    # may use, so the only clear seat is the upper skin
    blocker = it.bay_volume("blocker", 0.54, pack.box_mm, plan, offset_mm=WALL)
    spec = mission.spars[-1]              # the TE corridor
    free = sp.place(plan, spec, WALL)
    held = sp.place(plan, spec, WALL, reserved=(blocker,))
    assert held.clash_mm <= free.clash_mm + 1e-9
    assert held.clash_mm == 0.0, (
        f"a clear seat exists and the solver must find it "
        f"(got {held.clash_mm:.2f} mm at {held.anchor})")


def test_two_spars_do_not_take_the_same_seat():
    """Spars are fitted in declaration order and each sees the ones before
    it, so two 8 mm tubes cannot both claim the same corner."""
    _, mission, _, plan = _fleet("trainer_v3")
    fits = sp.fit_all(plan, mission.spars, WALL)
    vols = [it.spar_volume(f, WALL, plan) for f in fits]
    assert it.clashes(vols, plan, WALL) == []


def test_the_fleet_spar_and_pack_conflict_is_now_visible():
    """The headline finding, pinned.

    Placed as the missions used to place them -- every bay on the floor at
    its declared nominal seat, the spar solved for reach alone -- an 8 mm
    tube shares space with the pack on all three aircraft. This test
    reproduces that state deliberately, so that the detection cannot
    regress even after the seats and the solver have fixed the designs."""
    found = {}
    for name in MISSIONS:
        u, mission, base, plan = _fleet(name)
        p = unit_to_physical(u)
        vols = [it.bay_volume(b.name, p[b.x_var] if b.x_var else b.x_frac,
                              b.box_mm, plan, offset_mm=WALL)
                for b in mission.bays]
        # the OLD spar fit: reach only, no seat, no reserved volumes
        naive = [sp.place(plan, s, WALL) for s in mission.spars]
        naive = [it.Volume(f.spec.name, f.x_frac - 0.5 * f.spec.d_mm
                           / (plan.stations[0].chord_m * 1000.0),
                           f.x_frac + 0.5 * f.spec.d_mm
                           / (plan.stations[0].chord_m * 1000.0),
                           0.0, f.reach_eta, f.spec.d_mm, anchor=it.MID)
                 for f in naive]
        pack = mission.bays[0].name
        found[name] = [(a, b, mm) for a, b, mm in it.clashes(vols + naive, plan, WALL)
                       if pack in (a, b) and any(
                           s.name in (a, b) for s in mission.spars)]
    for name in MISSIONS:
        assert found[name], f"{name}: the spar-through-the-pack clash vanished"


# ------------------------------------------------------- bays and joints

def test_a_bay_may_not_straddle_a_print_joint():
    """Micro's 2S 450 reaches eta 0.170 and panel p0 ends at 0.140, so the
    pack was declared across the joint between two separately printed
    shells. There is no geometry that makes that work."""
    _, mission, _, plan = _fleet("micro")
    s = vase.PrintSettings(bed_z_mm=250.0, spar_d_mm=8.0)
    joints = vase.panel_etas(plan, s)
    wide = it.bay_volume("wide", 0.40, (40.0, 200.0, 10.0), plan)
    first, last = it.straddles(wide, joints)
    assert last > first >= 0, "a 200 mm wide bay must cross a joint"
    narrow = it.bay_volume("narrow", 0.40, (40.0, 8.0, 10.0), plan)
    f2, l2 = it.straddles(narrow, joints)
    assert f2 == l2 == 0, "a bay at the centreline lives in p0 alone"


def test_a_bay_holds_its_contents_at_one_station():
    """The trainer's "AR630 + esc" bay was declared at 0.42c while the two
    masses it contains sat at 0.34c and 0.50c -- three stations for two
    objects in one box, and the CG was computed from the wrong two."""
    u, mission, base, plan = _fleet("trainer_v3")
    p = unit_to_physical(u)
    _, seats = seat_bays(plan, mission, p, WALL)
    bay = next(b for b in mission.bays if b.holds)
    root_c = plan.stations[0].chord_m

    settings = vase.PrintSettings(filament_density_gcc=0.55, spar_d_mm=8.0)
    ev = evaluate(u, mission, base, settings)
    held = {i.name: i for i in ev.mass.items if i.name in bay.holds}
    assert len(held) == len(bay.holds), f"expected {bay.holds}, got {list(held)}"
    xs = {round(i.x_m / root_c, 6) for i in held.values()}
    assert len(xs) == 1, f"one bay, one station, got {xs}"


def test_the_solver_seats_the_electronics_clear_of_the_pack():
    """No value of batt_x in its whole range could separate them.

    The trainer's pack and its electronics were declared in the same 16 mm
    of space: where the pack fits in depth it overlaps the electronics
    bay, and where it clears them chordwise the section is too shallow.
    The second bay's seat therefore has to be solved, not declared."""
    u, mission, base, plan = _fleet("trainer_v3")
    p = unit_to_physical(u)
    vols, seats = seat_bays(plan, mission, p, WALL)
    assert it.clashes(vols, plan, WALL) == [], (
        "the solver must find seats that do not overlap")
    pack, elec = mission.bays[0], mission.bays[1]
    assert seats[pack.name] == pytest.approx(p[pack.x_var]), (
        "the optimizer still owns the pack: it is the trim lever")
    assert elec.x_lo <= seats[elec.name] <= elec.x_hi


def test_the_solved_seat_is_the_seat_the_mass_uses():
    """`bay_fits` warns against a sweeping CHECK, and rightly: answering
    'does some seat exist' while modelling the mass elsewhere passed a
    design whose pack hung off the nose. A solved seat is different only
    if it is also the station the mass is placed at."""
    u, mission, base, plan = _fleet("trainer_v3")
    p = unit_to_physical(u)
    vols, seats = seat_bays(plan, mission, p, WALL)
    settings = vase.PrintSettings(filament_density_gcc=0.55, spar_d_mm=8.0)
    ev = evaluate(u, mission, base, settings)
    root_c = plan.stations[0].chord_m
    for bay in mission.bays:
        for item_name in bay.holds:
            item = next(i for i in ev.mass.items if i.name == item_name)
            assert item.x_m == pytest.approx(seats[bay.name] * root_c, rel=1e-9)


def test_a_clash_is_penalised_by_how_far_it_reaches():
    """Random designs that clash must rank by severity, not fail flat."""
    base = cst.load_selig(ASSETS / "mh45.dat")
    m = Mission.micro()
    settings = vase.PrintSettings(filament_density_gcc=0.55, spar_d_mm=8.0)
    rng = np.random.default_rng(11)
    scores = []
    for _ in range(60):
        ev = evaluate(rng.random(N_DIM), m, base, settings,
                      size_structure=False)
        share = [r for r in ev.reasons if "share" in r]
        if share:
            scores.append(ev.score)
    if len(scores) > 1:
        assert len(set(scores)) > 1, (
            "every clashing design scored identically: the penalty is a "
            "cliff, not a slope")


# ------------------------------------------------ can the bay be closed?

def test_the_ramp_budget_is_quadrature_not_subtraction():
    """The distinction decides whether a battery bay is possible at all.

    A feature ramping in the THICKNESS direction adds its own dy to
    whatever dx that contour point already carries from the wing's taper
    and sweep, and `overhang_deg` takes the nearest-point distance, so the
    bound is hypot(dx, dy)/dz <= tan(theta). Subtracting linearly gives
    the trainer's centre body 0.36 mm/mm -- a 20 mm feature would need
    56 mm of span and would not fit the 111 mm panel. In quadrature the
    same panel has about 0.8 mm/mm and needs 25 mm, which it has."""
    _, mission, _, plan = _fleet("trainer_v3")
    s = vase.PrintSettings(bed_z_mm=250.0, spar_d_mm=8.0)
    p0 = vase.build_panels(plan, s)[0]

    rate = vase.ramp_budget(p0, 0.12, 0.70)
    lim = np.tan(np.radians(s.max_overhang_deg))
    ang, _ = vase.overhang_deg(p0)
    linear = lim - np.tan(np.radians(ang))

    assert 0.0 < rate <= lim
    assert rate > linear, (
        f"quadrature {rate:.3f} must beat subtraction {linear:.3f}; if it "
        f"does not, the wing is using the whole budget in one direction")
    assert vase.ramp_span_mm(p0, 20.0, 0.12, 0.70) < p0.height_mm, (
        "a 20 mm feature must fit one ramp in the trainer's centre body")


def test_the_ramp_budget_improves_outboard():
    """The wing's own taper and sweep slow down as the chord runs out, so
    the panels that have the least DEPTH have the most ramp rate."""
    _, mission, _, plan = _fleet("trainer_v3")
    s = vase.PrintSettings(bed_z_mm=250.0, spar_d_mm=8.0)
    pans = vase.build_panels(plan, s)
    rates = [vase.ramp_budget(p, 0.12, 0.70) for p in pans]
    assert rates[-1] > rates[0], f"{rates}"


def test_a_shallow_panel_cannot_close_a_deep_bay():
    """Micro's centre body is 24.5 mm tall and its pack is 17 mm deep.

    Closing the bay needs 33 mm of span at the rate available and there
    are 10 mm left outboard of the pack, so the bay cannot be closed
    inside p0 at all -- it has to run clear through the panel and be shut
    by the joint, or move outboard. That is a per-aircraft architectural
    consequence of the print constraint, and it is the reason this gate
    exists rather than a blanket rule about bay depth."""
    u, mission, base, plan = _fleet("micro")
    settings = vase.PrintSettings(filament_density_gcc=0.55, spar_d_mm=8.0,
                                  bed_z_mm=250.0)
    ev = evaluate(u, mission, base, settings)
    closes = [r for r in ev.reasons if "cannot be closed" in r]
    assert closes, f"micro p0 must fail the closure gate; got {ev.reasons}"
    assert any(mission.bays[0].name in r for r in closes)


def test_the_trainer_can_close_its_bays():
    """The gate must not reject an aircraft that has the room: the
    trainer's 111 mm centre body carries a 24 mm pack with ramps at both
    ends and 16 mm to spare."""
    u, mission, base, plan = _fleet("trainer_v3")
    settings = vase.PrintSettings(filament_density_gcc=0.55, spar_d_mm=8.0,
                                  bed_z_mm=250.0)
    ev = evaluate(u, mission, base, settings)
    assert not [r for r in ev.reasons if "cannot be closed" in r], ev.reasons
