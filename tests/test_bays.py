"""The payload bay is cut, not just reserved.

A `Bay` used to be volume: a box checked against the section's depth,
against the spars and against the print joints, and then not cut. The
battery went in through a hole someone made with a knife.

The cut is the rib detour of `printing/ribs.py` widened from a slit to a
chord band, and it inherits that module's invariants: constant point
count per layer, and every vertex one extrusion width from every
non-adjacent part of the contour. Four bugs came out of building it and
three of them were the same bug the rib code already documents, arriving
by different routes.
"""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from washout.geom import cst
from washout.printing import bays, vase
from washout.printing.ribs import RibSpec, min_clearance_mm, ribs_that_fit
from washout.search.design import (MISSIONS, Mission, build, evaluate,
                                   seat_bays, unit_to_physical)

ASSETS = Path(__file__).resolve().parent.parent / "assets"
RESULTS = Path(__file__).resolve().parent.parent / "results" / "fleet"
WALL = 0.45


def _plan(name):
    index = json.loads((RESULTS / "index.json").read_text(encoding="utf-8"))
    d = json.loads((RESULTS / index[name] / "design.json").read_text(encoding="utf-8"))
    base = cst.load_selig(ASSETS / "mh45.dat")
    mission = getattr(Mission, name)()
    u = np.array(d["u"])
    return u, mission, base, build(u, mission, base)


def _built(name):
    u, mission, base, plan = _plan(name)
    s = vase.PrintSettings(filament_density_gcc=0.55, spar_d_mm=8.0,
                           bed_z_mm=250.0,
                           max_overhang_deg=mission.max_overhang_deg or 50.0)
    return evaluate(u, mission, base, s, want_panels=True), mission


def _loop(plan, eta, settings):
    st = plan.at(eta)
    return (vase.thicken_for_nozzle(st.airfoil.coords(121),
                                    st.chord_m * 1000.0, settings),
            st.chord_m * 1000.0)


# ------------------------------------------------------- the detour itself

def test_the_bay_has_a_floor_with_closed_section_under_it():
    """Not a cut to the lower skin, and the difference is load-bearing.

    Cutting full depth opens the section over the whole band, which
    destroys the solution the spar seat solver had found: on the trainer
    the TE spar clears the pack by taking the upper skin while the pack
    sits on the lower one, and a full-depth cut removes the region the
    tube was seated in. A real bay has a floor, and the cell beneath it is
    where the spar lives and most of the torsion box survives."""
    _, _, _, plan = _plan("trainer_v3")
    s = vase.PrintSettings()
    loop, c = _loop(plan, 0.02, s)
    depth = 24.0
    lim = bays.floor_limits(loop, 0.30, 0.58, c, depth, WALL, WALL)
    assert lim is not None
    y_closed, y_open = lim
    assert (y_closed - y_open) * c == pytest.approx(depth, abs=0.01), (
        "the floor must sit exactly the box's depth below the skin")
    # and there must be section left underneath
    n = (len(loop) + 1) // 2
    lower = loop[n - 1:]
    sel = (lower[:, 0] >= 0.30) & (lower[:, 0] <= 0.58)
    assert y_open > lower[sel, 1].max(), "no closed cell left under the floor"


def test_the_floor_clears_the_lower_skin_beyond_the_band_too():
    """The clearance rule is vertex-to-SEGMENT, so the lower skin just
    past a corner is part of that corner's neighbourhood whether or not it
    is inside the band. On the trainer's root the lower surface just aft
    of the bay is 0.4 mm higher than anything inside it, and a floor
    placed against the in-band maximum left its corner 0.255 mm from a
    segment it was never measured against."""
    _, _, _, plan = _plan("trainer_v3")
    s = vase.PrintSettings()
    loop, c = _loop(plan, 0.02, s)
    spec = bays.BaySpec("b", 0.291, 0.586, span_mm=20.0, ramp_mm=30.0,
                        depth_mm=33.0, floor_gap_mm=WALL)
    got = bays.insert_bay(loop, c, 0.0, spec, WALL) * c
    assert min_clearance_mm(got, skip=8) >= WALL, (
        f"{min_clearance_mm(got, skip=8):.3f} mm is under one bead")


def test_the_point_count_is_constant_even_where_the_bay_cannot_be_cut():
    """The invariant has to be STRUCTURAL, not conditional.

    Returning the loop unchanged where the section cannot hold the bay
    left that layer two vertices short of every other layer in the panel,
    and the contour array would not broadcast. It only showed up on the
    panels carrying two bays, which is the ones that matter."""
    _, _, _, plan = _plan("micro")
    s = vase.PrintSettings()
    loop, c = _loop(plan, 0.02, s)
    roomy = bays.BaySpec("ok", 0.30, 0.50, 20.0, 30.0, 8.0, WALL)
    absurd = bays.BaySpec("no", 0.90, 0.99, 20.0, 30.0, 500.0, WALL)
    a = bays.insert_bay(loop, c, 0.0, roomy, WALL)
    b = bays.insert_bay(loop, c, 0.0, absurd, WALL)
    assert len(a) == len(b) == len(loop) + bays.POINTS_PER_BAY


def test_the_depth_clamps_rather_than_refusing():
    """Geometry never self-intersects; gates report insufficiency. A floor
    asked for more depth than the section has is placed as deep as the
    section allows, and `Volume.fits` is what says the box does not fit."""
    _, _, _, plan = _plan("micro")
    s = vase.PrintSettings()
    loop, c = _loop(plan, 0.5, s)
    deep = bays.BaySpec("deep", 0.30, 0.50, 20.0, 30.0, 500.0, WALL)
    got = bays.insert_bay(loop, c, 0.0, deep, WALL) * c
    assert min_clearance_mm(got, skip=8) >= WALL


def test_the_bay_ramps_closed_and_leaves_a_groove():
    """Where the bay has closed the detour does not vanish -- it becomes a
    one-bead groove, which keeps the point count constant without
    coincident vertices and scribes the hatch rim on the part."""
    spec = bays.BaySpec("b", 0.30, 0.55, span_mm=20.0, ramp_mm=25.0,
                        depth_mm=20.0, floor_gap_mm=WALL)
    assert spec.depth_frac(0.0) == 1.0
    assert spec.depth_frac(20.0) == 1.0
    assert spec.depth_frac(32.5) == pytest.approx(0.5)
    assert spec.depth_frac(45.0) == 0.0
    assert spec.depth_frac(999.0) == 0.0


# --------------------------------------------- the truss and the openings

def test_a_rib_is_pushed_to_a_side_that_does_not_move_with_z():
    """Pushing to the NEARER edge is the obvious rule and it is
    discontinuous: as a rib sweeps across an exclusion band's centre,
    'nearer' flips and the rib jumps the band's whole width in one layer.
    Harmless for a 10 mm spar corridor, and 64 degrees of overhang once a
    payload bay became a 72 mm-wide band."""
    spec = RibSpec(n_ribs=3, pitch_mm=30.0, avoid=((0.44, 36.0),))
    chord = 244.0
    seen = [spec.stations(z, chord) for z in np.linspace(0.0, 60.0, 121)]
    jumps = max(float(np.abs(b - a).max()) for a, b in zip(seen, seen[1:]))
    assert jumps < 0.02, f"a rib jumped {jumps:.3f}c between layers"


def test_a_panel_that_is_mostly_openings_takes_no_truss():
    """micro's centre body carries two bays that between them exclude
    0.24c to 0.77c. Squeezing three webs into what is left gave 83 degrees
    of overhang; the honest answer is that the panel cannot carry a
    chordwise truss, and the buckling gate should price that rather than
    the geometry pretending otherwise."""
    wide = RibSpec(n_ribs=3, pitch_mm=30.0,
                   avoid=((0.41, 27.0), (0.67, 15.0)))
    assert ribs_that_fit(wide, 160.0, 0.5) < 3
    clear = RibSpec(n_ribs=3, pitch_mm=30.0)
    assert ribs_that_fit(clear, 160.0, 0.5) == 3
    off = RibSpec(n_ribs=3, pitch_mm=30.0, enabled=False)
    assert ribs_that_fit(off, 160.0, 0.5) == 0


def test_the_ramp_budget_is_measured_on_a_bare_panel():
    """Measuring it on a panel that already has the cut is circular: the
    ramp's own dive and climb walls move in Z, so the budget comes back
    zero and the ramp infinite. The first version reported that a 24 mm
    bay needed an infinite span to close."""
    _, mission, _, plan = _plan("trainer_v3")
    s = vase.PrintSettings(bed_z_mm=250.0, spar_d_mm=8.0)
    bare = vase.build_panels(plan, s)[0]
    rate_bare = vase.ramp_budget(bare, 0.29, 0.59)
    assert rate_bare > 0.0, "a bare panel must have some budget left"
    cut = vase.build_panels(
        plan, s, bays=(("b", 0.29, 0.59, 0.039, 24.0, 30.0),))[0]
    assert vase.ramp_budget(cut, 0.29, 0.59) < rate_bare, (
        "the cut must consume budget -- that is why it cannot measure it")


# ------------------------------------------------------- end to end

def test_every_cut_panel_still_passes_its_clearance_gate():
    """The whole fleet, with the bays actually cut."""
    for name in MISSIONS:
        ev, mission = _built(name)
        for pan in ev.panels:
            chk = vase.check(pan)
            g = {x.name: x for x in chk.gates}
            assert g["min wall separation"].passed, (
                f"{name} {pan.name}: {g['min wall separation'].line()}")


def test_the_openings_are_reserved_against_the_spars():
    """What a spar must avoid is the CUT, not the box: the cut runs from
    the upper skin down to the floor, and a tube seated in the region the
    cut removes is a tube in mid-air. On the trainer this is what moves
    the TE spar off the upper skin, where the seat solver had put it to
    clear the pack."""
    ev, mission = _built("trainer_v3")
    assert not [r for r in ev.reasons if "share" in r], (
        f"the seat solver should have found clear seats: {ev.reasons}")
    for f in ev.spar_fits:
        assert f.clash_mm == 0.0, f"{f.spec.name} clashes {f.clash_mm:.1f} mm"


def test_reach_is_a_constraint_and_clearance_is_the_objective():
    """Reach used to lead, and that was wrong as soon as anything else was
    in the wing: stations clear of the trainer's bay reach slightly less
    far, so they lost on reach before clearance was ever considered, and
    the solver reported an 8 mm clash on a design that had a clear seat.
    Reach only has to satisfy `min_spar_reach_frac`."""
    from washout import spars as sp
    from washout.geom import interior as it

    u, mission, base, plan = _plan("trainer_v3")
    p = unit_to_physical(u)
    s = vase.PrintSettings(bed_z_mm=250.0, spar_d_mm=8.0)
    vols, seats, _ = seat_bays(plan, mission, p, WALL, vase.panel_etas(plan, s))
    cuts = tuple(it.Volume(f"{v.name} opening", v.x0, v.x1, v.eta0, v.eta1,
                           height_mm=v.height_mm + WALL, anchor=it.UPPER)
                 for v in vols)
    spec = mission.spars[-1]
    held = sp.place(plan, spec, WALL, reserved=tuple(vols) + cuts,
                    min_reach=mission.min_spar_reach_frac)
    assert held.clash_mm == 0.0, (
        f"a clear seat exists at {held.x_frac:.2f}c and the solver must "
        f"take it; got {held.clash_mm:.1f} mm of clash")
    assert held.reach_eta >= mission.min_spar_reach_frac
