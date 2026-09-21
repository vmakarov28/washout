"""The rib slits, and the invariants every detour of the skin loop owes.

This file used to be `test_bays.py`, and most of it tested the payload
bays: recesses cut into the skin so a battery could be loaded through an
opening the program made. Those are gone -- an opening in vase mode is
sealed from the cavity, so nothing could be routed between two of them,
and the wing is loaded through the open ends of its panels before they
are bonded. Nine tests went with the feature they described: the
floor and its closed section, the depth clamp, the ramp budget and
where it bites, the groove fade, the pocket width, the exclusion
band, the kept vertices, and the cut panel's clearance.

What is kept is everything whose subject outlived the bays: the rib
truss and its layout, the general clearance path, the constant point
count, the spar reservation, and watertightness. The bugs those record
were found while the bays existed but none of them was ABOUT a bay --
they are about what happens when a single closed loop is made to carry
a feature, which is still exactly what a rib slit does.
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from washout.geom import cst
from washout.printing import vase
from washout.printing.ribs import RibSpec, min_clearance_mm, ribs_that_fit
from washout.search.design import (MISSIONS, Mission, build, evaluate,
                                   seat_bays, unit_to_physical)

ASSETS = Path(__file__).resolve().parent.parent / "assets"
RESULTS = Path(__file__).resolve().parent.parent / "results" / "fleet"
WALL = 0.45


@lru_cache(maxsize=None)
def _plan(name):
    index = json.loads((RESULTS / "index.json").read_text(encoding="utf-8"))
    d = json.loads((RESULTS / index[name] / "design.json").read_text(encoding="utf-8"))
    base = cst.load_selig(ASSETS / "mh45.dat")
    mission = getattr(Mission, name)()
    u = np.array(d["u"])
    return u, mission, base, build(u, mission, base)


# `evaluate` is deterministic -- `test_the_export_is_deterministic` pins
# that -- and these tests only read the result, so one evaluation per
# mission serves the whole module.
@lru_cache(maxsize=None)
def _built(name):
    u, mission, base, plan = _plan(name)
    s = vase.PrintSettings(filament_density_gcc=0.55, spar_d_mm=8.0,
                           bed_z_mm=250.0,
                           max_overhang_deg=mission.max_overhang_deg or 50.0)
    return evaluate(u, mission, base, s, want_panels=True), mission


# --------------------------------------------------------- the rib truss


def test_a_rib_is_pushed_to_a_side_that_does_not_move_with_z():
    """Pushing to the NEARER edge is the obvious rule and it is
    discontinuous: as a rib sweeps across an exclusion band's centre,
    "nearer" flips and the rib jumps the band's whole width in one layer.
    Harmless for a 10 mm spar corridor, and 64 degrees of overhang once a
    band got wide. The bands are spar corridors now, but the
    discontinuity is a property of the rule, not of what made the band."""
    spec = RibSpec(n_ribs=3, pitch_mm=30.0, avoid=((0.44, 36.0),))
    chord = 244.0
    seen = [spec.stations(z, chord) for z in np.linspace(0.0, 60.0, 121)]
    jumps = max(float(np.abs(b - a).max()) for a, b in zip(seen, seen[1:]))
    assert jumps < 0.02, f"a rib jumped {jumps:.3f}c between layers"


def test_a_panel_that_is_mostly_excluded_takes_no_truss():
    """Squeezing three webs into what two wide exclusion bands leave gave
    83 degrees of overhang. The honest answer is that the panel cannot
    carry a chordwise truss, and the buckling gate should price that
    rather than the geometry pretending otherwise."""
    wide = RibSpec(n_ribs=3, pitch_mm=30.0,
                   avoid=((0.41, 27.0), (0.67, 15.0)))
    assert ribs_that_fit(wide, 160.0, 0.5) < 3
    clear = RibSpec(n_ribs=3, pitch_mm=30.0)
    assert ribs_that_fit(clear, 160.0, 0.5) == 3
    off = RibSpec(n_ribs=3, pitch_mm=30.0, enabled=False)
    assert ribs_that_fit(off, 160.0, 0.5) == 0


def test_a_rib_never_leaves_its_own_gap_or_reaches_its_neighbour():
    """The sweep amplitude is bounded by the room a rib ACTUALLY has.

    Two failures, both found on demon1's truncated tip panel. Clamping to
    0.30 of a gap's width while the inset is 0.25 of it let a rib swing
    0.05 of a gap past the edge, into the exclusion band. And two ribs in
    DIFFERENT gaps converge toward each other through the band between
    them, which neither one's own gap can see -- two legs came within
    0.34 mm against a 0.45 mm limit.

    So each amplitude is bounded by half the distance to its neighbour's
    base, less the slit and a bead, and by its own distance to each gap
    edge. Checked over a full sweep cycle rather than at one phase."""
    from washout.printing.ribs import free_bands

    spec = RibSpec(n_ribs=4, pitch_mm=20.0,
                   avoid=((0.42, 18.0), (0.62, 10.0))).with_layout(180.0)
    assert spec.layout, "this case must place some ribs"
    gaps = list(free_bands(spec, 180.0))
    for z in np.linspace(0.0, 40.0, 81):
        xs = spec.stations(float(z), 180.0)
        for x in xs:
            assert any(g0 - 1e-9 <= x <= g1 + 1e-9 for g0, g1 in gaps), (
                f"rib at {x:.4f}c left its gap at z={z:.1f}: {gaps}")
        if len(xs) > 1:
            gap_mm = np.diff(np.sort(xs)).min() * 180.0
            assert gap_mm >= 2.0 * spec.slit_allowance_mm - 1e-6, (
                f"two ribs {gap_mm:.3f} mm apart at z={z:.1f}")


def test_the_rib_layout_does_not_change_between_layers():
    """Only the sweep phase varies with Z. The gaps come from the local
    chord and the share between them is an integer, so a layout
    recomputed per layer made ribs hop between gaps from one layer to the
    next -- the same quantisation failure `insert_ribs` documents, for the
    fourth time in this codebase."""
    spec = RibSpec(n_ribs=3, pitch_mm=25.0,
                   avoid=((0.45, 20.0),)).with_layout(200.0)
    bases = [b for b, _ in spec.layout]
    for z in np.linspace(0.0, 50.0, 51):
        xs = spec.stations(float(z), 200.0)
        assert len(xs) == len(bases), f"rib count changed at z={z}"
    # and a layout solved for one chord is not silently resolved for another
    assert spec.stations(0.0, 200.0).size == spec.stations(0.0, 150.0).size


# ----------------------------------------- what every detour still owes


def test_the_point_count_is_constant_through_a_panel():
    """The invariant has to be STRUCTURAL, not conditional.

    Returning a loop unchanged where a feature could not be placed left
    that layer short of every other layer in the panel, and the contour
    array would not broadcast. The rib count is solved once per panel and
    the point budget follows it, so a slit that sweeps out of its gap
    still spends its vertices."""
    for name in MISSIONS:
        ev, _ = _built(name)
        for pan in list(ev.panels) + list(ev.elevons):
            n = {len(c) for c in pan.contours}
            assert len(n) == 1, f"{name} {pan.name}: point count varies {n}"


def test_a_ribbed_panel_measures_clearance_generally():
    """The plain-section clearance test pairs upper[i] with lower[i] by
    index. A slit inserts vertices into ONE skin, so from the slit aft
    every pairing is off and the gate would measure a vertex against the
    wrong mirror point. Ribbed, the gate takes the general
    vertex-to-segment path instead."""
    _, _, _, plan = _plan("trainer_v3")
    s = vase.PrintSettings(bed_z_mm=250.0, spar_d_mm=8.0, ribs=True)
    pan = next(p for p in vase.build_panels(plan, s) if p.has_ribs)
    chk = vase.check(pan)
    g = next(x for x in chk.gates if x.name == "min wall separation")
    step = max(len(pan.contours) // 40, 1)
    general = min(min_clearance_mm(c, skip=8) for c in pan.contours[::step])
    assert abs(g.value - general) < 1e-9, (
        f"gate measured {g.value:.3f} mm, the general test says {general:.3f}")
    assert "ribs" in g.detail


def test_every_panel_still_passes_its_clearance_gate():
    """The whole fleet."""
    for name in MISSIONS:
        ev, mission = _built(name)
        for pan in ev.panels:
            chk = vase.check(pan)
            g = {x.name: x for x in chk.gates}
            assert g["min wall separation"].passed, (
                f"{name} {pan.name}: {g['min wall separation'].line()}")


def test_every_panel_skins_to_a_watertight_mesh():
    """A slicer handed a mesh with a hole makes a part with a hole.

    `vase.check` never tested it, and could not have: `min wall
    separation` skips vertices within 8 indices of each other, exactly so
    that a contour is not judged against its own next point, so a
    COINCIDENT PAIR is invisible to it. Only `stl.export` noticed, at the
    end of the pipeline, and what it reported was six non-manifold edges
    on the trainer's centre body and on demon1's -- a zero-length edge
    stalling the ear-clipper into an invalid fan.

    The invariant is a minimum spacing, held in x, and it is a gate."""
    from washout.printing import stl

    for name in MISSIONS:
        ev, mission = _built(name)
        for pan in ev.panels:
            for k in range(len(pan.contours)):
                loop = pan.contours[k]
                seg = np.linalg.norm(np.roll(loop, -1, axis=0) - loop, axis=1)
                assert seg.min() > 1e-4, (
                    f"{name} {pan.name} layer {k}: two vertices "
                    f"{seg.min():.2e} mm apart")
            verts, tris = stl.skin(pan)
            rep = stl.manifold_report(tris)
            assert rep["watertight"], f"{name} {pan.name}: {rep}"
            g = {x.name: x for x in vase.check(pan).gates}
            assert g["vertex spacing"].passed, g["vertex spacing"].line()


# ------------------------------------------------- the spars and the box


def test_the_payload_boxes_are_reserved_against_the_spars():
    """A tube seated where the pack sits is a tube through the pack. The
    reservation used to be the CUT, which reached deeper than the box;
    with no cut the box is the whole reservation."""
    ev, mission = _built("trainer_v3")
    assert not [r for r in ev.reasons if "share" in r], (
        f"the seat solver should have found clear seats: {ev.reasons}")
    for f in ev.spar_fits:
        assert f.clash_mm == 0.0, f"{f.spec.name} clashes {f.clash_mm:.1f} mm"


def test_reach_is_a_constraint_and_clearance_is_the_objective():
    """Reach used to lead, and that was wrong as soon as anything else was
    in the wing: stations clear of the trainer's pack reach slightly less
    far, so they lost on reach before clearance was ever considered, and
    the solver reported an 8 mm clash on a design that had a clear seat.
    Reach only has to satisfy `min_spar_reach_frac`."""
    from washout import spars as sp

    u, mission, base, plan = _plan("trainer_v3")
    p = unit_to_physical(u)
    s = vase.PrintSettings(bed_z_mm=250.0, spar_d_mm=8.0)
    vols, seats, _ = seat_bays(plan, mission, p, WALL, vase.panel_etas(plan, s))
    spec = mission.spars[-1]
    held = sp.place(plan, spec, WALL, reserved=tuple(vols),
                    min_reach=mission.min_spar_reach_frac)
    assert held.clash_mm == 0.0, (
        f"a clear seat exists at {held.x_frac:.2f}c and the solver must "
        f"take it; got {held.clash_mm:.1f} mm of clash")
    assert held.reach_eta >= mission.min_spar_reach_frac
