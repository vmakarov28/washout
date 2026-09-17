"""The control surface is a printed part, and the spar is not inside it.

The realisation this rests on: the hinge line runs spanwise and print Z
IS the span, so an elevon is a vase-mode part in the same orientation as
the wing panel it came off. Before this, `elevon_chord` and `elevon_eta`
were searched -- they set the control authority the score is computed
from -- and the exported shell had its trailing edge still attached. The
aircraft that flew was not the aircraft that was scored.

Three bugs fell out of connecting them, and two were older than the
elevon work. Each is pinned below.
"""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from washout.geom import cst
from washout.printing import elevons as elv
from washout.printing import vase
from washout.printing.ribs import RibSpec, insert_ribs
from washout.search.design import MISSIONS, Mission, build, evaluate, unit_to_physical

ASSETS = Path(__file__).resolve().parent.parent / "assets"
RESULTS = Path(__file__).resolve().parent.parent / "results" / "fleet"


def _fleet(name):
    index = json.loads((RESULTS / "index.json").read_text(encoding="utf-8"))
    d = json.loads((RESULTS / index[name] / "design.json").read_text(encoding="utf-8"))
    base = cst.load_selig(ASSETS / "mh45.dat")
    mission = getattr(Mission, name)()
    u = np.array(d["u"])
    return u, mission, base, build(u, mission, base)


# ----------------------------------------------- the cut lands on a joint

def test_the_print_breaks_at_the_hinge_station():
    """A panel truncated half way up would need a surface normal to the
    span, which is a ROOF, and spiralize has no top layers and cannot
    bridge. So the elevon station has to be a print joint."""
    u, mission, base, plan = _fleet("trainer_v3")
    p = unit_to_physical(u)
    s = vase.PrintSettings(bed_z_mm=250.0, spar_d_mm=8.0,
                           elevon_chord=p["elevon_chord"],
                           elevon_eta=p["elevon_eta"])
    etas = vase.panel_etas(plan, s)
    starts = [a for a, _ in etas]
    assert any(abs(a - p["elevon_eta"]) < 1e-9 for a in starts), (
        f"no panel starts at the hinge station {p['elevon_eta']:.3f}: {starts}")
    # and every panel is wholly one side of it
    for a, b in etas:
        assert a >= p["elevon_eta"] - 1e-9 or b <= p["elevon_eta"] + 1e-9, (
            f"panel {a:.3f}-{b:.3f} straddles the hinge station")


def test_the_exported_shell_carries_the_elevon_the_search_scored():
    """`print_settings` must be the settings the verdict was reached with.

    They were a local variable, so `evaluate` handed `choose_structure` the
    CALLER's settings and the exporter got those back -- elevon chord zero
    and spar corridors empty. The search cut the trailing edge off and kept
    the truss clear of the tubes; the export did neither."""
    for name in MISSIONS:
        u, mission, base, plan = _fleet(name)
        p = unit_to_physical(u)
        settings = vase.PrintSettings(filament_density_gcc=0.55, spar_d_mm=8.0,
                                      bed_z_mm=250.0)
        ev = evaluate(u, mission, base, settings)
        ps = ev.print_settings
        assert ps.elevon_chord == pytest.approx(p["elevon_chord"]), name
        assert ps.elevon_eta == pytest.approx(p["elevon_eta"]), name
        assert len(ps.spar_corridors) == len(mission.spars), name
        assert len(ps.spar_avoid) == len(mission.spars), name


def test_a_truncated_panel_keeps_the_loop_invariants():
    """Constant point count and upper/lower pairing at matching x. The STL
    skinner joins layer k index i to layer k+1 index i, and the wall gate
    pairs index i against its mirror; a truncation that dropped points
    would shear the mesh and blind the gate at once."""
    af = cst.load_selig(ASSETS / "mh45.dat")
    loop = af.coords(121)
    cut = elv.truncate_loop(loop, 0.72)
    assert len(cut) == len(loop)
    n = (len(cut) + 1) // 2
    up, lo = cut[:n][::-1], cut[n - 1:]
    assert np.allclose(up[:, 0], lo[:, 0]), "pairs must share an x"
    assert cut[:, 0].max() == pytest.approx(0.72)
    assert np.all(up[1:, 1] > lo[1:, 1]), "upper must stay above lower"


# ------------------------------------------------------ the hinge geometry

def test_the_bevel_angle_is_the_deflection_angle():
    """Derived, not chosen. A nose-face point a depth d below the axis on a
    face bevelled beta sits at (d tan(beta), -d); deflecting down by delta
    takes its x to d(tan(beta) cos(delta) - sin(delta)), which stays out of
    the wing exactly while beta >= delta."""
    for delta in (8.0, 12.0, 20.0):
        d = 10.0
        beta = delta                       # the claim
        x_after = d * (np.tan(np.radians(beta)) * np.cos(np.radians(delta))
                       - np.sin(np.radians(delta)))
        assert x_after >= -1e-9, f"beta = delta must suffice at {delta} deg"
        short = d * (np.tan(np.radians(delta - 2.0)) * np.cos(np.radians(delta))
                     - np.sin(np.radians(delta)))
        assert short < 0.0, "a shallower bevel must NOT clear"


def test_the_elevon_nose_keeps_a_printable_flat():
    """A bevel meeting the hinge axis exactly is a feather edge, thinner
    than the nozzle, and vase mode would cross its own bead. The first
    version of the bevel did exactly that -- it closed the nose to zero
    thickness and the wall gate read 0.039 mm."""
    af = cst.load_selig(ASSETS / "mh45.dat")
    loop = af.coords(121)
    e = elv.elevon_loop(loop, 0.72, 140.0, 0.8, 16.0, 121)
    assert len(e) == 2 * 121, "two blunt faces means no shared vertex"
    up, lo = e[:121], e[121:][::-1]          # both TE -> nose
    t_nose = float(up[-1, 1] - lo[-1, 1]) * 140.0
    assert t_nose > 0.9, f"nose is {t_nose:.3f} mm thick"
    assert np.all(up[:, 1] - lo[:, 1] > 0.0), "the loop must never close"
    assert np.allclose(up[:, 0], lo[:, 0]), "pairs must share an x"


def test_every_elevon_part_passes_its_own_print_gates():
    """And the gates must be the RIGHT gates. An elevon has no spar, so
    asking whether an 8 mm tube fits its 6 mm section rejects a part that
    was never meant to hold one; and the 300 mm2 adhesion floor is sized
    for a 111 mm tall centre body, not a 25 mm trailing-edge wedge."""
    u, mission, base, plan = _fleet("trainer_v3")
    p = unit_to_physical(u)
    s = vase.PrintSettings(bed_z_mm=250.0, spar_d_mm=8.0,
                           elevon_chord=p["elevon_chord"],
                           elevon_eta=p["elevon_eta"])
    parts = elv.build_elevons(plan, s, vase.panel_etas(plan, s),
                              mission.max_elevon_deflect_deg + s.hinge_margin_deg)
    assert parts, "the trainer must produce elevon parts"
    for part in parts:
        assert part.role == "elevon"
        chk = vase.check(part)
        assert "spar bore" not in [g.name for g in chk.gates]
        assert chk.ok, f"{part.name}: {chk.report()}"


def test_a_spar_may_not_live_inside_the_control_surface():
    """Newly askable: before the hinge line reached the geometry there was
    no line for a spar to be aft of. A tube there has nothing to run
    through and the surface cannot move."""
    u, mission, base, plan = _fleet("trainer_v3")
    settings = vase.PrintSettings(filament_density_gcc=0.55, spar_d_mm=8.0,
                                  bed_z_mm=250.0)
    ev = evaluate(u, mission, base, settings)
    xh = 1.0 - ev.print_settings.elevon_chord
    root_c_mm = ev.plan.stations[0].chord_m * 1000.0
    for f in ev.spar_fits:
        aft = f.x_frac + 0.5 * f.spec.d_mm / root_c_mm
        if f.reach_eta > ev.print_settings.elevon_eta:
            assert aft <= xh + 1e-9, (
                f"{f.spec.name} reaches {aft:.3f}c, hinge at {xh:.3f}c, and "
                f"the design was not rejected for it")


# --------------------------------------- two bugs older than the elevons

def test_the_ribs_stay_out_of_the_spar_corridor():
    """The corridor was specified as a fraction of the ROOT chord and
    applied as a fraction of the LOCAL chord.

    At the root they agree. Outboard the local chord is a third of the
    root's, so the corridor came out three times too narrow and the truss
    ran through the tube. Measured on the gen5 trainer, the largest circle
    that fitted at the LE corridor was 7.65 mm at the root and 2.73 mm at
    the tip -- for an 8 mm spar. Every ribbed panel this project exported
    before this fix has ribs through its spars.

    A centre is a chord fraction because that is how the spar is fitted; a
    half-width is a physical millimetre and does not shrink with the
    chord. They are different kinds and must be carried as such."""
    u, mission, base, plan = _fleet("trainer_v3")
    settings = vase.PrintSettings(filament_density_gcc=0.55, spar_d_mm=8.0,
                                  bed_z_mm=250.0)
    ev = evaluate(u, mission, base, settings)
    ps = ev.print_settings
    assert ps.ribs, "this test is about the ribbed panels"

    for pan in vase.build_panels(ev.plan, ps):
        for f in pan.spar_x_local:
            worst = min(
                vase._inscribed_gap(
                    L, L[:, 0].min() + f * (L[:, 0].max() - L[:, 0].min()))
                for L in pan.contours)
            assert worst >= ps.spar_d_mm, (
                f"{pan.name}: only {worst:.2f} mm of corridor at {f:.3f} of "
                f"the loop, for a {ps.spar_d_mm:.0f} mm tube")


def test_the_bore_is_the_largest_circle_that_fits():
    """A spar is a round tube entering along print Z, so what decides
    whether it goes in is the largest circle the contour holds -- not the
    section's thickness and not the contour's vertical extent.

    On a plain section the three nearly agree, which is the check that the
    inscribed measurement is right. On a ribbed one they do not, and the
    difference is the ribs, which is the point."""
    u, mission, base, plan = _fleet("trainer_v3")
    s = vase.PrintSettings(bed_z_mm=250.0, spar_d_mm=8.0,
                           spar_corridors=((0.21, 1.0),))
    pan = vase.build_panels(plan, s)[0]
    L = pan.contours[0]
    x = L[:, 0].min() + 0.21 * (L[:, 0].max() - L[:, 0].min())
    insc = vase._inscribed_gap(L, x)
    vert = vase._vertical_extent(L, x)
    assert insc == pytest.approx(vert, rel=0.10), (
        f"on a plain section they must agree: {insc:.2f} vs {vert:.2f}")
    assert insc <= vert + 1e-9, "a circle cannot beat the vertical extent"


def test_the_rib_walk_uses_the_loops_own_trailing_edge():
    """`insert_ribs` walked its segments from x = 1.0.

    A panel truncated at the hinge line ends at x_hinge, so every point of
    the first segment was interpolated past the end of the data and
    np.interp clamped them all onto the cut face -- a pile of coincident
    vertices, which the clearance gate reads as a wall touching itself and
    the bore gate read as a section a third of its real depth."""
    af = cst.load_selig(ASSETS / "mh45.dat")
    cut = elv.truncate_loop(af.coords(121), 0.72)
    spec = RibSpec(n_ribs=3, pitch_mm=30.0, x_first=0.20 * 0.72,
                   x_last=0.72 * 0.72, x_clip=(0.08 * 0.72, 0.88 * 0.72))
    ribbed = insert_ribs(cut, 200.0, 0.0, spec, 0.5, 0.5)
    assert ribbed[:, 0].max() <= 0.72 + 1e-9, "no vertex past the cut face"
    # no pile-up: consecutive points must not be coincident
    d = np.linalg.norm(np.diff(ribbed, axis=0), axis=1)
    assert d.max() > 0.0
    assert (d < 1e-9).sum() == 0, f"{(d < 1e-9).sum()} coincident vertices"
