"""The CAD export: surfaces fitted to the print, solids, STEP, 3D gates.

The fitting half is numpy and runs everywhere. The kernel half needs the
OpenCASCADE bindings and skips without them -- the search never imports
them, and the suite must not either.

The first STEP attempt (never committed) lofted the printed contour, rib
slits and all, and came out as 9,635 faces and 30 MB per panel: valid,
and useless in CAD. These tests pin what replaced it.
"""

import json
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from washout.cad import bspline as bs
from washout.geom import cst
from washout.printing import elevons as elv
from washout.printing import stl, vase
from washout.search.design import Mission, build, evaluate, unit_to_physical

ASSETS = Path(__file__).resolve().parent.parent / "assets"
RESULTS = Path(__file__).resolve().parent.parent / "results" / "fleet"


@lru_cache(maxsize=None)
def _parts(name="micro_fpv"):
    index = json.loads((RESULTS / "index.json").read_text(encoding="utf-8"))
    d = json.loads((RESULTS / index[name] / "design.json").read_text(encoding="utf-8"))
    u = np.array(d["u"])
    plan = build(u, getattr(Mission, name)(), cst.load_selig(ASSETS / "mh45.dat"))
    p = unit_to_physical(u)
    s = vase.PrintSettings(elevon_chord=p["elevon_chord"], elevon_eta=p["elevon_eta"])
    spans = vase.panel_etas(plan, s)
    wing = vase.build_panels(plan, s, z_step_mm=1.0)
    elev = elv.build_elevons(plan, s, spans, 16.0, z_step_mm=1.0)
    return plan, s, wing, elev


# ---------------------------------------------------------- the fitting


def test_a_panel_is_a_skin_and_its_trailing_edge():
    """One face per smooth piece: the skin wraps from the trailing edge
    round the nose and back, so the seam is at the trailing edge and the
    leading edge is the smooth inside of a face, where offsets and fillets
    in CAD need it to be. A panel outboard of the hinge is a skin and its
    hinge cut."""
    plan, s, wing, elev = _parts()
    for part in wing:
        truncated = part.frame.eta0 >= s.elevon_eta - 1e-9
        surfs, rep = bs.fit_part(part, every_mm=6.0, truncated=truncated)
        assert [x.name for x in surfs] == ["skin", "hinge cut" if truncated
                                           else "trailing edge"]
        assert surfs[1].u_deg == 1, "the blunt edge is ruled"


def test_the_fitted_surfaces_follow_the_print_between_their_sections():
    """A surface that matches the sections it was fitted to and wanders
    between them would pass one check and fail the other, so both are
    made: on the fitted sections, and on held-out layers half way between.
    The trainer's elevons wandered 3.8 mm between sections before a vertex
    was put on the chamfer's corner; 0.05 mm is the gate."""
    plan, s, wing, elev = _parts()
    for part in wing + elev:
        surfs, rep = bs.fit_part(part, every_mm=3.0)
        assert rep["fit_dev_mm"] <= 0.05, part.name
        assert rep["held_out_dev_mm"] <= 0.05, part.name


def test_adjacent_pieces_share_their_boundary_exactly():
    """Pinned end points make neighbouring pieces' boundary poles the same
    numbers, so their faces share edges exactly and sew without a gap --
    not within a tolerance."""
    plan, s, wing, elev = _parts()
    surfs, _ = bs.fit_part(elev[0], every_mm=6.0)
    for a, b in zip(surfs, surfs[1:] + surfs[:1]):
        assert np.array_equal(a.poles[-1], b.poles[0]), (a.name, b.name)


def test_an_elevon_splits_at_its_chamfer_and_nowhere_else():
    """Five pieces on every layer: upper skin, nose, chamfer, lower skin,
    trailing edge. The corners the nozzle's floor puts near a thin
    section's nose come and go along the part; a piece has to exist on
    every section to be a face, so the split is at the chamfer only."""
    plan, s, wing, elev = _parts()
    for part in elev:
        counts = {len(bs.piece_bounds(c, part.n_upper)) for c in part.contours}
        assert counts == {5}, (part.name, counts)


# ---------------------------------------------------------- the kernel


def test_a_panel_is_four_faces_and_one_valid_solid(tmp_path):
    """Four faces -- skin, trailing edge, root plane, tip plane -- where the
    first attempt had 9,635; one closed valid solid; the volume the layers
    enclose to 0.5%; and a file of hundreds of kilobytes, not 30 MB. The
    STEP file is read back and checked too, names included."""
    pytest.importorskip("OCP")
    from washout.cad import brep
    from OCP.STEPControl import STEPControl_Reader
    plan, s, wing, elev = _parts()
    part = wing[0]
    surfs, _ = bs.fit_part(part, every_mm=6.0)
    solid, info = brep.part_solid(surfs)
    assert brep.is_valid(solid)
    assert [n for n, _ in info["faces"]].count("skin") == 1
    assert len(info["faces"]) == 4
    V, T = stl.skin(part)
    assert brep.volume_mm3(solid) == pytest.approx(stl.volume_mm3(V, T), rel=0.005)
    doc = brep.StepDocument(part.name)
    doc.add(part.name, solid, "vase", info["faces"])
    path = tmp_path / "p0.step"
    assert doc.write(path)
    assert path.stat().st_size < 2_000_000
    text = path.read_text(encoding="latin-1")
    assert "PLANE" in text and "B_SPLINE_SURFACE_WITH_KNOTS" in text
    for face in ("skin", "trailing edge", "root face", "tip face"):
        assert f"'{face}'" in text
    r = STEPControl_Reader()
    r.ReadFile(str(path))
    r.TransferRoots()
    assert brep.is_valid(r.OneShape())


def test_the_export_puts_the_tubes_inside_the_wing(tmp_path):
    """The gate a section-at-a-time pipeline could not ask: the carbon
    cylinder, in 3D, clear of the printed skin and inside the wing along
    its whole axis. Before the spars were straight, the build sheet's
    tip-to-tip tube left the skin a quarter of the way out; this is the
    check that would have said so the first time anyone ran it."""
    pytest.importorskip("OCP")
    from washout.cad.export import export_step
    index = json.loads((RESULTS / "index.json").read_text(encoding="utf-8"))
    d = json.loads((RESULTS / index["micro_fpv"] / "design.json").read_text(encoding="utf-8"))
    m = Mission.micro_fpv()
    ev = evaluate(np.array(d["u"]), m, cst.load_selig(ASSETS / "mh45.dat"),
                  vase.PrintSettings(filament_density_gcc=0.55, spar_d_mm=m.spar_d_mm,
                                     bed_z_mm=250.0), z_step_mm=2.0)
    rep = export_step(ev, tmp_path, every_mm=6.0, z_step_mm=1.0)
    gates = {g.name: g for g in rep["_gates"]}
    tube = gates["main spar tube in shell"]
    assert tube.passed, tube.line()
    failed = [g.line() for g in rep["_gates"] if not g.passed]
    assert not failed, "\n".join(failed)
    asm = (tmp_path / "cad" / "micro_fpv_assembly.step").read_text(encoding="latin-1")
    for p in ("micro_fpv_p0 (left)", "micro_fpv_p0 (right)", "main spar 6x4 (right)"):
        assert p in asm
    assert "CYLINDRICAL_SURFACE" in asm
    saved = json.loads((tmp_path / "cad" / "micro_fpv_cad.json").read_text())
    assert saved["gates"] and all("value" in g for g in saved["gates"])


def test_demon1s_elevons_are_valid_solids():
    """gen7's demon1 winner exported an elevon whose root cap was an
    invalid, self-intersecting face: a 45-degree corner where the nose
    floor meets the chamfer sat inside the chamfer's piece, and the cubic
    fitted across it folded back over the nose flat. Split at that corner,
    with a vertex put on it, every elevon is one valid solid and follows
    the print to the gate. (The design itself is infeasible -- its servo
    shaft sits inboard of its elevon -- and is kept for its geometry.)"""
    pytest.importorskip("OCP")
    from washout.cad import brep
    d = json.loads((RESULTS / "gen7_demon1_v120" / "design.json").read_text(encoding="utf-8"))
    u = np.array(d["u"])
    plan = build(u, Mission.demon1(), cst.load_selig(ASSETS / "mh45.dat"))
    p = unit_to_physical(u)
    s = vase.PrintSettings(elevon_chord=p["elevon_chord"], elevon_eta=p["elevon_eta"])
    spans = vase.panel_etas(plan, s)
    for part in elv.build_elevons(plan, s, spans, 16.0, z_step_mm=1.0):
        surfs, rep = bs.fit_part(part, every_mm=3.0)
        assert rep["held_out_dev_mm"] <= 0.05, (part.name, rep)
        solid, _ = brep.part_solid(surfs)
        assert brep.is_valid(solid), (part.name, [x.name for x in surfs])
