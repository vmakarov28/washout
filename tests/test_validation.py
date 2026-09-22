"""The gates. Every one of these caught a real bug during the build.

Run with `python -m pytest -q`. These are not unit tests of internals --
they are the physics and geometry claims the pipeline makes, written down
so a refactor cannot quietly break them. Where a number came from
published data or closed-form theory, the source is named in the test.
"""

from __future__ import annotations

import numpy as np
import pytest

from washout.geom import cst, planform
from washout.printing import stl, vase
from washout.aero.vlm import VLM

ASSETS = __import__("pathlib").Path(__file__).resolve().parents[1] / "assets"


def demo_bwb(af, half_span=0.45, root=0.28, body_t=1.8, name="t"):
    """A representative 4-station BWB for the printing tests."""
    return planform.bwb(
        half_span_m=half_span, root_chord_m=root, body_eta=0.16,
        body_chord_frac=0.88, kink_eta=0.36, kink_chord_frac=0.58,
        tip_chord_frac=0.36, sweep_body_deg=36.0, sweep_mid_deg=26.0,
        sweep_outer_deg=16.0, dihedral_deg=4.0, twist_body_deg=0.0,
        twist_kink_deg=-1.0, twist_tip_deg=-5.0, root_airfoil=af,
        tip_airfoil=af, body_thickness_scale=body_t, name=name)


def rect(ar: float, af: cst.Airfoil) -> planform.Planform:
    return planform.Planform(
        0.5 * ar, tuple(planform.Station(e, 1.0, 0.0, 0.0, 0.0, af)
                        for e in (0.0, 1.0)), "rect")


def cm0_of(af: cst.Airfoil, ar: float = 200.0) -> float:
    """Section zero-lift pitching moment about the quarter chord, via a
    near-2D wing (AR 200) so the induced field is negligible."""
    v = VLM(rect(ar, af), ns=16, nc=12)
    r0, r1 = v.solve(0.0, 0.25), v.solve(2.0, 0.25)
    a_zl = -r0.CL * 2.0 / (r1.CL - r0.CL)
    return float(v.solve(a_zl, 0.25).Cm)


# --------------------------------------------------------------- geometry


def test_naca_fit_reproduces_thickness_and_camber():
    assert cst.naca4("0012").t_max == pytest.approx(0.12, abs=0.002)
    assert cst.naca4("4412").camber_max == pytest.approx(0.04, abs=0.002)


def test_mh45_round_trips():
    af = cst.load_selig(ASSETS / "mh45.dat")
    # the file's own name says 9.85%
    assert af.t_max == pytest.approx(0.0985, abs=0.001)
    assert af.is_valid()


def test_cst_carries_trailing_edge_camber():
    """The bug that made reflex unrepresentable: C(x) = x^0.5 (1-x) is
    ZERO at x=1, so without an explicit x*z_te term the camber line is
    pinned to the chord line at the trailing edge and every reflexed
    section fits with a violent downward hook."""
    x = cst.cosine_x(200)
    t = 0.10 * (1 - x) * np.sqrt(np.maximum(x, 0)) * 4
    cam = np.where(x < 0.7, 0.0, 0.05 * (x - 0.7) / 0.3)
    loop = np.concatenate([np.stack([x, cam + 0.5 * t], 1)[::-1],
                           np.stack([x, cam - 0.5 * t], 1)[1:]])
    af = cst.fit(loop, order=10)
    assert af.camber(np.array([1.0]))[0] == pytest.approx(0.05, abs=0.003)


def test_thickness_scaling_leaves_camber_alone():
    af = cst.load_selig(ASSETS / "mh45.dat")
    fat = af.scaled_thickness(1.7)
    assert fat.t_max == pytest.approx(1.7 * af.t_max, rel=0.02)
    assert fat.camber_max == pytest.approx(af.camber_max, rel=0.02)


# ---------------------------------------------------------------- the VLM


def test_symmetric_section_makes_no_lift_or_moment():
    v = VLM(rect(6.0, cst.naca4("0012")), ns=20, nc=6)
    r = v.solve(0.0, 0.25)
    assert abs(r.CL) < 1e-9
    assert abs(r.Cm) < 1e-9


def test_aerodynamic_centre_is_at_quarter_chord():
    """A straight untapered wing's AC sits at 0.25c. This is the sharpest
    single check on the chordwise discretisation."""
    v = VLM(rect(20.0, cst.naca4("0012")), ns=40, nc=6)
    r0, r1 = v.solve(0.0, 0.0), v.solve(4.0, 0.0)
    x_ac = -(r1.Cm - r0.Cm) / (r1.CL - r0.CL)
    assert x_ac == pytest.approx(0.25, abs=0.01)


def test_lift_slope_approaches_two_pi_in_the_2d_limit():
    v = VLM(rect(1000.0, cst.naca4("0012")), ns=40, nc=4)
    slope = (v.solve(4.0, 0.0).CL - v.solve(0.0, 0.0).CL) / np.radians(4)
    assert slope == pytest.approx(2 * np.pi, rel=0.01)


def test_elliptic_wing_has_span_efficiency_near_one():
    """Trefftz-plane induced drag: e -> 1 for elliptic loading, and must
    come out BELOW that for a rectangular wing."""
    e = np.linspace(0, 1, 41)
    ch = np.sqrt(np.maximum(1 - e**2, 1e-9))
    af = cst.naca4("0012")
    ell = planform.Planform(
        np.pi * 8 / 8, tuple(planform.Station(float(v), float(c),
                                              float(0.25 * (1 - c)), 0.0, 0.0, af)
                             for v, c in zip(e, ch)), "ell")
    e_ell = VLM(ell, ns=50, nc=4).solve(5.0, 0.0).e_oswald
    e_rect = VLM(rect(ell.aspect_ratio, af), ns=50, nc=4).solve(5.0, 0.0).e_oswald
    assert e_ell == pytest.approx(1.0, abs=0.02)
    assert e_rect < e_ell


def test_naca4412_pitching_moment_matches_published_data():
    """Published Cm_c/4 for NACA 4412 is about -0.09 to -0.10. An
    inviscid lattice runs slightly heavy, so the band is generous -- but
    the SIGN and the order of magnitude are not negotiable."""
    assert cm0_of(cst.naca4("4412")) == pytest.approx(-0.10, abs=0.025)


def test_camber_scales_the_moment_linearly():
    c2 = cm0_of(cst.naca4("2412"))
    c4 = cm0_of(cst.naca4("4412"))
    c6 = cm0_of(cst.naca4("6412"))
    assert c4 / c2 == pytest.approx(2.0, rel=0.05)
    assert c6 / c2 == pytest.approx(3.0, rel=0.05)


def test_reflex_raises_cm0_monotonically_and_with_the_right_sign():
    """The whole tailless design problem in one assertion: trailing edge
    UP must give a NOSE-UP zero-lift moment."""
    af = cst.load_selig(ASSETS / "mh45.dat")
    vals = [cm0_of(cst.deflect_te(af, d)) for d in (0.0, 2.0, 4.0, 6.0)]
    assert all(b > a for a, b in zip(vals, vals[1:]))
    assert vals[-1] > 0.05


# ------------------------------------------------------------- printing


def _panel():
    af = cst.load_selig(ASSETS / "mh45.dat")
    p = demo_bwb(af, half_span=0.5, root=0.30, body_t=1.5)
    return p, vase.build_stack(p, vase.PrintSettings(), 0.4, 1.0, "t")


def test_every_layer_clears_the_nozzle():
    """Guarantees one simple closed loop per layer, which is what
    spiralize mode requires and what the STL skinner assumes."""
    _, st = _panel()
    assert st.wall_separation_mm().min() >= st.settings.min_wall_mm


def test_mesh_is_watertight_and_consistently_wound():
    _, st = _panel()
    verts, tris = stl.skin(st)
    rep = stl.manifold_report(tris)
    assert rep["non_manifold_edges"] == 0
    assert rep["inconsistent_windings"] == 0
    assert rep["watertight"]


def test_mesh_volume_matches_the_planform_integral():
    """An independent check on the whole loft: the skinned solid's
    divergence-theorem volume must agree with the planform's own
    section-area integral."""
    plan, _ = _panel()
    panels = vase.build_panels(plan, vase.PrintSettings())
    mesh_cm3 = sum(stl.volume_mm3(*stl.skin(p)) for p in panels) / 1000.0
    assert mesh_cm3 == pytest.approx(plan.volume_m3() * 1e6 / 2, rel=0.03)


def test_coarse_sampling_preserves_mass_and_overhang():
    """The search samples layers coarsely for speed. Mass is weighted by
    layers-per-sample and overhang by the ACTUAL z rise, so both must be
    invariant -- getting the second wrong reported 82 degrees of overhang
    on a wing that has 42."""
    plan, _ = _panel()
    fine = vase.build_panels(plan, vase.PrintSettings())
    coarse = vase.build_panels(plan, vase.PrintSettings(), z_step_mm=2.0)
    assert (sum(p.mass_g() for p in coarse)
            == pytest.approx(sum(p.mass_g() for p in fine), rel=0.03))
    a_f = max(vase.overhang_deg(p)[0] for p in fine)
    a_c = max(vase.overhang_deg(p)[0] for p in coarse)
    assert a_c == pytest.approx(a_f, abs=6.0)


def test_panels_fit_the_z_envelope():
    plan, _ = _panel()
    s = vase.PrintSettings()
    for p in vase.build_panels(plan, s):
        assert p.height_mm <= s.bed_z_mm


# ---------------------------------------------------------- the search loop


def test_trim_is_converged_at_the_working_lattice_resolution():
    """The search and the final verdict must agree, so the resolution they
    share has to be converged in the quantity that decides feasibility.

    Trim converges much more slowly than lift: it is where a small MOMENT
    difference is driven to zero. On a representative design cruise speed
    reads 13.5 m/s at 12x4 and 10.6 at 32x8 -- a 27% error. Searching at
    one resolution and judging at another wasted a 7384-evaluation run.

    32x8 is NOT fully converged: it still sits ~4% in cruise speed from
    42x10, and this test pins that residual rather than hiding it. That
    is a deliberate trade -- 4% of cruise speed is 0.4 m/s, well inside
    the tier-0 profile-drag error of 20-40%, and the resolution costs
    ~1.7 s per evaluation against ~2.4 s. What was actually fatal was not
    the residual but the MISMATCH, and that is now impossible by
    construction: LATTICE_NS/NC are used by the search and the verdict
    alike.
    """
    from washout.geom import cst as _cst
    from washout.search.design import (LATTICE_NC, LATTICE_NS, Mission,
                                          evaluate, physical_to_unit)
    from washout.search.optimize import SEED_PHYSICAL

    base = _cst.load_selig(ASSETS / "mh45.dat")
    m = Mission.trainer_v3()
    s = vase.PrintSettings(filament_density_gcc=0.55)
    # An explicit design, not whatever seed happens to be recorded: this
    # test is about the LATTICE, so its geometry must not move when the
    # optimizer's starting point does.
    # Every bound gets its midpoint, then the handful this test actually
    # cares about are overridden. Naming all of them explicitly broke
    # this test three separate times as the design vector grew -- once
    # for the airfoil/elevon/prop variables, once for the fifth station
    # and polyhedral. The test is about the LATTICE; it should not have
    # an opinion about how many design variables exist.
    from washout.search.design import BOUNDS as _B
    phys = {b.name: 0.5 * (b.lo + b.hi) for b in _B}
    phys.update(
        span_m=0.90, root_chord=0.30, body_eta=0.18, body_chord_frac=0.90,
        kink_gap=0.35, kink_taper=0.67, outer_taper=0.72, tip_taper=0.80,
        sweep_body=42.0, sweep_mid_delta=-12.0, sweep_outer_delta=-6.0,
        sweep_tip_delta=0.0, dihedral=5.0, winglet_cant=5.0,
        twist_root=0.0, washout=4.0, washout_exp=1.0,
        body_thickness=1.80, batt_x=0.20, reflex_deg=3.5, camber_scale=1.10)
    # Guard against the silent failure above: an override that names a
    # variable which no longer exists does nothing at all.
    assert set(phys) == {b.name for b in _B}, (
        f"unknown design variables: {set(phys) - {b.name for b in _B}}")
    u = physical_to_unit(phys)
    here = evaluate(u, m, base, s, z_step_mm=2.0)
    finer = evaluate(u, m, base, s, ns=LATTICE_NS + 10, nc=LATTICE_NC + 2,
                     z_step_mm=2.0)
    assert here.v_cruise == pytest.approx(finer.v_cruise, rel=0.05)
    assert here.cl_trim == pytest.approx(finer.cl_trim, rel=0.08)


def test_search_and_verdict_use_the_same_lattice():
    """The mismatch itself, pinned. run_search's final evaluation must be
    at the resolution it searched at, not a 'nicer' one."""
    import inspect

    from washout.search import optimize
    src = inspect.getsource(optimize.run_search)
    assert "ns=ns, nc=nc" in src, "final evaluation must reuse the search lattice"
    assert "ns: int = LATTICE_NS" in inspect.getsource(optimize)


def test_payload_bays_are_checked_for_volume_not_just_mass():
    """A search produced a 'feasible' 483 g aircraft whose battery bay was
    17.7 mm deep for a 26 mm pack. Mass without volume is not a payload:
    with only a mass budget the optimizer shrinks the centre body for
    free, because nothing charges it for the space it removes.

    The binding dimension is depth across the pack's own WIDTH and along
    its full LENGTH, not on the centreline -- a blended body tapers fast
    and the centreline is always the most flattering station."""
    from washout.geom import cst as _cst
    from washout.search.design import Bay, bay_fits

    af = _cst.load_selig(ASSETS / "mh45.dat")
    pack = Bay("4S 1500", 0.27, (76.0, 35.0, 26.0))

    thin = demo_bwb(af, half_span=0.5, root=0.20, body_t=1.37, name="thin")
    fat = demo_bwb(af, half_span=0.5, root=0.40, body_t=2.1, name="fat")
    ok_thin, have_thin, _ = bay_fits(thin, pack, 0.45)
    ok_fat, have_fat, _ = bay_fits(fat, pack, 0.45)
    assert not ok_thin, f"a 20 cm-chord body should not swallow a 26 mm pack ({have_thin:.1f})"
    assert ok_fat, f"a 36 cm-chord body at 1.9x thickness should ({have_fat:.1f})"
    assert have_fat > have_thin


def test_bay_is_checked_where_the_mass_actually_sits():
    """A sweeping fit check answers 'does some seat exist', which is the
    wrong question -- the CG that trims the aircraft comes from where the
    pack IS. A search exploited exactly that gap: battery at 0.085c for
    trim, fit measured at 0.27c, and a 76 mm pack left hanging 19 mm off
    the nose of a 227 mm chord."""
    from washout.geom import cst as _cst
    from washout.search.design import Bay, bay_fits

    af = _cst.load_selig(ASSETS / "mh45.dat")
    plan = demo_bwb(af, half_span=0.5, root=0.227, body_t=1.61, name="run4")
    pack = Bay("4S 1500", 0.27, (76.0, 35.0, 26.0), x_var="batt_x")
    ok_good, have_good, _ = bay_fits(plan, pack, 0.45, x_frac=0.27)
    ok_nose, have_nose, _ = bay_fits(plan, pack, 0.45, x_frac=0.085)
    assert ok_good, f"0.27c is the seat that works ({have_good:.1f} mm)"
    assert not ok_nose, "0.085c hangs the pack off the nose and must fail"
    assert have_nose == 0.0


def test_every_mission_seed_actually_flies():
    """optimize.py claims the search starts from something that flies, so
    that had better be true. It was not for three runs: the seed trimmed
    at the old 16x4 lattice and, at the converged 32x8, its pitching
    moment never crossed zero anywhere in the bracket. A seed that cannot
    trim teaches the early generations nothing."""
    from washout.geom import cst as _cst
    from washout.search.design import Mission, evaluate, physical_to_unit
    from washout.search.optimize import SEEDS

    base = _cst.load_selig(ASSETS / "mh45.dat")
    for name, seed in SEEDS.items():
        m = getattr(Mission, name)()
        s = vase.PrintSettings(ribs=True, rib_count=3, spar_d_mm=4.0,
                               filament_density_gcc=0.55)
        ev = evaluate(physical_to_unit(seed), m, base, s, z_step_mm=3.0)
        assert ev.ok, f"{name} seed infeasible: {'; '.join(ev.reasons)}"


# ------------------------------------------------------- ribs & structure


def _ribbed(nr: int):
    af = cst.load_selig(ASSETS / "mh45.dat")
    p = demo_bwb(af)
    s = vase.PrintSettings(ribs=nr > 0, rib_count=nr, spar_d_mm=4.0)
    return p, vase.build_stack(p, s, 0.40, 0.95, "t")


def test_ribs_keep_the_layer_a_single_simple_loop():
    """Spiralize prints ONE contour per layer, so a rib cannot be a
    separate loop joined to the skin -- that is a T-junction and the
    curve stops being simple. A rib is a DETOUR of the skin loop, and
    the proof is that no part of the contour comes within one extrusion
    width of any non-adjacent part."""
    from washout.printing.ribs import min_clearance_mm
    for nr in (2, 3, 4):
        _, st = _ribbed(nr)
        worst = min(min_clearance_mm(c, skip=8) for c in st.contours[::37])
        assert worst >= st.settings.min_wall_mm, f"{nr} ribs: {worst:.3f} mm"


def test_rib_point_count_is_constant_across_layers():
    """The STL skinner joins layer k index i to layer k+1 index i, so a
    layer that gained or lost a vertex because a rib happened to land on
    one would shear the whole mesh. Ribs are inserted into gaps between
    existing vertices, never on top of them."""
    from washout.printing.ribs import POINTS_PER_RIB
    for nr in (2, 3, 4):
        _, st = _ribbed(nr)
        assert st.contours.shape[1] == 241 + POINTS_PER_RIB * nr


def test_ribbed_mesh_is_still_watertight():
    """Ear clipping replaced the upper/lower ladder precisely because a
    rib detour doubles back in x and the ladder assumed it could not."""
    for nr in (0, 3):
        _, st = _ribbed(nr)
        rep = stl.manifold_report(stl.skin(st)[1])
        assert rep["watertight"], f"{nr} ribs: {rep}"
        assert rep["inconsistent_windings"] == 0


def test_ribs_cost_mass_and_the_cost_is_bounded():
    _, plain = _ribbed(0)
    _, three = _ribbed(3)
    ratio = three.mass_g() / plain.mass_g()
    assert 1.05 < ratio < 1.6, f"ribs changed mass by {ratio:.2f}x"


def test_spar_is_selected_to_survive_the_load_case():
    """Structure is CHOSEN, not assumed: the lightest stock tube that
    passes ultimate load and a deflection limit. A heavier aircraft must
    never select a lighter tube."""
    from washout import structure
    from washout.aero.vlm import VLM

    af = cst.load_selig(ASSETS / "mh45.dat")
    plan = demo_bwb(af)
    pt = VLM(plan, 24, 6).solve(6.0, 0.10)
    light = structure.select(plan, pt, 0.35, 0.45, n_limit=3.0)
    heavy = structure.select(plan, pt, 1.40, 0.45, n_limit=3.0)
    assert light.ok
    assert heavy.spar.od_mm >= light.spar.od_mm
    assert heavy.stress_ult_mpa > light.stress_ult_mpa
    assert light.tip_defl_pct >= 0.0


def test_rib_sweep_respects_the_remaining_overhang_budget():
    """A tapered swept panel already spends most of the overhang
    allowance moving its own section sideways -- 37.9 deg of 50 on the
    trainer root panel with no ribs at all. The truss gets the REMAINDER,
    not its own allowance.

    Sizing the rib against the full limit in isolation produced 58 deg of
    real overhang, and because the excess came from the wing rather than
    the rib it was independent of rib pitch, which is what eventually
    identified it after two wrong diagnoses."""
    af = cst.load_selig(ASSETS / "mh45.dat")
    plan = demo_bwb(af)
    for (e0, e1) in ((0.0, 0.30), (0.30, 0.65), (0.65, 1.0)):
        s = vase.PrintSettings(ribs=True, rib_count=3, rib_pitch_mm=25.0,
                               spar_d_mm=4.0)
        st = vase.build_stack(plan, s, e0, e1, "p")
        ang, _ = vase.overhang_deg(st)
        assert ang <= s.max_overhang_deg, (
            f"panel {e0}-{e1}: {ang:.1f} deg of overhang with ribs")


def test_search_and_verdict_sample_the_print_identically():
    """The search must be scored against the check that decides.

    micro searched 6000 designs at 1.0 mm z sampling, found thousands
    'feasible', and exported none: run_search re-evaluated the winner
    WITHOUT passing z_step_mm, so the verdict re-checked overhang at the
    0.25 mm layer height and saw slopes the coarse pass had smoothed
    over. The optimizer was not wrong -- it was answering a question
    nobody was going to ask again.

    This pins the interface, not the number: whatever z step the search
    uses, the final evaluation must use the same one.
    """
    import inspect
    from washout.search import optimize

    src = inspect.getsource(optimize.run_search)
    verdict = src[src.index("best = evaluate("):]
    verdict = verdict[:verdict.index(")") + 1]
    assert "z_step_mm=search_z_step_mm" in verdict, (
        "the final evaluate() must inherit the search's z sampling; "
        f"got:\n{verdict}")

    # and the default must be the real layer height, not a coarse proxy
    sig = inspect.signature(optimize.run_search)
    assert sig.parameters["search_z_step_mm"].default is None, (
        "default z step must be None (= use the true layer height), so "
        "the search cannot pass a print check the slicer would fail")


def test_the_section_actually_changes_along_the_span():
    """A blended wing body must be able to blend.

    design.py used to pass ONE airfoil object as both root_airfoil and
    tip_airfoil, which made bwb()'s internal blend() a lerp between a
    shape and itself. Measured on a finished design, camber_max was
    identical at all four stations to five decimal places: the aircraft
    had one aerodynamic section, stretched in thickness. Nothing failed,
    no gate fired -- the optimizer just returned the best member of a
    family that could not express the answer.
    """
    from washout.geom import cst as _cst
    from washout.search.design import BOUNDS as _B, Mission, build, physical_to_unit

    base = _cst.load_selig(ASSETS / "mh45.dat")
    phys = {b.name: 0.5 * (b.lo + b.hi) for b in _B}
    phys.update(camber_scale=1.40, tip_camber_scale=0.40,
                reflex_deg=6.0, tip_reflex_deg=0.0, blend_exp=1.0)
    plan = build(physical_to_unit(phys), Mission.trainer_v3(), base)

    cam = [plan.at(e).airfoil.camber_max for e in plan.controls]
    assert cam[0] > cam[-1], f"camber must fall outboard, got {cam}"
    assert cam[0] - cam[-1] > 1e-3, (
        f"root and tip camber differ by only {cam[0]-cam[-1]:.2e} -- the "
        f"blend is a no-op again: {cam}")
    # and it must be monotone, not oscillating through the loft
    assert all(a >= b - 1e-9 for a, b in zip(cam, cam[1:])), cam


def test_polyhedral_puts_the_dihedral_where_it_was_asked_for():
    """Per-segment dihedral, not one angle for the whole span."""
    from washout.geom import cst as _cst
    from washout.geom.planform import Segment, lofted

    af = _cst.load_selig(ASSETS / "mh45.dat")
    half = 0.5
    p = lofted(half, 0.2, 0.0, af, af, (
        Segment(0.5, 1.0, 0.0, 0.0, 0.0),      # flat inboard
        Segment(1.0, 1.0, 0.0, 45.0, 0.0),     # 45 deg outboard
    ))
    z = [s.z_le_m for s in p.stations]
    assert z[1] == pytest.approx(0.0, abs=1e-12), "inboard must stay flat"
    # outboard segment spans half the half-span at 45 deg -> rise == run
    run = half * 0.5
    assert z[2] == pytest.approx(run, rel=1e-9), f"{z}"


def test_a_winglet_reduces_induced_drag():
    """The Trefftz plane has to be two-dimensional to see a winglet.

    Induced drag was computed by projecting the whole wake onto the y
    axis, which is exact for a planar wing and blind to everything else.
    A winglet's entire purpose is to move shed vorticity OUT of that
    plane, so the planar version priced one at precisely zero -- the
    optimizer would have paid its mass and wetted area for nothing and
    correctly refused to fit one.
    """
    from washout.geom import cst as _cst
    from washout.geom.planform import Segment, lofted
    from washout.aero.vlm import VLM

    af = _cst.load_selig(ASSETS / "mh45.dat")

    def wing(tip_dihedral):
        return lofted(0.45, 0.20, 1.0, af, af, (
            Segment(0.50, 0.85, 20.0, 2.0, 0.0),
            Segment(0.85, 0.55, 20.0, 2.0, -1.0),
            Segment(1.00, 0.45, 20.0, tip_dihedral, -2.0),
        ))
    out = {}
    for d in (2.0, 60.0):
        p = wing(d)
        pt = VLM(p, ns=32, nc=8).solve(5.0, p.x_mac_le_m + 0.25 * p.mac_m)
        out[d] = (pt.CL, pt.CDi)
    (cl_f, cdi_f), (cl_w, cdi_w) = out[2.0], out[60.0]
    # compare at equal CL: CDi ~ CL^2, so normalise
    eff_flat = cl_f ** 2 / cdi_f
    eff_wing = cl_w ** 2 / cdi_w
    assert eff_wing > eff_flat * 1.02, (
        f"winglet must improve CL^2/CDi by >2%: flat {eff_flat:.2f} "
        f"vs winglet {eff_wing:.2f}")


def test_dihedral_effect_matches_its_own_closed_form():
    """Cl_beta by strip integration, checked against the textbook limit.

    With constant chord and constant dihedral the integral collapses to
    -a*Gamma/4. If this drifts, the integration or the lift slope is
    wrong -- and the 3D slope matters: using the section's 2*pi here put
    Cl_beta about 30% above published values.
    """
    from washout.geom import cst as _cst
    from washout.geom.planform import Segment, lofted
    from washout.aero import lateral as _lat

    af = _cst.load_selig(ASSETS / "mh45.dat")
    gamma_deg = 5.0
    p = lofted(0.5, 0.2, 0.0, af, af, (
        Segment(0.5, 1.0, 0.0, gamma_deg, 0.0),
        Segment(1.0, 1.0, 0.0, gamma_deg, 0.0),
    ))
    # LINEAR loft, deliberately. The production loft is mirrored about
    # the centreline so the two halves join tangent, which rounds the
    # root: local dihedral runs 0 -> ~1.5 Gamma -> Gamma across the first
    # segment. That is the right shape for an aircraft and the wrong one
    # for this test, which checks the INTEGRAL against a closed form that
    # assumes Gamma is constant all the way in.
    from dataclasses import replace as _replace
    p = _replace(p, smooth=False)
    _tot, dihedral_part, _sweep = _lat.cl_beta(p, 0.0)
    a3d = 2.0 * np.pi * p.aspect_ratio / (p.aspect_ratio + 2.0)
    closed = -a3d * np.radians(gamma_deg) / 4.0
    assert dihedral_part == pytest.approx(closed, rel=0.02), (
        f"integrated {dihedral_part:+.5f} vs closed form {closed:+.5f}")


def test_the_structure_uses_the_spar_the_geometry_was_built_around():
    """The bore is printed to a fixed size; the tube has to match it.

    The search keeps rib corridors clear for an 8 mm tube and checks
    every panel joint is deep enough to pass one. structure.select()
    then chose the lightest tube that carried the load, which on micro
    was 4x2 -- a 4 mm tube in an 8 mm hole. Both halves were right on
    their own terms and the pair was wrong.
    """
    from washout.geom import cst as _cst
    from washout import structure
    from washout.aero.vlm import VLM

    af = _cst.load_selig(ASSETS / "mh45.dat")
    plan = demo_bwb(af)
    pt = VLM(plan, 24, 6).solve(4.0, plan.x_mac_le_m + 0.2 * plan.mac_m)
    free = structure.select(plan, pt, 0.30, skin_t_mm=0.45, n_limit=3.0)
    held = structure.select(plan, pt, 0.30, skin_t_mm=0.45, n_limit=3.0,
                            min_od_mm=8.0)
    assert held.spar.od_mm >= 8.0, (
        f"asked for >= 8 mm, got {held.spar.od_mm} mm ({held.spar.name})")
    # the free choice is allowed to be lighter; that is the whole point
    assert free.spar.od_mm <= held.spar.od_mm


def _plan_from_fixture(d):
    from washout.geom.cst import Airfoil
    from washout.geom.planform import Planform, Station
    return Planform(d["half_span_m"], tuple(
        Station(s["eta"], s["chord_m"], s["x_le_m"], s["z_le_m"], s["twist_deg"],
                Airfoil(au=np.array(s["au"]), al=np.array(s["al"]),
                        te_gap=s["te_gap"], te_camber=s["te_camber"]))
        for s in d["stations"]))


def test_the_designs_called_goofy_fail_the_fairness_gate():
    """The three gen2 winners, frozen as built, must stay rejected.

    They passed every gate that existed and were described, correctly,
    as goofy: sweep that waved between segments, trailing edges that
    hooked, twist that zig-zagged, 200 mm winglets on a 900 mm wing.
    This pins the fairness gate to the actual shapes that prompted it,
    rather than to shapes invented to fail it.
    """
    import json
    import pathlib
    from washout.geom import fairness as fz

    path = pathlib.Path(__file__).parent / "fixtures" / "gen2_goofy.json"
    fx = json.loads(path.read_text())
    assert set(fx) == {"trainer_v3_v72", "demon1_v71", "micro_v71"}
    for name, d in fx.items():
        bad = fz.measure(_plan_from_fixture(d)).violations(fz.Limits())
        reasons = " | ".join(r for r, _ in bad)
        assert len(bad) >= 3, f"{name} should fail on several counts: {reasons}"
        assert "leading edge waves" in reasons, f"{name}: {reasons}"


def test_faired_loft_does_not_overshoot_the_sweep_it_was_given():
    """PCHIP on station POSITIONS overshot the SLOPES between them.

    A random draw with monotone sweeps of 21, 19, 16 and 12.5 degrees
    produced a leading edge sweeping 27 degrees just outboard of the
    nose. faired() interpolates the slopes themselves, linearly between
    segment midpoints, so the leading edge stays within about a degree
    of the steepest segment -- the residue is the Planform
    re-interpolating the dense stations it emits.
    """
    from washout.geom import cst as _cst
    from washout.geom.planform import Segment, faired

    af = _cst.load_selig(ASSETS / "mh45.dat")
    sweeps = (21.0, 19.0, 16.0, 12.5)
    p = faired(0.45, 0.24, 1.0, af, af, (
        Segment(0.14, 0.86, sweeps[0], 3.0, 0.5),
        Segment(0.41, 0.60, sweeps[1], 3.0, -1.0),
        Segment(0.75, 0.47, sweeps[2], 8.0, -2.0),
        Segment(1.00, 0.39, sweeps[3], 18.0, -3.0)))
    eta = np.linspace(0.0, 1.0, 401)
    x = np.array([p.at(float(e)).x_le_m for e in eta])
    lam = np.degrees(np.arctan(np.gradient(x, eta * p.half_span_m)))
    assert lam.max() <= max(sweeps) + 1.5, f"peak LE sweep {lam.max():.2f} deg"


def test_the_halves_join_tangent_on_the_centreline():
    """No V at the nose, no notch in the trailing edge.

    PCHIP fitted to the right half alone estimated its slope at eta = 0
    from one side, so the mirrored halves met at an angle: demon1's nose
    spike and micro's trailing-edge notch. Both edges must now leave the
    centreline square to the span.
    """
    from washout.geom import cst as _cst
    from washout.search.design import Mission, N_DIM, build

    base = _cst.load_selig(ASSETS / "mh45.dat")
    rng = np.random.default_rng(3)
    for _ in range(20):
        p = build(rng.random(N_DIM), Mission.demon1(), base)
        d = 0.001
        a, b = p.at(0.0), p.at(d)
        dy = d * p.half_span_m
        le = np.degrees(np.arctan((b.x_le_m - a.x_le_m) / dy))
        te = np.degrees(np.arctan(((b.x_le_m + b.chord_m)
                                   - (a.x_le_m + a.chord_m)) / dy))
        assert abs(le) < 2.0 and abs(te) < 2.0, (
            f"edges leave the root at LE {le:.2f}, TE {te:.2f} deg")


def test_most_random_designs_are_one_fair_shape():
    """The generator's space must be mostly aeroplanes.

    With every station independent, 0 of 300 random draws passed the
    fairness gate, so the optimizer would have spent its budget hunting
    a corner of the space. With chord monotone, sweep one-humped and
    twist monotone by construction, and the loft run through the slopes,
    over half pass. The floor sits well below the measured rate so it
    catches a regression, not noise.
    """
    from washout.geom import cst as _cst
    from washout.geom import fairness as fz
    from washout.search.design import Mission, N_DIM, build

    base = _cst.load_selig(ASSETS / "mh45.dat")
    m = Mission.trainer_v3()
    rng = np.random.default_rng(11)
    fair = total = 0
    for _ in range(120):
        p = build(rng.random(N_DIM), m, base)
        if not p.is_valid()[0]:
            continue
        total += 1
        fair += not fz.measure(p).violations(m.fairness)
    assert total >= 100
    assert fair / total >= 0.35, f"only {fair}/{total} random draws are fair"


def test_the_print_is_split_only_as_far_as_the_envelope_demands():
    """A joint is not free, so there must not be one that is not needed.

    This used to assert that EVERY control station is a panel break. That
    was stronger than the reason for it: the faired loft emits ~35 dense
    stations and the printer must not care about them, which is a rule
    about where a break may LAND, not about how many there must be. Read
    the strong way it cut a 480 mm wing into four panels a side with
    every one under a third of the envelope -- four parts, six bonded
    faces and three steps in the skin, to describe curvature the loft had
    already described.

    The invariant that actually matters is minimality: no two adjacent
    panels may be merged. Merging them must either overflow the Z
    envelope or swallow the elevon's root station, which has to be a
    joint because a trailing edge cannot vanish mid-panel."""
    from washout.geom import cst as _cst
    from washout.search.design import Mission, N_DIM, build, unit_to_physical

    base = _cst.load_selig(ASSETS / "mh45.dat")
    u = np.random.default_rng(1).random(N_DIM)
    p = build(u, Mission.trainer_v3(), base)
    phys = unit_to_physical(u)
    s = vase.PrintSettings(elevon_chord=phys["elevon_chord"],
                           elevon_eta=phys["elevon_eta"])
    assert len(p.stations) > 2 * len(p.controls)
    spans = vase.panel_etas(p, s)
    limit = s.bed_z_mm - 8.0
    hinge = phys["elevon_eta"]
    for (a, _), (b0, b) in zip(spans, spans[1:]):
        merged = vase.arc_length_mm(p, a, b)
        swallows = a < hinge - 1e-9 < b
        assert merged > limit or swallows, (
            f"panels {a:.3f}-{b0:.3f} and {b0:.3f}-{b:.3f} merge to "
            f"{merged:.0f} mm, inside the {limit:.0f} mm envelope, and do "
            f"not straddle the hinge at {hinge:.3f}: that joint is not needed")
    # and every panel still fits
    for a, b in spans:
        assert vase.arc_length_mm(p, a, b) <= limit + 1e-6


def test_the_design_sheet_never_takes_the_export_down(tmp_path, monkeypatch):
    """A plotting bug must not cost the printable parts.

    do_export draws the figure BEFORE it writes the STLs, and caught only
    ImportError -- so any exception inside the plotting code at the end
    of a two-hour search would have aborted the export with nothing
    written. The 3D panel is the most fragile part (it is the newest and
    leans on the most matplotlib), so it is forced to fail here: the sheet
    must still be written without it. Then the whole sheet is forced to
    fail: figure() must return rather than raise.
    """
    from washout import report
    from washout.geom import cst as _cst
    from washout.geom import fairness as fz
    from washout.search.design import Mission, N_DIM, build, evaluate

    base = _cst.load_selig(ASSETS / "mh45.dat")
    m = Mission.micro()
    rng = np.random.default_rng(2)
    u = None
    for _ in range(200):
        cand = rng.random(N_DIM)
        p = build(cand, m, base)
        if p.is_valid()[0] and not fz.measure(p).violations(m.fairness):
            u = cand
            break
    assert u is not None, "no fair micro design in 200 draws"
    s = vase.PrintSettings()
    ev = evaluate(u, m, base, s)

    def boom(*_a, **_k):
        raise RuntimeError("simulated 3D failure")

    monkeypatch.setattr(report, "_draw_3d", boom)
    out = report.figure(ev, s, tmp_path / "no3d.png", title="t")
    assert out.exists(), "sheet must still be written when the 3D view fails"

    monkeypatch.setattr(report, "_figure", boom)
    out = report.figure(ev, s, tmp_path / "none.png", title="t")
    assert out == tmp_path / "none.png"          # returned, did not raise
