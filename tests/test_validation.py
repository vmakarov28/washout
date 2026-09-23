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


def _fin_on(p, h: float, below: float = 0.0, chord: float | None = None):
    """A rectangular plate at each tip of `p`: full tip chord by default
    (an END PLATE, the geometry Hoerner's rule was measured on), or
    `chord` long."""
    from washout.aero.fins import TipFins
    c = p.at(1.0).chord_m if chord is None else chord
    return TipFins(area_m2=h * c, height_m=h, root_chord_m=c,
                   tip_chord_m=c, below_frac=below, sweep_deg=0.0,
                   thickness_m=0.002, x_root_le_m=p.at(1.0).x_le_m,
                   y_m=p.half_span_m, z_root_m=0.0)


def test_the_branched_wake_reduces_to_the_chain_without_fins():
    """Tip fins make the Trefftz wake a T at each tip, which the chain of
    strips could not hold, so the sheet is solved as nodes and strips.
    On a wing with nothing hanging from it the general form must be the
    chain, to round-off -- otherwise every fin-less design in every
    generation would move for no physical reason."""
    from washout.aero.vlm import trefftz_cdi
    p = rect(6.0, cst.naca4("0012"))
    v = VLM(p, ns=40, nc=6)
    a = np.radians(5.0)
    gamma = v._lu @ (-(v.lat.normal @ np.array([np.cos(a), 0.0, np.sin(a)])))
    g_strip = np.bincount(v.lat.strip, weights=gamma)
    pts, edge, g = v._sheet(g_strip)
    general = trefftz_cdi(pts, edge, g, None, None, None, v.lat.area)
    assert general == pytest.approx(v.solve(5.0, 0.25).CDi, rel=1e-12)


def test_tip_fins_are_winglets_in_the_lattice():
    """The fins used to be flat plates for yaw stability ONLY, their
    end-plate effect deliberately left unclaimed -- while a curled wing
    tip, which IS in the lattice, got its full benefit. So the search
    could compare a winglet against a curl only on an uneven field, and
    micro_fpv's winner curled 27% of its semi-span and carried no fins.

    In the lattice, a vertical plate at each tip of a planar wing must
    (Munk: moving shed vorticity out of the plane of the wing):
      * raise the span efficiency on the SAME projected span by what end
        plates are measured to give: Hoerner's rule AR_eff = AR (1 + 1.9
        h/b): x1.095, x1.19 and x1.38 at h/b = 0.05, 0.1 and 0.2. The
        lattice gives 1.111, 1.226 and 1.401 -- within 3% -- and the band
        here is 5%, the scatter of the measurements the rule was fitted
        to;
      * load a fin standing above the tip INBOARD -- the tip vortex
        carries the flow round the tip from below, so above it the flow
        runs inward -- and the two fins' side forces must cancel;
      * load a fin split evenly above and below the tip antisymmetrically,
        with no net side force at all."""
    p = rect(6.0, cst.naca4("0012"))
    bare = VLM(p, ns=40, nc=6).solve(5.0, 0.25)
    for h_over_b, hoerner in ((0.05, 1.095), (0.10, 1.19), (0.20, 1.38)):
        h = h_over_b * p.span_m
        v = VLM(p, ns=40, nc=6, fins=_fin_on(p, h))
        pt = v.solve(5.0, 0.25)
        gain = (pt.CL ** 2 / pt.CDi) / (bare.CL ** 2 / bare.CDi)
        assert 1.0 < gain, h_over_b
        assert gain == pytest.approx(hoerner, rel=0.05), (h_over_b, gain)

        a = np.radians(5.0)
        vinf = np.array([np.cos(a), 0.0, np.sin(a)])
        gam = v._lu @ (-(v.lat.normal @ vinf))
        fin = v.lat.strip >= v.lat.n_wing_strips
        dF = gam[:, None] * np.cross(vinf, v.lat.b - v.lat.a)
        stbd = fin & (v.lat.cp[:, 1] > 0)
        port = fin & (v.lat.cp[:, 1] < 0)
        assert dF[stbd, 1].sum() < 0.0, "the starboard fin must pull inboard"
        assert dF[stbd, 1].sum() == pytest.approx(-dF[port, 1].sum(), rel=1e-9)

    v = VLM(p, ns=40, nc=6, fins=_fin_on(p, 0.1 * p.span_m, below=0.5))
    a = np.radians(5.0)
    vinf = np.array([np.cos(a), 0.0, np.sin(a)])
    gam = v._lu @ (-(v.lat.normal @ vinf))
    dF = gam[:, None] * np.cross(vinf, v.lat.b - v.lat.a)
    fin = v.lat.strip >= v.lat.n_wing_strips
    stbd = fin & (v.lat.cp[:, 1] > 0)
    assert abs(dF[stbd, 1].sum()) < 1e-9


def test_a_fin_on_a_curled_tip_converges_as_it_is_refined():
    """On fpv_micro_fpv_s4 -- swept fins on a 35-degree curled tip -- the
    first fin lattice's side-force slope went -0.24, -0.18, -0.05, +11.8
    as the fin was refined, and span efficiency 1.20 to 0.001. Three
    separate faults, each found by this refinement:
      * strips clustered at the fin's free tip made 0.2 mm strips under
        6 mm chordwise panels (the circulation alternated in sign);
      * the wing's last cosine strip puts control points a fraction of a
        millimetre from the fin's root vortices, uncored;
      * a fin root that followed the tip's camber line climbed 3 mm along
        the chord, and its own straight trailing legs fell through the
        control points of the strips below.
    Uniform strips, a flat root, and a core across the junction sized to
    the spread of the wing's tip vortices about it: a refinement must now
    move the answer by little and in one direction, and the fin's slope
    must sit between the isolated-plate and reflection-plane closed forms.

    (It does NOT land on the flat-plate model's 1.5x calibrated slope: on
    this curled tip the lattice finds the fins about 20% less effective in
    yaw. That calibration was made for a fin standing on a wing tip, and
    this one stands on a winglet.)"""
    import json as _json
    from washout.search.design import Mission, evaluate
    from washout.aero import dynamics as dyn
    m = Mission.micro_fpv()
    d = _json.loads((ASSETS.parent / "results" / "fleet" / "fpv_micro_fpv_s4"
                     / "design.json").read_text(encoding="utf-8"))
    ev = evaluate(np.array(d["u"]), m, cst.load_selig(ASSETS / "mh45.dat"),
                  vase.PrintSettings(filament_density_gcc=0.55,
                                     spar_d_mm=m.spar_d_mm),
                  size_structure=False)
    plan, fins, t = ev.plan, ev.fins, ev.trim
    assert fins is not None and t is not None
    _, _, _, z_cg = dyn.inertia(plan, ev.mass, t.alpha_deg, 0.5, fins)
    ref = np.array([t.x_cg_m, 0.0, z_cg])
    d0 = VLM(plan).lateral_derivatives(t.alpha_deg, ref)
    got = []
    for nf, ncf in ((8, 4), (12, 6), (16, 8), (24, 8)):
        v = VLM(plan, fins=fins, nf=nf, ncf=ncf)
        dd = v.lateral_derivatives(t.alpha_deg, ref) - d0
        got.append((dd[0, 0], dd[2, 0], v.solve(t.alpha_deg, t.x_cg_m).e_oswald))
    cy = np.array([g[0] for g in got])
    cn = np.array([g[1] for g in got])
    e = np.array([g[2] for g in got])
    assert np.all(np.diff(cy) > 0.0), cy          # monotone, shrinking
    assert abs(cy[-1] - cy[-2]) < 0.05 * abs(cy[-1]), cy
    assert abs(cn[-1] - cn[-2]) < 0.05 * abs(cn[-1]), cn
    assert np.ptp(e) < 0.03, e
    from washout.aero.fins import lift_slope
    per_fin = cy[-1] * plan.area_m2 / (2.0 * fins.area_m2)
    ar = fins.height_m ** 2 / fins.area_m2
    assert -lift_slope(2.0 * ar) < per_fin < -lift_slope(ar), (per_fin, ar)


def test_a_fin_on_a_wing_tip_sits_between_the_two_closed_forms():
    """The flat-plate fin model took Helmbold's slope on 1.5x the
    geometric aspect ratio -- the CALIBRATED part, because a wing tip is a
    partial reflection plane for the fin root. The lattice now solves the
    fin with the wing, so its side-force slope has to fall between the
    two closed forms that bracket it: the isolated plate (AR) and the fin
    on an infinite reflection plane (2 AR)."""
    from washout.aero.fins import lift_slope
    p = rect(6.0, cst.naca4("0012"))
    ref = np.array([0.25, 0.0, 0.0])
    d0 = VLM(p, ns=40, nc=6).lateral_derivatives(0.0, ref)
    for h_over_b in (0.05, 0.10):
        h = h_over_b * p.span_m
        f = _fin_on(p, h, chord=0.3 * h)       # a fin, not an end plate
        d1 = VLM(p, ns=40, nc=6, fins=f).lateral_derivatives(0.0, ref)
        per_fin = (d1[0, 0] - d0[0, 0]) * p.area_m2 / (2.0 * f.area_m2)
        ar = f.height_m ** 2 / f.area_m2
        assert -lift_slope(2.0 * ar) < per_fin < -lift_slope(ar), (
            f"h/b {h_over_b}: {per_fin:+.3f} outside "
            f"[{-lift_slope(2 * ar):+.3f}, {-lift_slope(ar):+.3f}]")


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


# ------------------------------------------ panels are slices of the loft

def _fleet_design(name):
    import json as _json
    from washout.search.design import Mission, build, unit_to_physical
    root = ASSETS.parent / "results" / "fleet"
    index = _json.loads((root / "index.json").read_text(encoding="utf-8"))
    d = _json.loads((root / index[name] / "design.json").read_text(encoding="utf-8"))
    u = np.array(d["u"])
    mission = getattr(Mission, name)()
    plan = build(u, mission, cst.load_selig(ASSETS / "mh45.dat"))
    p = unit_to_physical(u)
    s = vase.PrintSettings(elevon_chord=p["elevon_chord"],
                           elevon_eta=p["elevon_eta"])
    return plan, s


def _off_loft_mm(plan, stack, x_lo=0.02, x_hi=0.85, every=6):
    """Largest distance from a printed skin vertex, put back where it
    flies, to the loft's own section at that vertex's span station --
    measured to the ANALYTIC surface on a grid in t = sqrt(x), which is
    where CST is a polynomial and where the nose is resolved. The first
    2% and the last 15% of chord are the nozzle's, not the loft's: the
    nose and the trailing edge are thickened to a printable floor on
    purpose."""
    H = plan.half_span_m * 1000.0
    worst = 0.0
    n_up = (stack.contours.shape[1] + 1) // 2
    for k in range(0, len(stack.z_mm), every):
        c = stack.contours[k]
        P = stack.to_flight(c[:, 0], c[:, 1], np.full(len(c), stack.z_mm[k]))
        for i in range(0, len(c), 2):
            x, y, z = P[i]
            st = plan.at(float(np.clip(y / H, 0.0, 1.0)))
            cm = st.chord_m * 1000.0
            a = np.radians(-st.twist_deg)
            rx = (x - st.x_le_m * 1000.0 - 0.25 * cm) / cm
            rz = (z - st.z_le_m * 1000.0) / cm
            u = rx * np.cos(a) + rz * np.sin(a) + 0.25
            v = -rx * np.sin(a) + rz * np.cos(a)
            if not x_lo <= u <= x_hi:
                continue
            t = np.clip(np.sqrt(u) + np.linspace(-0.06, 0.06, 2001), 0.0, 1.0)
            ys = (st.airfoil.y_upper(t * t) if i < n_up
                  else st.airfoil.y_lower(t * t))
            worst = max(worst, float(np.hypot(t * t - u, ys - v).min()) * cm)
    return worst


@pytest.mark.parametrize("name", ["trainer_v3", "micro_fpv"])
def test_a_layer_is_the_loft_cut_square_to_its_panel(name):
    """Every printed panel is the loft, not a straightened copy of it.

    `build_stack` used to stack the loft's `y = const` sections straight up
    the print axis and never read `z_le`, so each panel printed straight
    while the lattice scored a curve: micro_fpv's outer panel sat 9.0 mm
    off the loft, and every tilted panel printed 1/cos(phi) too thick when
    assembled (+13% on the trainer's tip panel). Each layer is now the
    loft cut square to the panel's own axis, and every skin vertex, put
    back where it flies, must lie on the loft's surface at its own span
    station. 0.06 mm is under a sixth of a bead; the slice as built is
    within 0.045 mm on the whole fleet."""
    plan, s = _fleet_design(name)
    for stack in vase.build_panels(plan, s, z_step_mm=2.0):
        off = _off_loft_mm(plan, stack)
        assert off < 0.06, f"{stack.name}: a printed vertex is {off:.3f} mm off the loft"


def test_the_centre_body_meets_its_mirror_flat():
    """The centre body's axis is horizontal, so its root face IS the
    symmetry plane. Any tilt and the two halves meet in a V: open on one
    side and interpenetrating on the other, which two printed parts
    cannot do."""
    plan, s = _fleet_design("micro_fpv")
    p0 = vase.build_panels(plan, s, z_step_mm=2.0)[0]
    assert p0.frame.phi_deg == 0.0
    c = p0.contours[0]
    P = p0.to_flight(c[:, 0], c[:, 1], np.zeros(len(c)))
    assert np.abs(P[:, 1]).max() < 1e-6


@pytest.mark.parametrize("name", ["trainer_v3", "micro_fpv"])
def test_two_panels_never_share_material_at_a_joint(name):
    """A joint between panels on different axes cannot be one plane. Hinged
    about the chord line, the two parts would interpenetrate above it and
    gape below; hinged about the joint section's top skin when the wing
    turns up, every point of each lies on its own side and the joint opens
    as a wedge on the lower skin instead -- glue, which the build sheet
    sizes. Checked on the actual printed layers either side of every
    joint, put back where they fly.

    gen7's trainer found the hole in the pivot rule: with both joints on
    their top points its first joint turned -0.3 degrees instead of +10,
    and the fallback to a chord-line pivot put the 10 degrees back about
    the chord line -- p0 and p1 shared 4.3 mm. That joint now sits at the
    height where it does not turn, and the faces mate flat."""
    plan, s = _fleet_design(name)
    pans = vase.build_panels(plan, s, z_step_mm=1.0)
    for inner, outer in zip(pans, pans[1:]):
        f_in, f_out = inner.frame, outer.frame
        q = np.array(f_out.origin_yz_mm)
        a_in = np.array([f_in.cos, f_in.sin])
        a_out = np.array([f_out.cos, f_out.sin])
        c = outer.contours[0]
        P = outer.to_flight(c[:, 0], c[:, 1], np.zeros(len(c)))[:, 1:]
        into_inner = -((P - q) @ a_in).min()
        c = inner.contours[-1]
        P = inner.to_flight(c[:, 0], c[:, 1],
                            np.full(len(c), inner.z_mm[-1]))[:, 1:]
        into_outer = ((P - q) @ a_out).max()
        assert into_inner < 0.05 and into_outer < 0.05, (
            f"{inner.name}/{outer.name}: parts overlap by "
            f"{max(into_inner, into_outer):.2f} mm")
        if abs(f_out.kink_deg) > 1.0:
            assert f_out.pivot in ("upper", "lower") and f_out.wedge_mm > 0.0


def test_an_elevon_is_cut_in_its_wing_panels_frame():
    """The elevon and the wing panel it came off must share a frame, layer
    for layer, or their hinge faces are two different surfaces meeting at
    the angle between two axes."""
    from washout.printing import elevons
    plan, s = _fleet_design("trainer_v3")
    spans = vase.panel_etas(plan, s)
    wing = vase.build_panels(plan, s, z_step_mm=2.0)
    elv = elevons.build_elevons(plan, s, spans, 16.0, z_step_mm=2.0)
    by_span = {(round(p.frame.eta0, 9), round(p.frame.eta1, 9)): p.frame for p in wing}
    for e in elv:
        assert e.frame == by_span[(round(e.frame.eta0, 9), round(e.frame.eta1, 9))]


def test_every_joint_is_on_the_build_sheet():
    """The angle each joint turns through, and the wedge it opens, are the
    builder's to set and to fill. Before this nothing reported either:
    a 17 degree kink on micro_fpv's only joint, and nowhere to read it."""
    import json as _json
    from washout import build_sheet
    from washout.search.design import Mission, evaluate
    root = ASSETS.parent / "results" / "fleet"
    index = _json.loads((root / "index.json").read_text(encoding="utf-8"))
    d = _json.loads((root / index["micro_fpv"] / "design.json").read_text(encoding="utf-8"))
    ev = evaluate(np.array(d["u"]), Mission.micro_fpv(),
                  cst.load_selig(ASSETS / "mh45.dat"),
                  vase.PrintSettings(filament_density_gcc=0.55, bed_z_mm=250.0),
                  want_panels=True, z_step_mm=2.0)
    text = build_sheet.render(ev, ev.panels + ev.elevons, ev.print_settings)
    assert "## Joints" in text
    wing = [p for p in ev.panels if p.role == "wing"]
    for prev, p in zip(wing, wing[1:]):
        row = next(line for line in text.splitlines()
                   if line.startswith(f"| `{prev.name}` / `{p.name}`"))
        assert f"{p.frame.kink_deg:+.1f} deg" in row
        assert f"{p.frame.wedge_mm:.1f} mm" in row
    m = build_sheet.manifest(ev, ev.panels + ev.elevons, ev.print_settings)
    assert all("placement" in part for part in m["half_wing_parts"]
               if part["profile"] == "vase")


def test_a_joint_that_does_not_turn_says_so():
    """gen7's trainer has a joint placed where it does not turn. The
    Joints table knew only skins and the chord line, so it would have told
    the builder that joint hinges about the chord line -- the pivot that
    put 4.3 mm of shared material into it."""
    from washout import build_sheet
    from washout.printing import frames
    plan, s = _fleet_design("trainer_v3")
    wing = vase.build_panels(plan, s, z_step_mm=2.0)
    flat = [p for p in wing if p.frame.pivot == "flat"]
    assert flat, "the tracked trainer is the design with a flat joint"
    rows = frames.joint_report([p.frame for p in wing])
    assert any("mate flat" in r for r in rows)
    from washout.search.design import Mission, evaluate
    import json as _json
    root = ASSETS.parent / "results" / "fleet"
    index = _json.loads((root / "index.json").read_text(encoding="utf-8"))
    d = _json.loads((root / index["trainer_v3"] / "design.json").read_text(encoding="utf-8"))
    ev = evaluate(np.array(d["u"]), Mission.trainer_v3(),
                  cst.load_selig(ASSETS / "mh45.dat"),
                  vase.PrintSettings(filament_density_gcc=0.55, bed_z_mm=250.0),
                  want_panels=True, z_step_mm=2.0)
    text = build_sheet.render(ev, ev.panels + ev.elevons, ev.print_settings)
    for p in flat:
        i = [q.name for q in ev.panels].index(p.name)
        row = next(line for line in text.splitlines()
                   if line.startswith(f"| `{ev.panels[i - 1].name}` / `{p.name}`"))
        assert "mate flat" in row and "chord line" not in row, row


# ------------------------------------------------- a spar is a straight tube

def _inside_section(plan, eta, x_mm, z_mm):
    """Independent of the fitter: is each flight (x, z) inside the placed
    section at eta? Built from `Planform.at` and the CST surfaces alone."""
    st = plan.at(float(eta))
    c = st.chord_m * 1000.0
    a = np.radians(-st.twist_deg)
    u = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, 401)))
    out = []
    for v in (st.airfoil.y_upper(u), st.airfoil.y_lower(u)):
        xs = st.x_le_m * 1000.0 + 0.25 * c + c * ((u - 0.25) * np.cos(a) - v * np.sin(a))
        zs = st.z_le_m * 1000.0 + c * ((u - 0.25) * np.sin(a) + v * np.cos(a))
        o = np.argsort(xs)
        out.append((xs[o], zs[o]))
    (xu, zu), (xl, zl) = out
    x_mm, z_mm = np.asarray(x_mm), np.asarray(z_mm)
    return ((x_mm > max(xu[0], xl[0])) & (x_mm < min(xu[-1], xl[-1]))
            & (z_mm < np.interp(x_mm, xu, zu)) & (z_mm > np.interp(x_mm, xl, zl)))


def _tube_ring(f, y, r, n=24):
    """Points around a straight tube's section by the plane at span y: an
    ellipse, because the tube crosses the plane at an angle."""
    sx, sz = f.slope
    x0, z0 = f.centre_mm(y)
    m = float(np.hypot(sx, sz))
    e1 = np.array([sx, sz]) / m if m > 1e-12 else np.array([1.0, 0.0])
    e2 = np.array([-e1[1], e1[0]])
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    p = (np.cos(th)[:, None] * r * np.sqrt(1 + sx * sx + sz * sz) * e1
         + np.sin(th)[:, None] * r * e2)
    return x0 + p[:, 0], z0 + p[:, 1]


@pytest.mark.parametrize("name", ["trainer_v3", "demon1", "micro", "micro_fpv"])
def test_a_spar_is_a_straight_tube_inside_the_wing(name):
    """The fit asked whether the section was deep enough at one chord
    fraction, station by station, which let the tube bend with the sweep
    and the dihedral: it claimed eta 0.76-1.00 where a straight tube
    reaches 0.66-0.69, and the build sheet asked for one tube tip to tip
    that left the skin at eta 0.25-0.43 on every aircraft.

    Checked here without the fitter's own machinery: the straight tube,
    with its fit clearance, is inside the section at 400 stations from
    the root to its reach -- more than twice the fitter's resolution --
    and would NOT be at a station just past where the fit stopped, so the
    reach is the geometry's and not a cautious guess."""
    from washout import spars as sp
    from washout.search.design import Mission
    plan, s = _fleet_design(name)
    mission = getattr(Mission, name)()
    H = plan.half_span_m * 1000.0
    for f in sp.fit_all(plan, mission.spars, 0.45,
                        min_reach=mission.min_spar_reach_frac):
        r = sp.fit_radius_mm(f.spec, 0.45)
        for y in np.linspace(0.0, f.reach_y_mm, 400):
            x, z = _tube_ring(f, y, r)
            assert _inside_section(plan, y / H, x, z).all(), (
                f"{name} {f.spec.name}: the straight tube leaves the wing at "
                f"eta {y / H:.3f}, inside its claimed reach {f.reach_eta:.3f}")
        if f.reach_eta < 0.97:
            sx, sz = f.slope
            lean = np.hypot(sx, sz)
            beyond = (f.reach_y_mm + r * lean / np.sqrt(1 + lean * lean)
                      + 2.0 * H / 160)
            x, z = _tube_ring(f, beyond, r)
            assert not _inside_section(plan, beyond / H, x, z).all(), (
                f"{name} {f.spec.name}: the tube still fits past its reach")


def test_the_old_fit_let_the_tube_bend():
    """The finding, pinned so it cannot quietly come back. The trainer's LE
    spar, asked the old question -- is there depth at 0.21c, station by
    station -- 'reaches' the tip. The straight tube does not get past
    seven tenths of the span."""
    from washout import spars as sp
    from washout.search.design import Mission
    plan, _ = _fleet_design("trainer_v3")
    spec = Mission.trainer_v3().spars[0]
    assert sp.reach_of(plan, 0.21, spec, 0.45) >= 0.95
    f = sp.fit_all(plan, (spec,), 0.45)[0]
    assert f.reach_eta <= 0.72 and not f.one_piece


@pytest.mark.parametrize("name", ["trainer_v3", "demon1", "micro", "micro_fpv"])
def test_every_fitted_tube_passes_the_print_bore(name):
    """The fit works on the loft in the flight frame; the bore gate works on
    the printed layers, at the tube's own centre, with the ellipse a tube
    cuts in a tilted layer. The fit's clearance is the stricter of the two,
    so a tube the fit seats must pass the gate -- and micro's did not, by
    0.09 mm at its last layer, because a cut tube's end face is square to
    its axis and on a 46 degree tube reaches 3.3 mm further out than the
    axis does, into section no station had checked."""
    from dataclasses import replace
    from washout import spars as sp
    from washout.search.design import Mission
    plan, s = _fleet_design(name)
    mission = getattr(Mission, name)()
    spans = vase.panel_etas(plan, s)
    d = mission.spar_d_mm
    fits = sp.fit_all(plan, mission.spars, 0.45, spans,
                      min_reach=mission.min_spar_reach_frac)
    lines = tuple((f.root_xz_mm[0], f.root_xz_mm[1], f.slope[0], f.slope[1],
                   f.reach_y_mm, 0.5 * d + 0.225 + f.spec.clearance_mm, d)
                  for f in fits)
    s = replace(s, spar_d_mm=d, spar_lines=lines)
    for pan in vase.build_panels(plan, s, z_step_mm=0.5):
        v, z = vase.spar_fit(pan)
        assert v >= d, f"{pan.name}: {v:.2f} mm of bore at z {z:.1f} for a {d:.0f} mm tube"


def test_a_tube_is_weighed_where_its_mass_is():
    """A straight tube swept 47 degrees has its mass well aft of its root
    seat. Charged at the seat, as every tube was, micro_fpv's CG sat well
    forward of where it is and its static margin read 0.132 where it is
    0.09 -- inside its band where it is outside it."""
    from washout import spars as sp
    from washout import structure as st
    from washout.search.design import Mission
    plan, _ = _fleet_design("micro_fpv")
    m = Mission.micro_fpv()
    f = sp.fit_all(plan, m.spars, 0.45, min_reach=m.min_spar_reach_frac)[0]
    root_c = plan.stations[0].chord_m * 1000.0
    (_, kg, x_frac), = st.spar_masses([f], 6.0, root_chord_mm=root_c)
    assert x_frac * root_c == pytest.approx(f.centroid_x_mm())
    assert f.centroid_x_mm() > f.root_xz_mm[0] + 50.0
    tube = st.tube_for_od(6.0)
    assert kg == pytest.approx(tube.mass_g(2.0 * f.reach_mm) * 1e-3)


def test_tube_stiffness_stops_where_the_tube_does():
    """The torsion model added every tube at every span station, so the
    outer third of each wing was stiffened by carbon that ends at eta
    0.66 -- and reversal, the binding aeroelastic limit on the fleet,
    goes as the square root of that stiffness."""
    from washout import aeroelastic as ael
    from washout import spars as sp
    from washout.search.design import Mission
    plan, _ = _fleet_design("micro_fpv")
    m = Mission.micro_fpv()
    fits = sp.fit_all(plan, m.spars, 0.45, min_reach=m.min_spar_reach_frac)
    assert ael.gj_spars_nmm2(fits, eta=0.5 * fits[0].reach_eta) > 0.0
    assert ael.gj_spars_nmm2(fits, eta=min(fits[0].reach_eta + 0.05, 1.0)) == 0.0


def test_the_build_sheet_asks_for_the_tube_that_exists():
    """BUILD.md said 'each tube runs tip to tip through the centre body --
    one length, not two meeting at the centreline' on a 48.7 degree swept
    wing, where that tube leaves the skin a quarter of the way out. It now
    says what the fit found: one tube a side and the V joiner's angles, or
    one tube only when the line really is parallel to the span."""
    import json as _json
    from washout import build_sheet
    from washout.search.design import Mission, evaluate
    root = ASSETS.parent / "results" / "fleet"
    index = _json.loads((root / "index.json").read_text(encoding="utf-8"))
    d = _json.loads((root / index["micro_fpv"] / "design.json").read_text(encoding="utf-8"))
    ev = evaluate(np.array(d["u"]), Mission.micro_fpv(),
                  cst.load_selig(ASSETS / "mh45.dat"),
                  vase.PrintSettings(filament_density_gcc=0.55, spar_d_mm=6.0,
                                     bed_z_mm=250.0),
                  want_panels=True, z_step_mm=2.0)
    text = build_sheet.render(ev, ev.panels + ev.elevons, ev.print_settings)
    bom = build_sheet.bom(ev, ev.panels + ev.elevons)
    for f in ev.spar_fits:
        if f.one_piece:
            assert "tip to tip" in text
        else:
            assert "tip to tip through the centre body" not in text
            assert f"V: {2 * f.sweep_deg:.0f} deg in plan" in text
            assert "V joiner" in bom


def test_an_untrimmed_design_keeps_what_it_knows():
    """A design rejected at trim used to come back with its plan, its mass
    and nothing else: no print settings, no spar fits, no mechanism, no
    parts -- all computed, all thrown away. So exporting one, which is how
    you find out WHY it will not fly, crashed on settings that were never
    handed back, and every check of the parts silently skipped it.

    micro's tracked design is the live example: weighed where its swept
    tube's mass really is, it no longer trims. It must still carry its
    parts, and its build sheet must say plainly that it does not trim."""
    import json as _json
    from washout import build_sheet
    from washout.search.design import Mission, evaluate
    root = ASSETS.parent / "results" / "fleet"
    index = _json.loads((root / "index.json").read_text(encoding="utf-8"))
    d = _json.loads((root / index["micro"] / "design.json").read_text(encoding="utf-8"))
    ev = evaluate(np.array(d["u"]), Mission.micro(),
                  cst.load_selig(ASSETS / "mh45.dat"),
                  vase.PrintSettings(filament_density_gcc=0.55, spar_d_mm=8.0,
                                     bed_z_mm=250.0),
                  want_panels=True, z_step_mm=2.0)
    if ev.trim is not None:
        pytest.skip("micro trims again -- re-searched; nothing left to pin here")
    assert any("no trim angle" in r for r in ev.reasons)
    assert ev.print_settings is not None and ev.panels
    assert ev.spar_fits and ev.linkage is not None and ev.mass is not None
    text = build_sheet.render(ev, ev.panels + ev.elevons, ev.print_settings)
    assert "does not trim" in text
    assert "## Joints" in text and "## Spar cut list" in text


# ------------------------------------------- the nose, and the chamfer corner

def test_the_nose_is_round_not_a_flat():
    """`thicken_for_nozzle` exists for the TRAILING edge, and it applied its
    floor along the whole chord -- including the nose, where the vertical
    thickness is zero by construction because the nose is round. So the
    shared leading-edge vertex went to +0.5 mm and the next point below it
    to -0.47 mm, 0.04 mm aft: every printed section had a 1 mm vertical
    flat for a nose, and no smooth skin could be fitted through it without
    a seam at the leading edge.

    The floor now blends in over 0.35c-0.65c. The nose is the aerofoil's,
    the trailing edge still carries its millimetre, and the wall gates no
    longer mistake one bead turning round the nose for two walls: they
    skip neighbours by distance along the loop, not only by index.

    One gate got harder, and it is recorded rather than tuned: micro's
    centre body overhangs 51.4 degrees at z = 14 mm, where its rounded
    planform nose sweeps fastest, against 48.8 with the flat nose. The
    flat was not more printable; it read better under a metric that
    measures to the nearest wall."""
    from washout.search.design import Mission, build
    base = cst.load_selig(ASSETS / "mh45.dat")
    plan, s = _fleet_design("micro_fpv")
    for e in (0.0, 0.5, 0.9):
        st = plan.at(e)
        c = st.chord_m * 1000.0
        raw = st.airfoil.coords(s.contour_points)
        th = vase.thicken_for_nozzle(raw, c, s)
        n = (len(raw) + 1) // 2
        assert np.allclose(th[n - 3:n + 2], raw[n - 3:n + 2]), "the nose moved"
        up, lo = th[:n][::-1], th[n - 1:]
        assert (up[-1, 1] - lo[-1, 1]) * c >= s.min_te_mm - 1e-6, "the TE lost its floor"
    for stack in vase.build_panels(plan, s, z_step_mm=2.0):
        assert vase.check(stack).ok, stack.name


def test_the_elevon_chamfer_corner_is_a_vertex_on_every_layer():
    """The corner where the bevel meets the lower skin fell between two
    cosine samples, so the printed contour cut it with a chord, and as it
    moved along the span the turn hopped from one vertex to the next --
    triangles twisted across it in the STL, and the CAD export, skinning
    pieces that were not the same piece from one layer to the next,
    wandered 3.8 mm between sections. It is a vertex now, at the same
    index on every layer, and the whole turn happens there."""
    from washout.printing import elevons as elv
    plan, s = _fleet_design("trainer_v3")
    spans = vase.panel_etas(plan, s)
    for part in elv.build_elevons(plan, s, spans, 16.0, z_step_mm=2.0):
        n = part.n_upper
        corner = set()
        for c in part.contours:
            a = np.diff(c[n:], axis=0)
            a /= np.linalg.norm(a, axis=1, keepdims=True)
            turn = np.degrees(np.arccos(np.clip((a[:-1] * a[1:]).sum(1), -1, 1)))
            corner.add(int(np.flatnonzero(turn >= 30.0)[-1]))
        assert len(corner) == 1, (part.name, sorted(corner))


def test_a_corner_between_two_samples_is_given_a_vertex_before_fitting():
    """demon1's elevon has a second corner the chamfer fix does not place:
    the nose floor meets the chamfer at 45 degrees between two grid
    points, its turn split 28.5 + 28.2 on some layers and 45 + 12 on
    others. Fitted across it, the chamfer's cubic folded back over the
    nose flat and the root cap came out a self-intersecting face -- an
    invalid solid in the gen7 export. Split at it without a vertex there,
    the boundary hopped a grid point along the part and the surface missed
    the held-out layers by 0.08 mm. `sharpen` inserts the true corner, the
    meeting of the two straight runs, and keeps every printed point.

    Here a floor and a 45-degree chamfer sampled so the corner falls
    between two samples, each taking a quarter of the kink or more: the
    inserted point is the exact intersection, the whole turn is then at
    that one vertex, and the pieces split there."""
    from washout.cad import bspline as bs
    upper = np.array([[10.0, 1.0], [5.0, 2.0], [0.5, 2.0], [0.0, 2.0]])
    n = len(upper)                            # loop[n - 1 : n + 1] is the nose flat
    for off in (0.04, 0.08, 0.32):     # 35+10, 31+14 and 16+29 degrees
        # floor along y = 1 to the corner at x = 1, then y = 2 - x
        xs = np.arange(off, 6.0, 0.4)
        lower = np.array([[x, min(1.0, 2.0 - x)] for x in xs])
        loop = np.vstack([upper, [[0.0, 1.0]], lower, [[10.0, 0.0]]])
        loop = np.column_stack([loop, np.zeros(len(loop))])
        sharp = bs.sharpen(loop, n)
        assert len(sharp) == len(loop) + 1, off
        k = int(np.flatnonzero(np.all(np.isclose(sharp[:, :2], [1.0, 1.0]), axis=1))[0])
        for p in loop:
            assert any(np.array_equal(p, q) for q in sharp), "a printed point was dropped"
        turns = bs._turns_deg(sharp[:, :2])   # turns[i] is at vertex i + 1
        assert np.isclose(turns[k - 1], 45.0), off
        assert turns[k - 2] < 1e-4 and turns[k] < 1e-4, off
        b = bs.piece_bounds(sharp, n, every_corner=True)
        assert (n, k) in b and any(q[0] == k for q in b), (off, b)


# ------------------------------------------------- the joint wedge inserts


def _fleet_eval(name, z_step_mm=2.0):
    import json as _json
    from washout.search.design import Mission, evaluate
    root = ASSETS.parent / "results" / "fleet"
    index = _json.loads((root / "index.json").read_text(encoding="utf-8"))
    d = _json.loads((root / index[name] / "design.json").read_text(encoding="utf-8"))
    m = getattr(Mission, name)()
    return evaluate(np.array(d["u"]), m, cst.load_selig(ASSETS / "mh45.dat"),
                    vase.PrintSettings(filament_density_gcc=0.55, spar_d_mm=m.spar_d_mm),
                    want_panels=True, z_step_mm=z_step_mm)


def test_a_single_layer_is_sliced_like_one_in_a_stack():
    """`slice_layers` takes its span rates as differences ACROSS the layers
    it is given. Asked for one layer, the rates were zero: every point slid
    along y alone, and the first joint insert's face sat 4 mm aft of the
    loft on micro_fpv's swept wing. The loft-conformance gate never saw it
    -- a chordwise slide along a nearly flat crest barely leaves the
    surface -- so it is pinned directly: one layer sliced alone must be
    that layer as the stack slices it."""
    from washout.printing import frames as fr
    from washout.printing.inserts import _plain_unit_loop
    plan, s = _fleet_design("micro_fpv")
    pans = vase.build_panels(plan, s, z_step_mm=1.0)
    loop = _plain_unit_loop(s, False)
    for p in pans:
        z = np.arange(0.0, min(p.frame.length_mm, 20.0), 0.5)
        stack, _ = fr.slice_layers(plan, p.frame, z, loop)
        for k in (0, 7):
            one, _ = fr.slice_layers(plan, p.frame, z[k:k + 1], loop)
            assert np.abs(one[0] - stack[k]).max() < 0.01, (p.name, k)


def test_a_turning_joint_is_filled_by_a_printed_insert():
    """The wedge a turning joint leaves -- 6 mm open at micro_fpv's lower
    skin, the loft that belongs to neither panel -- was "fill with glue"
    on the build sheet, a gap in CAD, and never weighed. It is a part now,
    and these are the claims it has to meet:

      * its two big faces LIE ON the two panels' end planes, so the three
        parts share faces;
      * it is watertight;
      * its volume is the loft's own, to 1%: the loft sliced at 200
        planes between the two faces, each slice clipped to the far side
        of the inner panel's tip plane, areas summed -- none of which
        uses the insert's two-face construction, whose side is a band of
        straight rulings between the faces. (tan(k) times the first
        moment of face A about the pivot line was tried as the closed
        form and is 5% out: it assumes the skin rotates about the pivot,
        and the loft's local dihedral moves it in height between the
        planes instead.) Compared bore-less and cut at 0.05 mm, so the
        two describe the same wedge;
      * it is never thicker than the frame's own wedge estimate, and not
        much thinner (the insert stops at the hinge; the estimate spans
        the whole section);
      * it is charged, both sides, in the mass budget."""
    from washout.printing.stl import manifold_report
    ev = _fleet_eval("micro_fpv")
    assert ev.inserts, "micro_fpv's joint turns 17 degrees and must carry an insert"
    ins = ev.inserts[0]
    outer = next(p for p in ev.panels if abs(p.frame.eta0 - ins.eta) < 1e-9)
    inner = next(p for p in ev.panels if abs(p.frame.eta1 - ins.eta) < 1e-9)
    fo, fi = outer.frame, inner.frame
    na = np.array([0.0, fo.cos, fo.sin])
    qa = np.array([0.0, *fo.origin_yz_mm])
    nb = np.array([0.0, fi.cos, fi.sin])
    qb = np.array([0.0, *fi.origin_yz_mm]) + fi.length_mm * nb
    V = ins.flight_verts
    on_a = np.abs((V - qa) @ na) < 1e-6
    on_b = np.abs((V - qb) @ nb) < 1e-6
    assert on_a.sum() >= len(V) // 2 - 1 and on_b.sum() >= len(V) // 2 - 1
    assert (on_a | on_b).all(), "every vertex is on one face or the other"
    assert manifold_report(ins.tris)["watertight"]

    # independent: the loft sliced between the planes, each slice clipped
    from dataclasses import replace as _replace
    from washout.printing import frames as fr
    from washout.printing.inserts import _plain_unit_loop, joint_inserts
    s0 = _replace(ev.print_settings, spar_lines=())
    bare = next(i for i in joint_inserts(ev.plan, s0, ev.panels, min_thickness_mm=0.05)
                if i.joint == ins.joint)
    loop = _plain_unit_loop(s0, s0.elevon_chord > 1e-6
                            and ins.eta >= s0.elevon_eta - 1e-9)
    ss = np.linspace(-(bare.max_gap_mm + 0.5), 0.0, 201)
    layers, _ = fr.slice_layers(ev.plan, fo, ss, loop)
    areas = []
    for s_k, c in zip(ss, layers):
        P = fo.to_flight(c[:, 0], c[:, 1], np.full(len(c), s_k))
        beyond = (P - qb) @ nb                  # > 0: past the inner tip face
        poly, pts2 = [], c
        for i in range(len(c)):                 # Sutherland-Hodgman, one plane
            j = (i + 1) % len(c)
            a_in, b_in = beyond[i] > 0.0, beyond[j] > 0.0
            if a_in:
                poly.append(pts2[i])
            if a_in != b_in:
                t = beyond[i] / (beyond[i] - beyond[j])
                poly.append(pts2[i] + t * (pts2[j] - pts2[i]))
        if len(poly) >= 3:
            q = np.array(poly)
            areas.append(0.5 * abs(np.dot(q[:, 0], np.roll(q[:, 1], -1))
                                   - np.dot(q[:, 1], np.roll(q[:, 0], -1))))
        else:
            areas.append(0.0)
    sliced = float(np.trapezoid(areas, ss))
    assert bare.volume_mm3 == pytest.approx(sliced, rel=0.01), (bare.volume_mm3, sliced)
    assert ins.volume_mm3 < bare.volume_mm3     # the bore and the crest come out

    assert ins.max_gap_mm <= fo.wedge_mm + 1e-6
    assert ins.max_gap_mm >= 0.8 * fo.wedge_mm
    item = next(i for i in ev.mass.items if i.name == f"joint insert {ins.joint}")
    assert item.mass_kg == pytest.approx(2.0 * ins.mass_g(0.55) / 1000.0, rel=1e-9)
    assert item.y_m == pytest.approx(ins.eta * ev.plan.half_span_m, abs=0.01)


def test_a_joint_that_mates_flat_gets_no_insert():
    """gen7's trainer: its first joint does not turn (the faces mate flat)
    and its second turns 22 degrees. One insert, at the second."""
    ev = _fleet_eval("trainer_v3")
    wing = [p for p in ev.panels if p.role == "wing"]
    flat = [j for j, p in enumerate(wing[1:], start=1) if p.frame.pivot == "flat"]
    assert flat and all(i.joint not in flat for i in ev.inserts)
    turning = [j for j, p in enumerate(wing[1:], start=1)
               if p.frame.pivot in ("upper", "lower")]
    assert sorted(i.joint for i in ev.inserts) == turning
    for i in ev.inserts:
        chk = vase.check_insert(i, ev.print_settings)
        assert chk.ok, chk.report()


def test_a_tube_against_the_skin_notches_the_insert_or_fills_it():
    """The spars are SEATED against a skin, so where a joint's wedge is
    thin the tube's bore reaches the insert's outline -- and a closed hole
    one sliver from the edge is not printable. The first insert gated that
    as a failed bore and rejected every design of the flatter-winged gen8
    family with it. Now:
      * where the tube leaves a neck of two beads, the outline is NOTCHED
        round it: a C-shaped channel, the outline area less exactly the
        part of the (grown) disc inside it;
      * where it severs the wedge, the joint is a declared glue fill,
        charged by weight and not gated for print.
    The notch is checked on a shape with a known answer: a 40 x 10
    rectangle (open along its top edge, the crest cut) and a disc of
    radius 3 centred 1 mm inside its bottom edge."""
    from washout.printing.inserts import _notch_one, _points_in_polygon
    rect = np.array([[40.0, 10.0], [40.0, 0.0], [0.0, 0.0], [0.0, 10.0]])
    # open arc: from the top right, down, along the bottom, up to top left
    open_arc = np.vstack([np.linspace(rect[0], rect[1], 11)[:-1],
                          np.linspace(rect[1], rect[2], 41)[:-1],
                          np.linspace(rect[2], rect[3], 11)])
    th = np.linspace(0.0, 2 * np.pi, 400, endpoint=False)
    hole = np.stack([20.0 + 3.0 * np.cos(th), 1.0 + 3.0 * np.sin(th)], 1)
    res = _notch_one(open_arc, hole, grow=0.0)
    assert res is not None
    pre, arc, post, c, r = res
    new = np.vstack([pre, arc[1:-1], post])
    area = 0.5 * abs(np.dot(new[:, 0], np.roll(new[:, 1], -1))
                     - np.dot(new[:, 1], np.roll(new[:, 0], -1)))
    # the disc's part inside the rectangle: all of it above y = 0
    d = 1.0                                        # centre above the edge
    seg = r * r * np.arccos(-d / r) + d * np.sqrt(r * r - d * d)
    assert area == pytest.approx(400.0 - seg, rel=2e-3)
    # the channel runs INSIDE the part, round the tube's far side
    assert arc[1:-1, 1].max() == pytest.approx(1.0 + 3.0, abs=0.05)
    assert _points_in_polygon(np.array([[20.0, 4.5]]), new)[0]
    assert not _points_in_polygon(np.array([[20.0, 2.0]]), new)[0]

    # and a wedge the tube severs is filled, not printed
    import json as _json
    from washout.search.design import Mission, evaluate
    m = Mission.micro_fpv_winglet()
    d = _json.loads((ASSETS.parent / "results" / "fleet" / "fpv_micro_fpv_s4"
                     / "design.json").read_text(encoding="utf-8"))
    ev = evaluate(np.array(d["u"]), m, cst.load_selig(ASSETS / "mh45.dat"),
                  vase.PrintSettings(filament_density_gcc=0.55, spar_d_mm=m.spar_d_mm),
                  z_step_mm=2.0, size_structure=False)
    ins = ev.inserts[0]
    assert ins.glue_fill
    assert vase.check_insert(ins, ev.print_settings).ok
    item = next(i for i in ev.mass.items if i.name == f"joint fill {ins.joint}")
    assert item.mass_kg == pytest.approx(2.0 * ins.fill_g() / 1000.0)
    assert not any("insert" in r for r in ev.reasons)


# ------------------------------------- the elevon in the lattice, and the stall


@pytest.mark.parametrize("ef", [0.165, 0.25])
def test_an_elevon_in_the_lattice_converges_to_thin_aerofoil_theory(ef):
    """A deflection is one more right-hand side: the normals aft of the
    hinge turned about it. In the 2D limit dCL/ddelta over dCL/dalpha must
    approach the closed-form flap effectiveness tau = 1 - (th - sin th)/pi
    (Glauert), and approach it MONOTONICALLY as the chord is refined --
    the hinge is a log singularity in the loading, so the discrete answer
    comes from below. At nc 8, the search's lattice, it is 8-10% low; the
    minimum speed it feeds moves 0.4% from nc 8 to nc 32 (next test)."""
    from washout.geom.cst import flap_effectiveness
    tau = flap_effectiveness(ef)
    ratios = []
    for nc in (8, 16, 32):
        v = VLM(rect(1000.0, cst.naca4("0012")), ns=40, nc=nc, hinge_xc=1.0 - ef)
        dn = v.elevon_dn(0.0, ef)
        c0, ca = v.solve(0.0, 0.0).CL, v.solve(4.0, 0.0).CL
        cd = v.solve(0.0, 0.0, 4.0, dn).CL
        ratios.append((cd - c0) / (ca - c0))
    assert ratios[0] < ratios[1] < ratios[2] < tau
    assert ratios[2] == pytest.approx(tau, rel=0.03)


def test_an_elevon_deflection_is_symmetric_and_lifts_trailing_edge_down():
    af = cst.naca4("0012")
    plan = demo_bwb(af)
    v = VLM(plan, ns=24, nc=8)
    dn = v.elevon_dn(0.4, 0.2)
    p0, p1 = v.solve(3.0, 0.1), v.solve(3.0, 0.1, 5.0, dn)
    assert p1.CL > p0.CL                     # TE down adds lift
    assert p1.Cm < p0.Cm                     # ...and pitches the nose down
    n = len(v.lat.y_strip) // 2
    dcl = p1.cl_local - p0.cl_local
    assert np.allclose(dcl[:n], dcl[n:2 * n], atol=1e-10)   # port = starboard
    eta = np.abs(v.lat.y_strip) / plan.half_span_m
    assert dcl[eta > 0.45].min() > dcl[eta < 0.2].max()     # it is ON the elevon


def test_the_stall_starts_where_the_textbook_says_it_does():
    """Critical-section method against the classical stall patterns of
    untwisted wings: a rectangular wing loads its root hardest and stalls
    there first; a sharply tapered one stalls near the tip; washout on the
    same tapered wing moves the start inboard. And no wing's CL_max can
    exceed the section cl_max it is built from.

    (An elliptic wing, uniform cl in closed form, was the first choice and
    is not usable: the lattice cannot resolve a chord that goes to zero,
    and its last strip reads 25% high however the planform is sampled.
    Every design here has a finite tip chord.)"""
    from washout.aero.performance import slow_flight
    af = cst.naca4("0012")

    def tapered(taper, twist_tip):
        return planform.Planform(3.0, (
            planform.Station(0.0, 1.0, 0.0, 0.0, 0.0, af),
            planform.Station(1.0, taper, 0.25 * (1 - taper), 0.0, twist_tip, af)), "t")

    got = {}
    for name, plan in (("rect", rect(6.0, af)), ("pointed", tapered(0.2, 0.0)),
                       ("washed", tapered(0.2, -6.0))):
        v = VLM(plan, ns=40, nc=6)
        no_elevon = np.zeros_like(v.lat.normal)
        got[name] = slow_flight(v, 0.25, 6.0, no_elevon, 1.0, 1.0, 12.0, 1.0)
        assert got[name].limit == "stall"
        assert got[name].cl_max < 1.0
    assert got["rect"].eta_critical < 0.1
    assert got["pointed"].eta_critical > 0.6
    assert got["washed"].eta_critical < got["pointed"].eta_critical - 0.2


def test_the_minimum_speed_does_not_hang_on_the_chordwise_lattice():
    """On the micro_fpv winner the minimum speed moves under 1% from the
    search's nc 8 to nc 32. That is a statement about THIS wing: on the
    demo BWB it moves 3%, most of it the base lattice's own trim angle
    (1.5 deg at nc 8, 2.8 at nc 32 -- the reflex camber resolving),
    which every tier-0 number already carries."""
    from washout.aero.performance import slow_flight
    ev = _fleet_eval("micro_fpv")
    p = _fleet_physical("micro_fpv")
    got = []
    for nc in (8, 32):
        v = VLM(ev.plan, ns=32, nc=nc, fins=ev.fins)
        a = v.trim_alpha(ev.mass.x_cg_m, bounds=(-10.0, 18.0))
        dn = v.elevon_dn(p["elevon_eta"], p["elevon_chord"])
        got.append(slow_flight(v, ev.mass.x_cg_m, a, dn, p["elevon_eta"], 0.85,
                               12.0, ev.mass.total_kg).v_min_ms)
    assert got[0] == pytest.approx(got[1], rel=0.01)


def _fleet_physical(name):
    import json as _json
    from washout.search.design import unit_to_physical
    root = ASSETS.parent / "results" / "fleet"
    index = _json.loads((root / "index.json").read_text(encoding="utf-8"))
    d = _json.loads((root / index[name] / "design.json").read_text(encoding="utf-8"))
    return unit_to_physical(np.array(d["u"]))


def test_the_micro_fpv_stall_starts_behind_its_cg():
    """Found 2026-09-23, pinned so the finding cannot be argued away.

    Every micro_fpv design through gen8 passed the tip-stall gate -- its
    outer 20% works at under 88% of the peak cl -- and every one still
    starts to stall at mid-span, about 0.18 MAC BEHIND the CG, because
    the whole outer wing is swept aft of it. Lift lost there pitches the
    nose up, into a deeper stall: the swept wing's pitch-up, and the
    opposite of what a beginner's aircraft must do. The flat stall speed
    (every section at cl_max at once, nothing spent on trim) also called
    this wing 6.40 m/s; trimmed, it is 7.38."""
    ev = _fleet_eval("micro_fpv")
    assert ev.slow is not None and ev.slow.limit == "stall"
    assert 0.35 < ev.slow.eta_critical < 0.6
    assert ev.slow.v_min_ms == pytest.approx(7.38, abs=0.05)
    assert any("pitches UP at the stall" in r for r in ev.reasons)
