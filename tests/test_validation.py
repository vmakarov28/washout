"""The gates. Every one of these caught a real bug during the build.

Run with `python -m pytest -q`. These are not unit tests of internals --
they are the physics and geometry claims the pipeline makes, written down
so a refactor cannot quietly break them. Where a number came from
published data or closed-form theory, the source is named in the test.
"""

from __future__ import annotations

import numpy as np
import pytest

from loft.geom import cst, planform
from loft.printing import stl, vase
from loft.aero.vlm import VLM

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
    from loft.geom import cst as _cst
    from loft.search.design import (LATTICE_NC, LATTICE_NS, Mission,
                                          evaluate, physical_to_unit)
    from loft.search.optimize import SEED_PHYSICAL

    base = _cst.load_selig(ASSETS / "mh45.dat")
    m = Mission.trainer_v3()
    s = vase.PrintSettings(filament_density_gcc=0.55)
    # An explicit design, not whatever seed happens to be recorded: this
    # test is about the LATTICE, so its geometry must not move when the
    # optimizer's starting point does.
    u = physical_to_unit(dict(
        span_m=0.90, root_chord=0.30, body_eta=0.18, body_chord_frac=0.90,
        kink_gap=0.35, kink_chord_frac=0.60, tip_chord_frac=0.35,
        sweep_body=42.0, sweep_mid=30.0, sweep_outer=18.0, dihedral=5.0,
        twist_body=0.0, twist_kink=-1.0, twist_tip=-4.0,
        body_thickness=1.80, batt_x=0.20, reflex_deg=3.5, camber_scale=1.10))
    here = evaluate(u, m, base, s, z_step_mm=2.0)
    finer = evaluate(u, m, base, s, ns=LATTICE_NS + 10, nc=LATTICE_NC + 2,
                     z_step_mm=2.0)
    assert here.v_cruise == pytest.approx(finer.v_cruise, rel=0.05)
    assert here.cl_trim == pytest.approx(finer.cl_trim, rel=0.08)


def test_search_and_verdict_use_the_same_lattice():
    """The mismatch itself, pinned. run_search's final evaluation must be
    at the resolution it searched at, not a 'nicer' one."""
    import inspect

    from loft.search import optimize
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
    from loft.geom import cst as _cst
    from loft.search.design import Bay, bay_fits

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
    from loft.geom import cst as _cst
    from loft.search.design import Bay, bay_fits

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
    from loft.geom import cst as _cst
    from loft.search.design import Mission, evaluate, physical_to_unit
    from loft.search.optimize import SEEDS

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
    from loft.printing.ribs import min_clearance_mm
    for nr in (2, 3, 4):
        _, st = _ribbed(nr)
        worst = min(min_clearance_mm(c, skip=8) for c in st.contours[::37])
        assert worst >= st.settings.min_wall_mm, f"{nr} ribs: {worst:.3f} mm"


def test_rib_point_count_is_constant_across_layers():
    """The STL skinner joins layer k index i to layer k+1 index i, so a
    layer that gained or lost a vertex because a rib happened to land on
    one would shear the whole mesh. Ribs are inserted into gaps between
    existing vertices, never on top of them."""
    from loft.printing.ribs import POINTS_PER_RIB
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
    from loft import structure
    from loft.aero.vlm import VLM

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
