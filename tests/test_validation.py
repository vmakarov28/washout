"""The gates. Every one of these caught a real bug during the build.

Run with `python -m pytest -q`. These are not unit tests of internals --
they are the physics and geometry claims the pipeline makes, written down
so a refactor cannot quietly break them. Where a number came from
published data or closed-form theory, the source is named in the test.
"""

from __future__ import annotations

import numpy as np
import pytest

from planeforge.geom import cst, planform
from planeforge.printing import stl, vase
from planeforge.aero.vlm import VLM

ASSETS = __import__("pathlib").Path(__file__).resolve().parents[1] / "assets"


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
    p = planform.bwb(0.5, 0.30, 0.30, 0.62, 0.28, 38.0, 24.0, 2.0,
                     -4.0, -1.0, af, af, 1.5, "t")
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
