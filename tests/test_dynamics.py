"""Lateral dynamics and tip fins.

Each test pins either a published result the model has to reproduce, or
a mistake that was actually made while building it.
"""

from pathlib import Path

import numpy as np
import pytest

from washout.aero import dynamics as dyn
from washout.aero import fins as fn
from washout.aero.vlm import VLM
from washout.geom import cst
from washout.geom.planform import Segment, lofted
from washout.printing import stl

ASSETS = Path(__file__).resolve().parent.parent / "assets"


def _flat(chord_fn, half_span, root_chord, dihedral=0.0, n=24):
    af = cst.scale_camber(cst.load_selig(ASSETS / "mh45.dat"), 0.0)
    segs = tuple(Segment(float(e), float(chord_fn(e)), 0.0, dihedral, 0.0)
                 for e in np.linspace(0.0, 1.0, n + 1)[1:])
    return lofted(half_span, root_chord, 0.0, af, af, segs)


def test_mode_solver_reproduces_the_navion():
    """Nelson, Flight Stability and Automatic Control: the Navion lateral
    example, dimensional derivatives in ft and s. Published roots: Dutch
    roll -0.487 +/- 2.335j, roll -8.435, spiral -0.0087. The assembly and
    the mode classification are checked against a real aircraft before
    they are trusted on a flying wing nobody has flown."""
    A = dyn.state_matrix(-45.72, 0.0, 0.0, -15.97, -8.395, 2.19,
                         4.549, -0.349, -0.76, 176.0, g=32.2)
    zeta, wn, spiral, roll = dyn.classify(np.linalg.eigvals(A))
    assert zeta is not None
    assert -zeta * wn == pytest.approx(-0.487, rel=0.02)
    assert wn * np.sqrt(1.0 - zeta ** 2) == pytest.approx(2.335, rel=0.02)
    assert roll == pytest.approx(-8.435, rel=0.01)
    assert spiral == pytest.approx(-0.0087, rel=0.10)


def test_roll_damping_agrees_with_lifting_line():
    """Elliptic wing, aspect ratio 6: lifting-line theory gives
    Cl_p = -(pi/4) A/(A+4) = -0.471. The lattice lands within 15%."""
    p = _flat(lambda e: max(np.sqrt(1.0 - e * e), 0.04), 0.3, 4 * 0.6 / (6 * np.pi))
    D = VLM(p, 32, 8).lateral_derivatives(0.0, np.array([0.025, 0.0, 0.0]))
    A = p.aspect_ratio
    assert D[1, 1] < 0.0
    assert D[1, 1] == pytest.approx(-(np.pi / 4.0) * A / (A + 4.0), rel=0.15)


def test_stability_axes_are_an_exact_rotation_of_body_axes():
    """The mistake this pins: body-axis derivatives fed to stability-axis
    equations. At 7.8 degrees of trim that leaked roll stiffness into yaw
    and reported every aircraft as violently divergent."""
    p = _flat(lambda e: 1.0 - 0.5 * e, 0.4, 0.15, dihedral=6.0)
    v = VLM(p, 24, 6)
    ref = np.array([0.05, 0.0, 0.01])
    alpha = 7.5
    Db = v.lateral_derivatives(alpha, ref, axes="body")
    Ds = v.lateral_derivatives(alpha, ref, axes="stability")
    a = np.radians(alpha)
    T = np.array([[1.0, 0.0, 0.0],
                  [0.0, np.cos(a), np.sin(a)],
                  [0.0, -np.sin(a), np.cos(a)]])
    assert np.allclose(Ds, T @ Db @ T.T, atol=1e-10)


def test_profile_drag_yaw_damping_is_the_textbook_value():
    p = _flat(lambda e: 1.0, 0.4, 0.12)
    assert dyn.profile_yaw_damping(p, 0.021) == pytest.approx(-0.021 / 3.0, rel=0.01)


def test_tip_fin_plates_are_printable_and_signed_right():
    """Flat plates: watertight, the right volume, sitting on the tip chord,
    and every derivative pointing the stabilising way."""
    p = _flat(lambda e: 1.0 - 0.4 * e, 0.45, 0.2)
    for area, aspect, below in ((0.02, 1.3, 0.0), (0.05, 1.0, 0.5), (0.06, 2.0, 0.0)):
        f = fn.tip_fins(p, area, aspect, below)
        verts, tris = f.mesh_mm()
        assert stl.manifold_report(tris)["watertight"]
        want = f.area_m2 * 1e6 * f.thickness_m * 1e3
        assert stl.volume_mm3(verts, tris) == pytest.approx(want, rel=1e-6)
        assert f.root_chord_m <= p.at(1.0).chord_m + 1e-12
        D = f.derivatives(p.area_m2, p.span_m, 0.02, 0.0, 5.0)
        assert D[0, 0] < 0.0 and D[2, 0] > 0.0 and D[2, 2] < 0.0 and D[1, 1] <= 0.0
    assert fn.tip_fins(p, 0.0, 1.3, 0.0) is None


def test_a_fin_split_about_the_cg_adds_no_roll_coupling():
    """The reason to offer fins at all: a vertical plate that reaches as
    far below the CG as above it adds yaw stiffness with zero dihedral
    effect, which a canted winglet cannot do."""
    p = _flat(lambda e: 1.0 - 0.4 * e, 0.45, 0.2)
    z = p.at(1.0).z_le_m
    up = fn.tip_fins(p, 0.03, 1.3, 0.0).derivatives(p.area_m2, p.span_m, 0.02, z, 0.0)
    mid = fn.tip_fins(p, 0.03, 1.3, 0.5).derivatives(p.area_m2, p.span_m, 0.02, z, 0.0)
    assert abs(mid[1, 0]) < 1e-9 < abs(up[1, 0])
    assert mid[2, 0] == pytest.approx(up[2, 0], rel=0.25)


def _trimmed_design(mission, seed=4, tries=60, min_sm=0.0):
    """A random design that gets as far as trim, with no fins, and with at
    least `min_sm` of static margin to spare.

    The margin is for tests that ADD mass aft and need the result still to
    trim: tip fins are 3 g at the tips. It became necessary when the spars
    were weighed at the middle of the straight tubes they are rather than
    at their root seats, which moved every swept design's CG aft and left
    the first random micro that trimmed with 0.01 of margin -- which 3 g
    of fin then took away."""
    from washout.printing import vase
    from washout.search.design import N_DIM, evaluate, physical_to_unit, unit_to_physical

    base = cst.load_selig(ASSETS / "mh45.dat")
    settings = vase.PrintSettings(filament_density_gcc=0.55)
    rng = np.random.default_rng(seed)
    for _ in range(tries):
        phys = unit_to_physical(rng.random(N_DIM))
        phys["fin_area_frac"] = 0.0
        ev = evaluate(physical_to_unit(phys), mission, base, settings)
        if (ev.trim is not None and ev.lateral is not None
                and ev.static_margin >= min_sm):
            return phys, ev, base, settings
    raise AssertionError(f"no trimmed design in {tries} draws")


def test_the_search_charges_fins_and_checks_the_dutch_roll():
    """Wiring, end to end: the same design with and without fins. Fins must
    cost mass and drag, must count toward the static yaw gate, and every
    trimmed design must carry a Dutch-roll analysis -- a fin the search
    could fit for free, or one the gates could not see, would make the
    whole option meaningless."""
    from washout.search.design import Mission, evaluate, physical_to_unit

    m = Mission.micro()
    phys, bare, base, settings = _trimmed_design(m, tries=400, min_sm=0.12)
    finned = evaluate(physical_to_unit({**phys, "fin_area_frac": 0.04,
                                        "fin_aspect": 1.3, "fin_below": 0.2}),
                      m, base, settings)
    assert bare.fins is None and finned.fins is not None
    assert finned.trim is not None and finned.lateral is not None
    assert bare.dynamics is not None and finned.dynamics is not None
    assert finned.mass_kg > bare.mass_kg
    assert finned.trim.CD0 > bare.trim.CD0
    assert finned.lateral.cn_beta > bare.lateral.cn_beta


def test_every_mission_gates_on_dutch_roll_damping():
    from washout.search.design import Mission

    t = Mission.trainer_v3()
    assert t.max_roll_yaw_ratio == pytest.approx(8.5)
    assert t.min_spiral_t2_s > 0.0
    for m in (t, Mission.demon1(), Mission.micro()):
        assert m.min_dutch_roll_zeta is not None and m.min_dutch_roll_zeta >= 0.08
