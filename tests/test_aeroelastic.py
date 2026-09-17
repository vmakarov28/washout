"""Does the wing twist itself apart, and do the elevons reverse?

Nothing in this program could see either until now: a grep for flutter,
divergence, torsion or GJ returned one docstring about the spiral mode.
demon1 is scored at 44 m/s on a single-wall foamed PLA shell with one
8 mm tube, and the speed objective pushes directly toward the failure
that had no gate.

The answer, once asked: demon1's elevons reverse at 43 m/s and the score
was computed at 47. The optimizer was maximising a speed at which the
controls work backwards.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from washout import aeroelastic as ael
from washout.aero.vlm import VLM
from washout.geom import cst
from washout.printing import vase
from washout.search.design import MISSIONS, Mission, build, evaluate

ASSETS = Path(__file__).resolve().parent.parent / "assets"
RESULTS = Path(__file__).resolve().parent.parent / "results" / "fleet"


def _built(name):
    index = json.loads((RESULTS / "index.json").read_text(encoding="utf-8"))
    d = json.loads((RESULTS / index[name] / "design.json").read_text(encoding="utf-8"))
    base = cst.load_selig(ASSETS / "mh45.dat")
    mission = getattr(Mission, name)()
    ev = evaluate(np.array(d["u"]), mission, base,
                  vase.PrintSettings(filament_density_gcc=0.55, spar_d_mm=8.0,
                                     bed_z_mm=250.0))
    return ev, mission


# ---------------------------------------------------- the closed-form parts

def test_bredt_batho_against_a_circular_tube():
    """A closed thin-walled circle has the exact answer J = 2 pi r^3 t,
    and Bredt-Batho must reproduce it. This is the reference check the
    formula is entitled to -- no constant is tuned to reach it."""
    r, tw = 20.0, 0.45
    area = np.pi * r * r
    per = 2.0 * np.pi * r
    gj = ael.gj_closed_nmm2(area, per, tw, g_mpa=1.0)
    exact = 2.0 * np.pi * r ** 3 * tw
    assert gj == pytest.approx(exact, rel=1e-9)


def test_an_open_section_is_orders_below_a_closed_one():
    """The whole reason cutting a hatch in the upper skin is a structural
    decision and not a cosmetic one."""
    r, tw = 20.0, 0.45
    closed = ael.gj_closed_nmm2(np.pi * r * r, 2 * np.pi * r, tw)
    openn = ael.gj_open_nmm2(2 * np.pi * r, tw)
    assert closed / openn > 100.0, f"ratio only {closed/openn:.1f}"


def test_series_compliance_is_below_the_root_value():
    """GJ falls outboard, and torsional compliance adds in SERIES, so the
    equivalent uniform stiffness is the harmonic mean. Using the stiff
    root cell everywhere would treat the whole wing as if it were as
    thick as its thickest station and hand the optimizer a speed it does
    not have."""
    vals = np.array([10.0, 5.0, 2.0])
    assert ael._harmonic(vals) < vals.mean()
    assert ael._harmonic(vals) < vals[0]
    assert ael._harmonic(np.array([4.0, 4.0])) == pytest.approx(4.0)
    assert ael._harmonic(np.array([1.0, 0.0])) == 0.0


def test_torsional_stiffness_scales_the_way_bredt_batho_says():
    """J = 4 A^2 t / s, so GJ goes as the enclosed area SQUARED and
    linearly in the wall. Checked as a scaling law, because a scaling law
    is what the closed form asserts and a single number would not
    distinguish it from a coincidence."""
    gj1 = ael.gj_closed_nmm2(3000.0, 400.0, 0.45)
    assert ael.gj_closed_nmm2(6000.0, 400.0, 0.45) / gj1 == pytest.approx(4.0)
    assert ael.gj_closed_nmm2(3000.0, 400.0, 0.90) / gj1 == pytest.approx(2.0)
    assert ael.gj_closed_nmm2(3000.0, 800.0, 0.45) / gj1 == pytest.approx(0.5)


# ----------------------------------------------------- the model's own claim

def test_reversal_does_not_depend_on_the_elastic_axis():
    """The derivation's nicest result, and worth pinning because it is
    what makes reversal the firmer of the two numbers.

        q_div = K / (c^2 e a)
        q_rev = - cl_d K / (c^2 a cm_d)

    The e*cl_d terms cancel when d(cl_total)/d(delta) is set to zero, so
    there is no `e` in the reversal result at all. Divergence inherits the
    elastic-axis estimate; reversal rests only on GJ, the span, the chord,
    the lift slope and the elevon's own geometry, all of which are
    computed. The test evaluates both expressions at three different `e`
    and asserts exactly that asymmetry.
    """
    k, c_mm, a = 5.0e5, 160.0, 4.0
    cl_d, cm_d = 1.2, -0.30

    def q_div(e):
        return k / (c_mm * c_mm * e * a)

    def q_rev(e):
        return -cl_d * k / (c_mm * c_mm * a * cm_d)

    es = (0.05, 0.15, 0.30)
    divs = [q_div(e) for e in es]
    revs = [q_rev(e) for e in es]
    assert len(set(round(r, 9) for r in revs)) == 1, (
        f"reversal moved with the elastic axis: {revs}")
    assert divs[0] > divs[1] > divs[2], (
        f"divergence must fall as the axis moves aft: {divs}")


def test_reversal_binds_before_divergence_on_this_airframe():
    """A flying wing's elevons are most of its trailing edge, so the
    surface that controls it is also the surface that twists it. On all
    three aircraft reversal is the lower speed, by about a factor of
    three -- which is not what you would guess from the fact that
    divergence is the failure everyone names."""
    for name in MISSIONS:
        ev, mission = _built(name)
        r = ev.aeroelastic
        assert r is not None, f"{name} has no aeroelastic result"
        assert r.v_rev_ms < r.v_div_ms, (
            f"{name}: reversal {r.v_rev_ms:.0f}, divergence {r.v_div_ms:.0f}")


def test_demon1_is_scored_above_its_own_reversal_speed():
    """The headline finding, pinned.

    demon1's objective is outright velocity and the score holds down
    elevon to reach it. Its elevons reverse at about 43 m/s and the score
    is computed at 47. The optimizer was maximising a speed at which the
    controls work backwards, because nothing could see it.

    The number is a LOWER bound -- Bredt-Batho on one closed cell ignores
    the extra cells the rib truss makes -- so the real margin is better
    than 0.91x. It is not better than 1.5x, which is what the mission
    asks for, and it is the gate that has to hold."""
    ev, mission = _built("demon1")
    r = ev.aeroelastic
    assert mission.min_aeroelastic_margin > 0.0, "demon1 must check this"
    assert r.margin < 1.0, (
        f"reversal {r.v_rev_ms:.0f} m/s vs design {r.design_v_ms:.0f} m/s "
        f"= {r.margin:.2f}x; if this now passes, say why in the commit")
    assert not ev.ok
    assert any("reversal" in x for x in ev.reasons), ev.reasons


def test_the_speed_the_margin_is_measured_against_is_the_scored_speed():
    """For a racer that is TOP speed, not hands-off trim speed. Measuring
    a racer's flutter margin against the speed it settles at with the
    sticks centred would miss the whole point of the objective."""
    ev, mission = _built("demon1")
    assert mission.objective == "speed"
    assert ev.aeroelastic.design_v_ms > ev.trim.v_ms + 5.0, (
        f"{ev.aeroelastic.design_v_ms:.1f} vs trim {ev.trim.v_ms:.1f}")


def test_the_open_section_penalty_is_quantified():
    """What a hatch in the upper skin would cost, before anyone cuts one.

    This is the number Phase 3 of the build roadmap is waiting on: the
    bay and the flutter gate are the same question."""
    for name in MISSIONS:
        ev, mission = _built(name)
        r = ev.aeroelastic
        assert 0.0 < r.gj_open_nmm2 < r.gj_nmm2
        assert 0.0 < r.v_div_open_ms < r.v_div_ms, (
            f"{name}: open {r.v_div_open_ms:.0f} vs closed {r.v_div_ms:.0f}")


def test_the_model_says_what_it_cannot_do():
    """Three caveats, and two of them point the opposite way from the
    third. A number whose error direction is unknown is not usable."""
    ev, _ = _built("trainer_v3")
    notes = " ".join(ev.aeroelastic.notes).lower()
    assert "static" in notes and "flutter" in notes
    assert "lower bound" in notes
    assert "elastic axis" in notes
