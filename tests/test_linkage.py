"""The mechanism has to deliver the deflection the score spends.

`dcm_ddeg` says how much moment a degree of elevon buys, and demon1's
entire objective is top speed, which is scored by holding
`max_elevon_deflect_deg` of down elevon. Nothing checked that the servo,
its arm, the pushrod and the horn can reach that angle, so the number at
the top of the fleet table was computed from a deflection the aircraft had
never been shown able to make.

Solved as a four-bar, not as the ratio r_servo / r_horn, which is the
small-angle limit of a parallel linkage and is wrong in the two ways that
matter: it is linear, so it cannot show a mechanism running out of travel,
and it is symmetric, so it cannot show the differential a real one has.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from washout import build_sheet, linkage as lkg
from washout.geom import cst
from washout.printing import vase
from washout.search.design import (MISSIONS, Mission, build, evaluate,
                                   seat_bays, unit_to_physical)

ASSETS = Path(__file__).resolve().parent.parent / "assets"
RESULTS = Path(__file__).resolve().parent.parent / "results" / "fleet"
WALL = 0.45


def _fleet(name):
    index = json.loads((RESULTS / "index.json").read_text(encoding="utf-8"))
    d = json.loads((RESULTS / index[name] / "design.json").read_text(encoding="utf-8"))
    base = cst.load_selig(ASSETS / "mh45.dat")
    mission = getattr(Mission, name)()
    u = np.array(d["u"])
    return u, mission, base, build(u, mission, base)


def _built(name):
    u, mission, base, plan = _fleet(name)
    settings = vase.PrintSettings(filament_density_gcc=0.55, spar_d_mm=8.0,
                                  bed_z_mm=250.0)
    return evaluate(u, mission, base, settings), mission


# ------------------------------------------------------------ the four-bar

def test_the_neutral_position_is_consistent():
    """The rod's length is fixed BY the neutral position, so at zero servo
    angle the surface must be at zero deflection. Anything else is a
    linkage that trims the aircraft and has to be adjusted out."""
    L = lkg.Linkage(hinge_x_mm=100.0, hinge_z_mm=5.0, servo_x_mm=75.0,
                    servo_z_mm=5.0)
    assert lkg.deflection_at(L, 0.0) == pytest.approx(0.0, abs=1e-9)


def test_a_longer_servo_arm_gives_more_deflection():
    got = []
    for r in (6.0, 8.0, 11.0):
        d, u, locked = lkg.sweep(lkg.Linkage(100.0, 5.0, 75.0, 5.0, r, 12.0))
        assert not locked
        got.append(d)
    assert got == sorted(got), got


def test_a_longer_horn_gives_less_deflection():
    """More leverage on the surface, less angle. The trade a builder makes
    when they move the clevis out a hole."""
    got = []
    for r in (12.0, 20.0, 30.0):
        d, u, locked = lkg.sweep(lkg.Linkage(100.0, 5.0, 75.0, 5.0, 11.0, r))
        got.append(d)
    assert got == sorted(got, reverse=True), got


def test_a_linkage_can_lock_and_says_so():
    """The failure a ratio cannot express: the servo still turning and the
    surface no longer following. Reported rather than clamped, because a
    locked linkage wants a different fix from a weak one."""
    d, u, locked = lkg.sweep(lkg.Linkage(100.0, 5.0, 75.0, 5.0,
                                         servo_arm_mm=14.0, horn_arm_mm=12.0))
    assert locked


def test_a_four_bar_is_differential():
    """Up and down travel are not equal, and the asymmetry grows as the
    horn swings through a bigger arc. A ratio model says they are equal."""
    _, up_small, _ = lkg.sweep(lkg.Linkage(100.0, 5.0, 75.0, 5.0, 11.0, 12.0))
    down_s, up_s, _ = lkg.sweep(lkg.Linkage(100.0, 5.0, 75.0, 5.0, 11.0, 12.0))
    down_b, up_b, _ = lkg.sweep(lkg.Linkage(100.0, 5.0, 75.0, 5.0, 11.0, 30.0))
    assert abs(abs(down_b) - abs(up_b)) > abs(abs(down_s) - abs(up_s))


def test_the_horn_arm_comes_from_the_section_not_a_constant():
    """A control horn screws to the elevon's LOWER surface while the hinge
    is on the upper one, so its arm is the section's own thickness plus the
    protrusion -- and the section thins outboard. The mechanism's leverage
    changes along the span whether or not anyone models it."""
    _, _, _, plan = _fleet("trainer_v3")
    inner = lkg.for_station(plan, 0.50, 0.72, 0.55, 11.0, 8.0, 60.0, WALL)
    outer = lkg.for_station(plan, 0.95, 0.72, 0.55, 11.0, 8.0, 60.0, WALL)
    assert inner.horn_arm_mm > outer.horn_arm_mm + 1.0, (
        f"{inner.horn_arm_mm:.1f} vs {outer.horn_arm_mm:.1f}")
    # and the leverage difference is real, not a rounding one
    assert lkg.sweep(outer)[0] > lkg.sweep(inner)[0]


def test_every_fleet_linkage_delivers_what_the_score_spends():
    """The gate, on the designs that are actually tracked."""
    for name in MISSIONS:
        ev, mission = _built(name)
        assert ev.linkage is not None, f"{name} has no linkage"
        down, up, locked = lkg.sweep(ev.linkage)
        assert not locked, f"{name}: linkage locks"
        got = min(down, -up)
        assert got >= mission.max_elevon_deflect_deg, (
            f"{name}: reaches {down:+.1f}/{up:+.1f}, score spends "
            f"+/-{mission.max_elevon_deflect_deg:.0f}")
        assert not [r for r in ev.reasons if "linkage" in r], ev.reasons


# ------------------------------------------------------------ servo bays

def test_the_servo_sits_beside_the_surface_it_drives():
    """Not on the centreline. A payload bay belongs in the body; a servo
    belongs out in the wing, which is why a Bay has a spanwise seat at
    all."""
    for name in MISSIONS:
        u, mission, base, plan = _fleet(name)
        p = unit_to_physical(u)
        s = vase.PrintSettings(bed_z_mm=250.0, spar_d_mm=8.0,
                               elevon_chord=p["elevon_chord"],
                               elevon_eta=p["elevon_eta"])
        joints = vase.panel_etas(plan, s)
        vols, seats, etas = seat_bays(plan, mission, p, WALL, joints)
        assert etas["servos"] is not None and etas["servos"] > 0.1, (
            f"{name}: servo bay at eta {etas['servos']}")
        bay = next(b for b in mission.bays if b.name == "servos")
        assert bay.eta_lo <= etas["servos"] <= bay.eta_hi


def test_the_servo_mass_moves_to_its_solved_seat():
    """Its station is an output now, so it has to be inside the evaluation
    loop. `choose_structure` taught this at a cost of four generations:
    ribs added after the verdict were 8-22 g aft of the CG and turned
    feasible designs infeasible. A servo is 9 to 18 g."""
    for name in MISSIONS:
        ev, mission = _built(name)
        root_c = ev.plan.stations[0].chord_m
        item = next(i for i in ev.mass.items if i.name == "servos x2")
        bay = next(b for b in mission.bays if b.name == "servos")
        # not at the old hardcoded nominal, and inside the solved band
        assert bay.x_lo * root_c <= item.x_m <= bay.x_hi * root_c


def test_a_servo_bay_never_clashes_silently():
    """It may clash -- it may not clash unnoticed.

    This asserted a clear seat on EVERY mission, and that stopped being
    true once the payload bays became real openings reserved against the
    spars: demon1's centre body is 175 mm of root chord carrying a 4S
    pack, an ESC, a receiver, two servos and two 8 mm tubes, and its servo
    bay now genuinely overlaps the TE spar by 5.5 mm. That is a finding
    about demon1, not a regression, so the invariant worth pinning is the
    weaker and more useful one: a clash makes the design INFEASIBLE and
    names itself in the reasons.

    The trainer keeps the strong claim, because a clear seat demonstrably
    exists there and the solver has to find it."""
    for name in MISSIONS:
        ev, mission = _built(name)
        clashes = [r for r in ev.reasons if "servos" in r and "share" in r]
        if clashes:
            assert not ev.ok, f"{name} clashes and was called feasible"

    ev, _ = _built("trainer_v3")
    assert not [r for r in ev.reasons if "servos" in r and "share" in r], (
        f"the trainer has room and the solver must find it: {ev.reasons}")
    assert not [r for r in ev.reasons
                if "servos" in r and "print joints" in r], ev.reasons


# ------------------------------------------------------------ build sheet

def test_the_cg_window_is_the_static_margin_band():
    """SM = (x_np - x_cg) / mac, so a band on SM is exactly a band on
    x_cg. No search, and it is the number a builder actually needs: not
    'balance here' but 'here, and this is how far out you may be'."""
    ev, mission = _built("trainer_v3")
    lo, hi = build_sheet.cg_window_mm(ev)
    mac = ev.plan.mac_m * 1000.0
    x_np = ev.trim.x_np_m * 1000.0
    for x_cg, inside in ((lo + 1e-6, True), (hi - 1e-6, True),
                         (lo - 1.0, False), (hi + 1.0, False)):
        sm = (x_np - x_cg) / mac
        ok = mission.min_static_margin <= sm <= mission.max_static_margin
        assert ok == inside, f"x_cg {x_cg:.2f} -> SM {sm:.4f}"


def test_the_throws_are_reported_in_millimetres():
    """A transmitter is set up with a ruler at the trailing edge, not by
    knowing the angle."""
    ev, mission = _built("trainer_v3")
    thr = build_sheet.throw_mm(ev)
    assert thr is not None
    down_mm, up_mm = thr
    assert 1.0 < down_mm < 60.0 and 1.0 < up_mm < 60.0, thr
    # and they must be consistent with the angle at the elevon's chord
    from washout import linkage as _lk
    down_deg, up_deg, _ = _lk.sweep(ev.linkage)
    eta_mid = 0.5 * (ev.print_settings.elevon_eta + 1.0)
    arm = ev.print_settings.elevon_chord * ev.plan.at(eta_mid).chord_m * 1000.0
    assert down_mm == pytest.approx(arm * np.sin(np.radians(abs(down_deg))))


def test_the_build_sheet_never_says_one_bottom_layer():
    """The prose it replaced did, while `spars.report` said the root face
    needs ZERO -- and one bottom layer seals it, which makes the spar
    corridor a closed pocket and the electronics unreachable. Two places
    disagreeing about the single most load-bearing setting in the project,
    one of them wrong."""
    for name in MISSIONS:
        ev, mission = _built(name)
        parts = vase.build_panels(ev.plan, ev.print_settings)
        text = build_sheet.render(ev, parts, ev.print_settings)
        assert "1 bottom layer" not in text
        assert "bottom layers | **0**" in text
        assert "Zero bottom layers" in text


def test_the_build_sheet_reports_the_failures_it_has():
    """A sheet for a design that misses its mission must say so. Handing
    someone printable parts and a clean-looking sheet for an aeroplane
    that does not meet its own gates is the worst of both."""
    ev, mission = _built("trainer_v3")
    parts = vase.build_panels(ev.plan, ev.print_settings)
    text = build_sheet.render(ev, parts, ev.print_settings)
    assert ev.reasons, "this test needs a failing design"
    assert "does NOT pass its mission" in text
    for r in ev.reasons:
        assert r in text


def test_the_build_sheet_lists_every_part_and_the_cut_list():
    ev, mission = _built("trainer_v3")
    from washout.printing import elevons as elv
    ps = ev.print_settings
    parts = vase.build_panels(ev.plan, ps) + elv.build_elevons(
        ev.plan, ps, vase.panel_etas(ev.plan, ps),
        ev.max_elevon_deflect_deg + ps.hinge_margin_deg)
    text = build_sheet.render(ev, parts, ps)
    for p in parts:
        assert f"`{p.name}.stl`" in text
    assert "fuselage" in text, "the builder has to know which panel that is"
    for f in ev.spar_fits:
        assert f.spec.name in text
        assert f"{2*f.reach_mm:.0f} mm" in text, "tip to tip, not one side"


# ------------------------------------------- the output contract, and determinism

def test_the_export_is_deterministic():
    """The README's central claim: `design.json` plus the print settings
    reproduce the exact STLs. Nothing tested it, and everything in
    `results/` depends on it -- the STLs are not kept there precisely
    because they are supposed to be a pure function of the vector.

    ROADMAP item 17."""
    from washout.printing import elevons as elv
    from washout.printing import stl

    ev, mission = _built("trainer_v3")
    ps = ev.print_settings

    def build_bytes():
        parts = vase.build_panels(ev.plan, ps) + elv.build_elevons(
            ev.plan, ps, vase.panel_etas(ev.plan, ps),
            ev.max_elevon_deflect_deg + ps.hinge_margin_deg)
        return {p.name: (p.contours.tobytes(), p.z_mm.tobytes())
                for p in parts}

    a, b = build_bytes(), build_bytes()
    assert a.keys() == b.keys()
    for k in a:
        assert a[k] == b[k], f"{k} is not reproducible from the same design"


def test_a_re_parsed_design_gives_the_same_geometry(tmp_path):
    """And it has to survive the round trip through JSON, because that is
    how a design actually reaches the exporter."""
    index = json.loads((RESULTS / "index.json").read_text(encoding="utf-8"))
    raw = (RESULTS / index["trainer_v3"] / "design.json").read_text(encoding="utf-8")
    u1 = np.array(json.loads(raw)["u"])
    round_trip = tmp_path / "d.json"
    round_trip.write_text(json.dumps({"u": list(map(float, u1))}),
                          encoding="utf-8")
    u2 = np.array(json.loads(round_trip.read_text(encoding="utf-8"))["u"])
    base = cst.load_selig(ASSETS / "mh45.dat")
    m = Mission.trainer_v3()
    p1, p2 = build(u1, m, base), build(u2, m, base)
    assert p1.mac_m == pytest.approx(p2.mac_m, rel=0, abs=0)
    assert p1.area_m2 == pytest.approx(p2.area_m2, rel=0, abs=0)


def test_the_manifest_covers_every_part_with_a_profile():
    """Machine-readable, so a slicer script can read it instead of a
    human reading a paragraph. The bed rotation was already solved and
    only ever printed to a terminal."""
    ev, mission = _built("trainer_v3")
    from washout.printing import elevons as elv
    ps = ev.print_settings
    parts = vase.build_panels(ev.plan, ps) + elv.build_elevons(
        ev.plan, ps, vase.panel_etas(ev.plan, ps),
        ev.max_elevon_deflect_deg + ps.hinge_margin_deg)
    man = build_sheet.manifest(ev, parts, ps)
    files = {e["file"] for e in man["half_wing_parts"]}
    for p in parts:
        assert f"{p.name}.stl" in files
    assert man["mirror"] is True, "the STLs are one half wing"
    assert man["profiles"]["vase"]["bottom_layers"] == 0
    assert man["profiles"]["vase"]["spiralize"] is True
    for e in man["half_wing_parts"]:
        assert e["profile"] in man["profiles"] or e["profile"] == "solid"


def test_the_bom_does_not_double_count_the_pairs():
    """The mass budget's items are per-AIRCRAFT totals, so a quantity of
    2 beside an 18 g mass means eighteen grams of servo in the aeroplane,
    not thirty-six."""
    ev, mission = _built("trainer_v3")
    parts = vase.build_panels(ev.plan, ev.print_settings)
    text = build_sheet.bom(ev, parts)
    assert "mass (all of them)" in text
    assert "| servos x2 |" not in text, "the count belongs in the qty column"
    for f in ev.spar_fits:
        assert f"{2*f.reach_mm:.0f} mm" in text
    total = sum(i.mass_kg for i in ev.mass.items)
    assert total == pytest.approx(ev.mass.total_kg - ev.mass.shell_kg)
