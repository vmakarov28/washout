"""The search must score the aircraft it exports: spar and ribs included.

Four generations of searches judged a bare vase shell, and only
`run.py export` sized the spar and the buckling-driven ribs. The ribs are
8-22 g, mostly aft of the CG. On the gen4 trainer they moved trim past its
8 degree limit and cut Dutch-roll damping from +0.083 to +0.058, so the
search's "feasible" winner failed once built. Each test here pins one part
of the fix.
"""

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from washout.geom import cst
from washout.printing import vase
from washout.search.design import (BOUNDS, N_DIM, Mission, _evaluate_once,
                                evaluate)

ASSETS = Path(__file__).resolve().parent.parent / "assets"


def _trimmed(mission, seed=4, tries=80):
    """A random design that reaches trim as a bare shell."""
    base = cst.load_selig(ASSETS / "mh45.dat")
    settings = vase.PrintSettings(filament_density_gcc=0.55)
    rng = np.random.default_rng(seed)
    for _ in range(tries):
        u = rng.random(N_DIM)
        bare = evaluate(u, mission, base, settings, size_structure=False)
        if bare.trim is not None:
            return u, bare, base, settings
    raise AssertionError(f"no trimmed design in {tries} draws")


def test_evaluate_scores_the_ribbed_aircraft_by_default():
    m = Mission.micro()
    u, bare, base, settings = _trimmed(m)
    built = evaluate(u, m, base, settings)
    # print_settings is now always populated -- it means "the settings this
    # verdict was reached with", which the exporter needs. What tells a
    # bare pass from a built one is the structure and the ribs.
    assert bare.structure is None and not bare.print_settings.ribs
    assert built.structure is not None and built.print_settings is not None
    assert built.print_settings.ribs
    assert built.print_settings.spar_d_mm >= m.spar_d_mm
    assert built.mass is not None and bare.mass is not None
    assert built.mass.shell_kg > bare.mass.shell_kg, "ribs must add shell mass"


def test_the_search_verdict_is_the_export_verdict():
    """Re-judging a design with the settings its own verdict was reached
    with must change nothing -- that is what 'export what was scored'
    means."""
    m = Mission.micro()
    u, _, base, settings = _trimmed(m)
    built = evaluate(u, m, base, settings)
    again = _evaluate_once(u, m, base, built.print_settings)
    assert again.mass_kg == pytest.approx(built.mass_kg, rel=1e-12)
    assert again.ok == built.ok or not built.structure.ok
    if (again.dynamics is not None and again.dynamics.zeta_dr is not None
            and built.dynamics is not None):
        assert again.dynamics.zeta_dr == pytest.approx(built.dynamics.zeta_dr,
                                                       rel=1e-9)


def test_run_py_exports_with_the_sized_settings():
    """Both run.py paths must hand the sized settings to the exporter. The
    search path used to pass the bare-shell settings straight through."""
    import run

    src = inspect.getsource(run.main)
    export_branch = src[src.index('a.command == "export"'):src.index("run_search(")]
    search_branch = src[src.index("run_search("):]
    assert "size_structure" in export_branch
    assert "print_settings" in export_branch
    assert "best.print_settings" in search_branch


def test_seeding_lifts_a_design_from_before_the_fin_variables(tmp_path):
    import run

    phys = {b.name: 0.5 * (b.lo + b.hi) for b in BOUNDS
            if not b.name.startswith("fin_")}
    path = tmp_path / "design.json"
    path.write_text(json.dumps({"physical": phys}))
    seed = run.load_seed_physical(path)
    assert set(seed) == {b.name for b in BOUNDS}
    assert seed["fin_area_frac"] == 0.0, "a lifted design must have no fins"

    phys.pop("washout")
    path.write_text(json.dumps({"physical": phys}))
    with pytest.raises(SystemExit):
        run.load_seed_physical(path)


def test_a_built_aircraft_evaluation_builds_one_lattice(monkeypatch):
    """Ribs change mass, not geometry, so the bare pass, the structure
    sizing and the ribbed pass must share one lattice. Each built its own
    before, and a built-aircraft evaluation cost 2.3-2.8x a bare one."""
    from washout.search import design

    m = Mission.micro()
    u, _, base, settings = _trimmed(m)
    real = design.VLM
    built = []

    def counting(*args, **kwargs):
        built.append(1)
        return real(*args, **kwargs)

    design._VLM_CACHE.clear()
    monkeypatch.setattr(design, "VLM", counting)
    ev = evaluate(u, m, base, settings)
    assert ev.structure is not None, "the design must reach structure sizing"
    assert len(built) == 1, f"lattice built {len(built)} times for one design"


def test_the_lattice_cache_does_not_change_the_verdict():
    from washout.search import design

    m = Mission.micro()
    u, _, base, settings = _trimmed(m)
    design._VLM_CACHE.clear()
    cold = evaluate(u, m, base, settings)
    warm = evaluate(u, m, base, settings)
    assert warm.score == pytest.approx(cold.score, rel=1e-12)
    assert warm.mass_kg == pytest.approx(cold.mass_kg, rel=1e-12)
    if cold.dynamics is not None and cold.dynamics.zeta_dr is not None:
        assert warm.dynamics.zeta_dr == pytest.approx(cold.dynamics.zeta_dr,
                                                      rel=1e-12)
    # and a different design must not be served the first one's lattice
    other = np.clip(u + 0.07, 0.0, 1.0)
    design._VLM_CACHE.clear()
    fresh = evaluate(other, m, base, settings)
    evaluate(u, m, base, settings)
    again = evaluate(other, m, base, settings)
    assert again.score == pytest.approx(fresh.score, rel=1e-12)


# --------------------------------------------------- the spar has a mass

def test_the_budget_charges_the_tubes_that_were_fitted():
    """A flat spar allowance is not a spar.

    `Mission._common` charged 30 g for "spar + joiners" on the trainer
    while `structure.select` sized a real 8x6 tube and `spars.fit_all`
    fitted TWO 8 mm corridors -- an LE spar reaching the tip and a TE spar
    reaching eta 0.76. The second tube was never weighed at all. The
    budgeted spar mass must now equal the tubes actually fitted, each
    priced over twice its own measured reach, because one tube runs tip to
    tip through the centre body."""
    from washout import spars as sp
    from washout import structure as struct

    m = Mission.trainer_v3()
    u, _, base, settings = _trimmed(m)
    ev = evaluate(u, m, base, settings)
    assert ev.structure is not None, "the design must reach structure sizing"

    budgeted = sum(i.mass_kg for i in ev.mass.items if i.name.startswith("spar "))
    tube = struct.tube_for_od(ev.structure.spar.od_mm)
    expected = sum(tube.mass_g(2.0 * f.reach_mm) * 1e-3 for f in ev.spar_fits)

    assert len(ev.spar_fits) == len(m.spars) == 2, "trainer declares two corridors"
    assert budgeted == pytest.approx(expected, rel=1e-12)
    assert not any(i.name == "spar + joiners" for i in ev.mass.items), (
        "the flat allowance must be gone, not merely supplemented")


def test_each_spar_is_weighed_at_the_station_it_was_fitted_to():
    """A spanwise tube's mass sits at its own chord station.

    The flat item sat at 0.30c whatever the fit solved for. On the trainer
    the LE and TE corridors come out 0.21c and 0.54c -- 80 mm apart on a
    244 mm root chord -- and that is a real CG difference, not a rounding
    one."""
    m = Mission.trainer_v3()
    u, _, base, settings = _trimmed(m)
    ev = evaluate(u, m, base, settings)
    root_c = ev.plan.stations[0].chord_m

    by_name = {i.name: i for i in ev.mass.items}
    for f in ev.spar_fits:
        item = by_name[f"spar {f.spec.name}"]
        assert item.x_m == pytest.approx(f.x_frac * root_c, rel=1e-12)


def test_the_gen5_trainer_does_not_survive_its_own_spar():
    """The headline design fails a gate once its tubes are weighed.

    gen5's trainer was logged at 331 g and 25.7 g/dm2, inside its
    26 g/dm2 wing-loading gate. Weighing the two 8 mm tubes it actually
    carries put it at ~355 g, outside it. It is ~361 g now: the skin is
    no longer cut open for the payload bays, so the six grams those
    openings used to remove are back. The aircraft got heavier; the
    estimate did not drift.

    This test exists so the mass can never quietly drift back down. The
    number it pins is the one a builder would put on a scale."""
    root = Path(__file__).resolve().parent.parent
    d = json.loads((root / "results" / "fleet" / "gen5_trainer_v3_v101"
                    / "design.json").read_text())
    base = cst.load_selig(ASSETS / "mh45.dat")
    m = Mission.trainer_v3()
    settings = vase.PrintSettings(filament_density_gcc=0.55, spar_d_mm=8.0,
                                  bed_z_mm=250.0)
    ev = evaluate(np.array(d["u"]), m, base, settings)

    assert ev.mass_kg * 1000 == pytest.approx(361.0, abs=4.0), (
        f"{ev.mass_kg*1000:.0f} g; logged 331, real ~361")
    loading = ev.mass_kg * 1000.0 / (ev.plan.area_m2 * 100.0)
    assert loading > m.max_wing_loading_gdm2, (
        f"wing loading {loading:.1f} should now miss "
        f"{m.max_wing_loading_gdm2:.0f} g/dm2")
    assert not ev.ok and any("wing loading" in r for r in ev.reasons)
