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
    assert bare.structure is None and bare.print_settings is None
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
