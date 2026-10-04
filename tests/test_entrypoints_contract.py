"""The CLI and the tunnel driver must not lie about what they did.

Two different shapes of the same failure, both found by inspection rather
than by a crash, because both fail QUIETLY:

  * `tunnel_sweep.py` wrote a header-only polar.csv and printed a success
    line when every alpha of the Re 150 000 sweep had failed.
  * `run.py --mission fpv_1m` was offered by --help and by tab completion
    and raised AttributeError, because argparse carried its own list of
    missions and the list had drifted from the factories.
"""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------- the sweep writer is strict

def _rows(alphas):
    return [{"alpha": a, "cl": 0.1 * a, "cd": 0.02, "ld": 5.0} for a in alphas]


def test_a_sweep_that_measured_nothing_writes_no_file(tmp_path):
    """The Re 150k sweep failed at every alpha and still produced a file.

    `parse_forces` is strict because a silent zero would look like a
    miraculously low-drag aerofoil and win the optimisation. The writer has
    to hold the same line."""
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import tunnel_sweep

    out = tmp_path / "polar.csv"
    with pytest.raises(SystemExit) as e:
        tunnel_sweep.write_polar(out, [], [-2.0, 0.0, 2.0])
    assert not out.exists(), "no measurements must mean no file"
    assert "3 of 3" in str(e.value)


def test_a_partial_sweep_writes_no_file_either(tmp_path):
    """A partial polar is worse than none.

    `MeasuredDrag.cd_at_cl` clamps at its endpoints rather than
    extrapolating, so a sweep that lost its high-alpha rows would quietly
    charge a separating wing the drag of the bucket."""
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import tunnel_sweep

    alphas = [0.0, 4.0, 8.0, 12.0, 16.0]
    out = tmp_path / "polar.csv"
    with pytest.raises(SystemExit) as e:
        tunnel_sweep.write_polar(out, _rows(alphas[:3]), alphas)
    assert not out.exists()
    assert "+12" in str(e.value) and "+16" in str(e.value)


def test_a_complete_sweep_is_written_sorted(tmp_path):
    import csv
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import tunnel_sweep

    alphas = [8.0, -2.0, 4.0]
    out = tunnel_sweep.write_polar(tmp_path / "polar.csv", _rows(alphas), alphas)
    got = list(csv.DictReader(out.open(encoding="utf-8")))
    assert [float(r["alpha"]) for r in got] == [-2.0, 4.0, 8.0]


# --------------------------------------- the CLI cannot name a phantom mission

def test_every_offered_mission_can_be_built():
    """argparse used to offer `fpv_1m`, which has no factory at all."""
    from washout.search.design import MISSIONS, Mission

    for name in MISSIONS:
        m = getattr(Mission, name)()
        assert m.name, f"{name} built a mission with no name"


def test_the_cli_offers_exactly_the_declared_missions():
    """One list, in design.py, beside the factories. Not two.

    Checked against the parser argparse actually builds, not against the
    source text: a source assertion trips over the comment that documents
    the fix, and would pass for a parser that had drifted again in some
    other way.

    The declared list is the fleet plus its search-space variants
    (MISSION_VARIANTS), each of which must build and must name a parent
    in the fleet."""
    import run

    from washout.search.design import MISSION_VARIANTS, MISSIONS, Mission

    choices = {a.dest: a.choices for a in run.build_parser()._actions
               if a.choices}
    assert tuple(choices["mission"]) == MISSIONS + tuple(MISSION_VARIANTS)
    for name in choices["mission"]:
        getattr(Mission, name)()          # every offered name must build
    for variant, parent in MISSION_VARIANTS.items():
        assert parent in MISSIONS, (variant, parent)
        v, p = getattr(Mission, variant)(), getattr(Mission, parent)()
        assert v.name == variant and v.objective == p.objective


def test_an_unknown_mission_is_refused_before_any_work():
    """argparse must reject it, not `getattr` 200 lines later.

    `--mission fpv_1m` used to build the print settings and only then
    raise AttributeError -- after the whole command line had been typed,
    and on a search, after the drag model had been loaded."""
    import run

    with pytest.raises(SystemExit):
        run.build_parser().parse_args(["check", "--mission", "fpv_1m"])


def test_every_mission_has_a_tracked_design_to_check():
    """`run.py check` re-scores the design this repo says is the answer.

    It replaced `seed`, which could not work: SEEDS is {} by design, so
    every mission printed 'no seed recorded' and exited 1 while the README
    advertised it as the first command to run."""
    from washout.search.design import MISSIONS

    index = json.loads((ROOT / "results" / "fleet" / "index.json")
                       .read_text(encoding="utf-8"))
    for name in MISSIONS:
        folder = index.get(name)
        assert folder, f"{name} has no entry in results/fleet/index.json"
        d = ROOT / "results" / "fleet" / folder / "design.json"
        assert d.exists(), f"{name} points at {folder}, which has no design.json"
        assert "u" in json.loads(d.read_text(encoding="utf-8"))
