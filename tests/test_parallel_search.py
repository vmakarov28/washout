"""A search evaluated in worker processes keeps the books a serial one would.

gen6 ran nine single-process searches on a 24-thread machine: 37% of it
at best, about 20% on average. `run_search(workers=N)` now evaluates each
generation in a pool of processes -- and the danger in that is the
bookkeeping, not the arithmetic: the best design, the feasible count and
the history live in the parent, and a version that updated them inside
the objective would update a worker's copy and lose it.
"""

import json
from pathlib import Path

import numpy as np

from washout.geom import cst
from washout.printing import vase
from washout.search.design import N_DIM, Mission, evaluate, physical_to_unit
from washout.search.optimize import run_search

ROOT = Path(__file__).resolve().parent.parent


def test_a_pooled_search_records_what_a_serial_one_would():
    """The initial population only (maxiter 0), so both paths evaluate the
    same designs: every one is counted, and the best the pool reports is
    the best of evaluating that population here, one by one."""
    base = cst.load_selig(ROOT / "assets" / "mh45.dat")
    m = Mission.micro_fpv()
    s = vase.PrintSettings(filament_density_gcc=0.55, spar_d_mm=m.spar_d_mm)
    seed = json.loads((ROOT / "results" / "fleet" / "fpv_micro_fpv_s4" / "design.json")
                      .read_text(encoding="utf-8"))["physical"]
    u, best, log = run_search(m, base, s, maxiter=0, popsize=1, seed=5,
                              workers=2, verbose=False, seed_physical=seed)
    n_pop = 1 * N_DIM
    assert log.evaluations == n_pop

    # the population run_search builds, rebuilt the same way
    rng = np.random.default_rng(5)
    init = rng.random((n_pop, N_DIM))
    seed_u = np.clip(physical_to_unit(seed), 0.0, 1.0)
    init[0] = seed_u
    init[1:6] = np.clip(seed_u + 0.05 * rng.standard_normal((5, N_DIM)), 0, 1)
    scores = [evaluate(x, m, base, s).score for x in init]
    assert log.best_score == max(scores)
    assert log.feasible == sum(evaluate(x, m, base, s).ok for x in init)
