"""The search: differential evolution over the design vector.

Why DE and not a gradient method: the feasible set is carved out by hard
geometric gates (bed size, overhang, spar bore) whose boundaries are not
differentiable, and the score is a penalised objective with plateaus. DE
does not care -- it only ranks. It is also embarrassingly parallel and
restartable, which matters when the intended usage is "leave it running
overnight".

The seed matters more than the algorithm. MH45 on a sane BWB planform is
already a flyable aeroplane, so it is injected into the initial
population: the search starts from something that flies and has to earn
every improvement, instead of spending its first thousand evaluations
discovering that wings need camber.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

from ..geom.cst import Airfoil
from ..printing import vase
from .design import (BOUNDS, LATTICE_NC, LATTICE_NS, N_DIM, Evaluation,
                     Mission, evaluate, physical_to_unit, unit_to_physical)

# Seeds are OPTIONAL now. Earlier versions leaned on a hand-found seed
# because the feasible set was hard to reach, and every geometry change
# then invalidated it -- four times. With the 4-station planform and the
# feasibility-dominant score, differential evolution finds feasibility on
# its own from a random population; a seed only speeds that up. An empty
# dict means "start from scratch", which is one less thing to keep true.
SEEDS: dict[str, dict] = {}
SEED_PHYSICAL: dict = {}
TRAINER_SEED: dict = {}


@dataclass
class SearchLog:
    geometry_failures: int = 0
    """Designs whose geometry could not be built at all. Not the same as
    infeasible: these never reached a gate, and a run with many of them
    is a run whose geometry has a hole in it."""
    best_score: float = -np.inf
    best_u: list | None = None
    # A high-scoring INFEASIBLE design is still infeasible. The penalised
    # score exists to give the optimizer a slope back into the feasible
    # set, not to define the answer, so the best feasible design is
    # tracked separately and is what gets returned and exported.
    best_feasible_score: float = -np.inf
    best_feasible_u: list | None = None
    evaluations: int = 0
    feasible: int = 0
    history: list = None

    def __post_init__(self):
        if self.history is None:
            self.history = []


def run_search(
    mission: Mission,
    base: Airfoil,
    settings: vase.PrintSettings,
    maxiter: int = 40,
    popsize: int = 12,
    seed: int = 0,
    search_z_step_mm: float | None = None,
    ns: int = LATTICE_NS,
    nc: int = LATTICE_NC,
    workers: int = 1,
    drag=None,
    out_dir: Path | None = None,
    verbose: bool = True,
    seed_physical: dict | None = None,
) -> tuple[np.ndarray, Evaluation, SearchLog]:
    """Returns (best design vector, its full evaluation, the log)."""
    log = SearchLog()
    t0 = time.perf_counter()

    def objective(u: np.ndarray) -> float:
        try:
            ev = evaluate(u, mission, base, settings, ns=ns, nc=nc,
                          z_step_mm=search_z_step_mm, drag=drag)
        except Exception as e:                       # noqa: BLE001
            # A design whose GEOMETRY cannot be built is infeasible, not
            # fatal. Four of gen6's nine searches died on one candidate
            # each, hours in, because two openings wanted the same chord
            # and the skin builder refused the layer -- scipy then saw a
            # non-number and stopped. The gates that should have caught
            # it are fixed, and this is the net under them: the cause is
            # logged, the design is rejected, and the run continues.
            log.evaluations += 1
            log.geometry_failures += 1
            if verbose and log.geometry_failures <= 5:
                print(f" !![{log.evaluations:6d}] geometry refused: "
                      f"{type(e).__name__}: {e}")
            return 1e7
        log.evaluations += 1
        log.feasible += int(ev.ok)
        if ev.ok and ev.score > log.best_feasible_score:
            log.best_feasible_score = float(ev.score)
            log.best_feasible_u = list(map(float, u))
        if ev.score > log.best_score:
            log.best_score = float(ev.score)
            log.best_u = list(map(float, u))
            log.history.append({
                "eval": log.evaluations, "score": ev.score, "ld": ev.ld,
                "v": ev.v_cruise, "mass_g": ev.mass_kg * 1000,
                "sm": ev.static_margin, "ok": ev.ok,
                "t_s": time.perf_counter() - t0,
            })
            if verbose:
                tag = "  " if ev.ok else " ~"
                print(f"{tag}[{log.evaluations:6d}] score {ev.score:7.3f}"
                      f"{ev.line()}", flush=True)
        return -ev.score

    rng = np.random.default_rng(seed)
    n_pop = popsize * N_DIM
    init = rng.random((n_pop, N_DIM))
    if seed_physical:
        seed_u = np.clip(physical_to_unit(seed_physical), 0.0, 1.0)
        init[0] = seed_u
        init[1:6] = np.clip(seed_u + 0.05 * rng.standard_normal((5, N_DIM)), 0, 1)

    if verbose:
        print(f"searching {N_DIM} dimensions, population {n_pop}, "
              f"maxiter {maxiter}, objective '{mission.objective}'"
              + (", seeded" if seed_physical else ", from random") + "\n")

    res = differential_evolution(
        objective, bounds=[(0.0, 1.0)] * N_DIM, init=init, maxiter=maxiter,
        tol=0.01, mutation=(0.4, 1.0), recombination=0.85, seed=seed,
        polish=False, workers=workers, updating="deferred" if workers != 1 else "immediate",
    )

    chosen = (log.best_feasible_u if log.best_feasible_u is not None
              else log.best_u if log.best_u is not None else list(res.x))
    best_u = np.array(chosen)
    # same lattice as the search: the final verdict must be the same
    # calculation, only the print sampling gets refined to the real layer
    best = evaluate(best_u, mission, base, settings, ns=ns, nc=nc,
                    want_panels=True, drag=drag, z_step_mm=search_z_step_mm)
    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "search_log.json").write_text(json.dumps({
            "best_u": list(map(float, best_u)),
            "best_physical": unit_to_physical(best_u),
            "evaluations": log.evaluations, "feasible": log.feasible,
            "best_feasible_score": log.best_feasible_score,
            "elapsed_s": time.perf_counter() - t0,
            "history": log.history,
        }, indent=2), encoding="utf-8")
    return best_u, best, log
