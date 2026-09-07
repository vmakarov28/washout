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

SEED_PHYSICAL = {
    # A seed that FLIES at the working lattice, verified by a test. The
    # previous one did not: it trimmed at 16x4 and, at the converged
    # 32x8, its Cm never crossed zero anywhere in [-6, 12] deg. A seed
    # that cannot trim teaches the first generations nothing, and the
    # docstring below claimed otherwise for three runs.
    "root_chord": 0.30, "kink_eta": 0.30, "kink_chord_frac": 0.62,
    "tip_chord_frac": 0.28, "sweep_le": 38.0, "kink_sweep_le": 24.0,
    "dihedral": 2.0, "twist_kink": -1.0, "twist_tip": -3.0,
    "body_thickness": 1.85, "batt_x": 0.30,
    "reflex_deg": 2.5, "camber_scale": 1.0,
}


@dataclass
class SearchLog:
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
    search_z_step_mm: float = 1.0,
    ns: int = LATTICE_NS,
    nc: int = LATTICE_NC,
    workers: int = 1,
    out_dir: Path | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, Evaluation, SearchLog]:
    """Returns (best design vector, its full evaluation, the log)."""
    log = SearchLog()
    t0 = time.perf_counter()

    def objective(u: np.ndarray) -> float:
        ev = evaluate(u, mission, base, settings, ns=ns, nc=nc,
                      z_step_mm=search_z_step_mm)
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

    seed_u = np.clip(physical_to_unit(SEED_PHYSICAL), 0.0, 1.0)
    rng = np.random.default_rng(seed)
    n_pop = popsize * N_DIM
    init = rng.random((n_pop, N_DIM))
    init[0] = seed_u
    # a tight cloud around the seed keeps early generations near a wing
    init[1:6] = np.clip(seed_u + 0.05 * rng.standard_normal((5, N_DIM)), 0, 1)

    if verbose:
        seed_ev = evaluate(seed_u, mission, base, settings, ns=ns, nc=nc,
                           z_step_mm=search_z_step_mm)
        print("seed design:", seed_ev.line().strip())
        if seed_ev.reasons:
            print("  seed issues:", "; ".join(seed_ev.reasons))
        print(f"searching {N_DIM} dimensions, population {n_pop}, "
              f"maxiter {maxiter}\n")

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
                    want_panels=True)
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
