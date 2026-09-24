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
import multiprocessing as mp
import os
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


# ------------------------------------------------ parallel evaluation
#
# A search used ONE core. gen6 ran nine searches on a 24-thread machine,
# which cannot exceed 37% of it, and as searches finished the average fell
# to about 20% -- while differential evolution with deferred updating is
# embarrassingly parallel across a generation. So each search can
# evaluate its population in a pool of worker processes. The bookkeeping
# -- best so far, feasible count, the history the log keeps -- stays in
# the parent, fed by the map DE calls, because a worker's copy of the log
# would be thrown away with the worker.

_WORKER: dict = {}


def lower_priority() -> None:
    """Below-normal priority: a search is background work, and the machine
    has other things on it -- a desktop, and a WSL session nothing here may
    disturb (ROADMAP.md, hard constraints)."""
    try:
        import psutil
        proc = psutil.Process()
        proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
    except Exception:                                   # noqa: BLE001
        pass


def _init_worker(mission, base, settings, ns, nc, z_step, drag) -> None:
    lower_priority()
    _WORKER.update(mission=mission, base=base, settings=settings, ns=ns,
                   nc=nc, z_step=z_step, drag=drag)


def _summary(ev: Evaluation) -> dict:
    return {"ok": bool(ev.ok), "score": float(ev.score), "ld": float(ev.ld),
            "v": float(ev.v_cruise), "mass_g": float(ev.mass_kg * 1000.0),
            "sm": float(ev.static_margin), "line": ev.line()}


def _evaluate_in_worker(u: np.ndarray):
    """-> (summary, None), or (None, why) when the geometry refused."""
    w = _WORKER
    try:
        ev = evaluate(u, w["mission"], w["base"], w["settings"], ns=w["ns"],
                      nc=w["nc"], z_step_mm=w["z_step"], drag=w["drag"])
    except Exception as e:                              # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"
    return _summary(ev), None


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

    def checkpoint(u: np.ndarray, score: float) -> None:
        """The best feasible design so far, on disk, every time it improves.

        It lived only in memory. gen10's eight searches died together 2.2 h
        in when the machine's WSL had to be restarted, and the best design
        any of them had found -- 7.88 m/s against the seed's 8.04 -- died
        with them. The file has the keys a seed is read by (`physical`), so
        a killed search can be resumed with --seed-design; the write is
        atomic, so a kill mid-write leaves the previous one."""
        if not out_dir:
            return
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "best_so_far.json.tmp"
        tmp.write_text(json.dumps({
            "u": list(map(float, u)), "physical": unit_to_physical(u),
            "score": float(score), "feasible": True,
            "evaluations": log.evaluations,
            "elapsed_s": time.perf_counter() - t0,
            "note": "a search in progress; design.json is written at the end",
        }, indent=2), encoding="utf-8")
        os.replace(tmp, d / "best_so_far.json")

    def record(u: np.ndarray, summary: dict | None, why: str | None) -> float:
        """The parent's bookkeeping for one evaluated design -> DE's value."""
        log.evaluations += 1
        if summary is None:
            # A design whose GEOMETRY cannot be built is infeasible, not
            # fatal. Four of gen6's nine searches died on one candidate
            # each, hours in, because two openings wanted the same chord
            # and the skin builder refused the layer -- scipy then saw a
            # non-number and stopped. The gates that should have caught
            # it are fixed, and this is the net under them: the cause is
            # logged, the design is rejected, and the run continues.
            log.geometry_failures += 1
            if verbose and log.geometry_failures <= 5:
                print(f" !![{log.evaluations:6d}] geometry refused: {why}")
            return 1e7
        log.feasible += int(summary["ok"])
        if summary["ok"] and summary["score"] > log.best_feasible_score:
            log.best_feasible_score = summary["score"]
            log.best_feasible_u = list(map(float, u))
            checkpoint(u, summary["score"])
        if summary["score"] > log.best_score:
            log.best_score = summary["score"]
            log.best_u = list(map(float, u))
            log.history.append({
                "eval": log.evaluations, "score": summary["score"],
                "ld": summary["ld"], "v": summary["v"],
                "mass_g": summary["mass_g"], "sm": summary["sm"],
                "ok": summary["ok"], "t_s": time.perf_counter() - t0,
            })
            if verbose:
                tag = "  " if summary["ok"] else " ~"
                print(f"{tag}[{log.evaluations:6d}] score {summary['score']:7.3f}"
                      f"{summary['line']}", flush=True)
        return -summary["score"]

    def objective(u: np.ndarray) -> float:
        try:
            ev = evaluate(u, mission, base, settings, ns=ns, nc=nc,
                          z_step_mm=search_z_step_mm, drag=drag)
        except Exception as e:                       # noqa: BLE001
            return record(u, None, f"{type(e).__name__}: {e}")
        return record(u, _summary(ev), None)

    pool = None
    de_workers = 1
    if workers > 1:
        # spawn, not fork: the same on Windows and Linux, and a worker that
        # starts clean cannot inherit a half-built cache from the parent
        pool = mp.get_context("spawn").Pool(
            workers, initializer=_init_worker,
            initargs=(mission, base, settings, ns, nc, search_z_step_mm, drag))

        def de_workers(func, population):
            # DE hands over its own wrapper of `objective`; the pool runs the
            # evaluation and THIS process keeps the log, in order
            xs = [np.asarray(x, dtype=float) for x in population]
            out = pool.map(_evaluate_in_worker, xs, chunksize=1)
            return [record(x, summ, why) for x, (summ, why) in zip(xs, out)]

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

    try:
        res = differential_evolution(
            objective, bounds=[(0.0, 1.0)] * N_DIM, init=init, maxiter=maxiter,
            tol=0.01, mutation=(0.4, 1.0), recombination=0.85, seed=seed,
            polish=False, workers=de_workers,
            updating="deferred" if workers != 1 else "immediate",
        )
    finally:
        if pool is not None:
            pool.close()
            pool.join()

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
            "workers": workers,
            "best_feasible_score": log.best_feasible_score,
            "elapsed_s": time.perf_counter() - t0,
            "history": log.history,
        }, indent=2), encoding="utf-8")
    return best_u, best, log
