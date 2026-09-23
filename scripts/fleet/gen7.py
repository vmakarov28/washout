#!/usr/bin/env python3
"""Generation 7: the fleet re-searched against the aircraft as it is built.

    python scripts/fleet/gen7.py launch      # start every search, detached
    python scripts/fleet/gen7.py status      # progress, and how busy the machine is
    python scripts/fleet/gen7.py pick        # each mission's winner, feasible first
    python scripts/fleet/gen7.py export      # the winners as STLs, build sheets and STEP
    python scripts/fleet/gen7.py track       # the winners into results/fleet

`status` shows each run's latest IMPROVEMENT and the evaluation it came
at, which is a lower bound on how far the run has got: a search that has
stopped improving keeps evaluating and prints nothing.

## Why a re-search

On 2026-09-22 two defects were fixed (docs/ROADMAP-CAD.md section 0): the
spars were modelled as tubes that bend with the wing, and the printed
panels as straight where the loft curves. Re-checked with both fixed, no
tracked design is feasible -- no straight tube reaches the trainer's or
demon1's outer joint, demon1's reversal margin fell to 0.70x, and two
swept tubes now weigh aft of where they were charged, taking micro_fpv
out of its static-margin band and micro out of trim altogether. Every
one of those is the aircraft changing, and a search against the gates
that see them is the only thing that closes them.

## Sized for the machine, all of it

gen6 ran nine single-process searches on a 24-thread machine: 37% at
best, and about 20% on average once searches began finishing at
different times. Each search here evaluates its population in its own
pool of worker processes (`run.py search --workers`), and the fleet is
twelve searches of three workers: 36 workers on 24 threads. The
overcommit is deliberate. A generation ends with a straggler or two while
its other workers sit idle, and searches finish at different times; with
more workers than threads, another search's work fills both gaps and the
machine stays busy to the end. Everything runs at below-normal priority,
because the machine has other work on it -- a desktop, and a WSL session
that nothing here may disturb (ROADMAP.md, hard constraints).

Layers are still sampled at the full 0.25 mm. A coarser search sampling
was measured and rejected: 1.65x faster, and it moved the overhang gate
by up to 23.5 degrees and flipped verdicts -- a search that scores a
different aircraft from the one that is built.

Each search is seeded with its mission's tracked design, the best any
generation has on the flight axes, as gen6 was.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "out" / "gen7"
POLAR = ("data/polars/trainer_mid_re60k.csv@60000,"
         "data/polars/thin_reflex_re100k.csv@100000")
MISSIONS = ("trainer_v3", "demon1", "micro", "micro_fpv")
SEEDS = (0, 1, 2)
WORKERS = 3
POPSIZE = 6
ITERS = 40

# Windows process-creation flags: a hidden console the pool's workers
# inherit (DETACHED would give each spawned worker a window of its own),
# a process group of its own, below-normal priority, and out of any job
# object this launcher runs in, so the searches outlive it.
CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_PROCESS_GROUP = 0x00000200
BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
CREATE_BREAKAWAY_FROM_JOB = 0x01000000


def tracked() -> dict:
    idx = json.loads((ROOT / "results" / "fleet" / "index.json").read_text(encoding="utf-8"))
    return {m: ROOT / "results" / "fleet" / idx[m] / "design.json" for m in MISSIONS}


def runs():
    for m in MISSIONS:
        for s in SEEDS:
            yield m, s, f"{m}_v12{s}"


def launch() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = tracked()
    record = []
    for m, s, tag in runs():
        cmd = [sys.executable, "run.py", "search", "--mission", m,
               "--iters", str(ITERS), "--popsize", str(POPSIZE), "--seed", str(s),
               "--workers", str(WORKERS), "--polar", POLAR,
               "--seed-design", str(seeds[m]), "--out", str(OUT / tag)]
        log = open(OUT / f"{tag}.log", "w", encoding="utf-8")
        flags = 0
        if os.name == "nt":
            flags = CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP | BELOW_NORMAL_PRIORITY_CLASS
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
        try:
            p = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                 creationflags=flags | (CREATE_BREAKAWAY_FROM_JOB if os.name == "nt" else 0),
                                 env=env)
        except OSError:
            # a job object that forbids breakaway: start inside it
            p = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                 creationflags=flags, env=env)
        record.append({"tag": tag, "mission": m, "seed": s, "pid": p.pid,
                       "cmd": cmd, "started": time.time()})
        print(f"  {tag:<16} pid {p.pid}")
    (OUT / "runs.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"\n{len(record)} searches x {WORKERS} workers -> {OUT}")


LINE = re.compile(r"^\s*(~)?\s*\[\s*(\d+)\] score\s+(-?[\d.]+)(.*)$")


def status() -> None:
    import psutil
    record = json.loads((OUT / "runs.json").read_text(encoding="utf-8"))
    now = time.time()
    print(f"{'run':<16} {'state':<8} {'h':>5} {'evals':>7} {'best':>10}  latest best")
    for r in record:
        alive = psutil.pid_exists(r["pid"]) and psutil.Process(r["pid"]).status() != "zombie"
        log = (OUT / f"{r['tag']}.log").read_text(encoding="utf-8", errors="replace")
        best, n, text = None, 0, ""
        for line in log.splitlines():
            mm = LINE.match(line)
            if mm:
                n = int(mm.group(2))
                best, text = float(mm.group(3)), ("feasible" if not mm.group(1) else "") + mm.group(4)
        done = (OUT / r["tag"] / "design.json").exists()
        state = "done" if done else ("running" if alive else "DIED")
        print(f"{r['tag']:<16} {state:<8} {(now - r['started']) / 3600:5.2f} {n:>7} "
              f"{best if best is not None else float('nan'):>10.2f}  {text.strip()[:70]}")
    print(f"\nmachine: {psutil.cpu_percent(interval=2.0):.0f}% of "
          f"{psutil.cpu_count()} threads, {psutil.virtual_memory().percent:.0f}% of memory")


def pick() -> None:
    """Each mission's best: any feasible design beats any infeasible one,
    then the score -- the same ordering the search itself uses."""
    best = {}
    for m, s, tag in runs():
        f = OUT / tag / "design.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        key = (bool(d.get("feasible")), float(d.get("score", -1e18)))
        if m not in best or key > best[m][0]:
            best[m] = (key, tag, d)
    for m in MISSIONS:
        if m in best:
            (ok, score), tag, d = best[m]
            print(f"{m:<11} {tag:<16} {'FEASIBLE' if ok else 'infeasible':<10} score {score:9.3f} "
                  f"| {'; '.join(d.get('reasons', []))[:120]}")
        else:
            print(f"{m:<11} no finished run")


def export() -> None:
    """Each mission's winner, exported the way it would be built: STLs,
    the build sheet, and cad/ with its 3D gates. Serial, at below-normal
    priority -- by the time this runs the machine is free anyway."""
    best = {}
    for m, s, tag in runs():
        f = OUT / tag / "design.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        key = (bool(d.get("feasible")), float(d.get("score", -1e18)))
        if m not in best or key > best[m][0]:
            best[m] = (key, tag)
    flags = BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 0
    for m, (_, tag) in best.items():
        out = OUT / "final" / m
        out.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "run.py", "export", "--mission", m,
               "--design", str(OUT / tag / "design.json"), "--polar", POLAR,
               "--out", str(out), "--step"]
        with open(out / "export.log", "w", encoding="utf-8") as log:
            rc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                creationflags=flags,
                                env={**os.environ, "PYTHONIOENCODING": "utf-8"}).returncode
        print(f"{m:<11} {tag:<16} -> {out}  (exit {rc})")


def track() -> None:
    """Each mission's winner into results/fleet, after `export`.

    A FEASIBLE winner becomes the tracked answer for its mission in
    index.json. An infeasible one is kept as a folder -- it is the best
    starting point the next search has -- but not indexed: the index
    names designs that meet their mission, or the last one that did."""
    import shutil
    fleet = ROOT / "results" / "fleet"
    index_path = fleet / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    best = {}
    for m, s, tag in runs():
        f = OUT / tag / "design.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        key = (bool(d.get("feasible")), float(d.get("score", -1e18)))
        if m not in best or key > best[m][0]:
            best[m] = (key, tag)
    for m, ((ok, score), tag) in best.items():
        dest = fleet / f"gen7_{tag}"
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(OUT / tag / "design.json", dest / "design.json")
        shutil.copy2(OUT / tag / "search_log.json", dest / "search_log.json")
        png = OUT / "final" / m / "design.png"
        if png.exists():
            shutil.copy2(png, dest / "design.png")
        if ok:
            index[m] = dest.name
        print(f"{m:<11} -> {dest.name}{'  (indexed)' if ok else '  (kept, not indexed: infeasible)'}")
    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    {"launch": launch, "status": status, "pick": pick,
     "export": export, "track": track}[sys.argv[1]]()
