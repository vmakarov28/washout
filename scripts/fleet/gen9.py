#!/usr/bin/env python3
"""Generation 9: micro_fpv for a beginner -- flown slow, and a stall that
drops the nose.

    python scripts/fleet/gen9.py launch      # start every search, detached
    python scripts/fleet/gen9.py status      # progress, and how busy the machine is
    python scripts/fleet/gen9.py pick        # each family's winner, feasible first
    python scripts/fleet/gen9.py export      # the winners: STLs, inserts, build sheet, STEP
    python scripts/fleet/gen9.py track       # the winners into results/fleet

## What changed since gen8 (2026-09-23)

The objective is the slowest TRIMMED speed (performance.slow_flight): the
first section to reach cl_max, or the up-elevon running out, rather than
the whole wing at the declared cl_max with nothing spent on trim. Two
gates are new: hands-off at least 1.3x that speed, and the first section
to stall ahead of the CG. Neither gen8 winner passes the second -- both
start to stall at mid-span, 0.18 and 0.31 MAC behind the CG, and pitch
up into it. An 8-generation probe moved the curl to 0.10 MAC behind and
found nothing feasible; the winglet did not move.

Five curl seeds and three winglet seeds, because the probe says the
curl is the family with somewhere to go. Each is seeded from its gen8
winner, which is infeasible now but is the nearest known point.

## Sized for the machine

Eight searches of three workers: 24 workers on 24 threads.
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
OUT = ROOT / "out" / "gen9"
POLAR = ("data/polars/trainer_mid_re60k.csv@60000,"
         "data/polars/thin_reflex_re100k.csv@100000")
FLEET = ROOT / "results" / "fleet"
FAMILIES = {
    "micro_fpv": ("gen8_micro_fpv_v132",) * 5,
    "micro_fpv_winglet": ("gen8_micro_fpv_winglet_v130",) * 3,
}
WORKERS = 3
POPSIZE = 6
ITERS = 40

CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_PROCESS_GROUP = 0x00000200
BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
CREATE_BREAKAWAY_FROM_JOB = 0x01000000


def runs():
    for m, seeds in FAMILIES.items():
        for s, seed_folder in enumerate(seeds):
            yield m, s, f"{m}_v14{s}", FLEET / seed_folder / "design.json"


def launch() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    record = []
    for m, s, tag, seed in runs():
        cmd = [sys.executable, "run.py", "search", "--mission", m,
               "--iters", str(ITERS), "--popsize", str(POPSIZE), "--seed", str(s),
               "--workers", str(WORKERS), "--polar", POLAR,
               "--seed-design", str(seed), "--out", str(OUT / tag)]
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
            p = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                 creationflags=flags, env=env)
        record.append({"tag": tag, "mission": m, "seed": s, "pid": p.pid,
                       "seed_design": str(seed), "cmd": cmd, "started": time.time()})
        print(f"  {tag:<24} pid {p.pid}  seeded from {seed.parent.name}")
    (OUT / "runs.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"\n{len(record)} searches x {WORKERS} workers -> {OUT}")


LINE = re.compile(r"^\s*(~)?\s*\[\s*(\d+)\] score\s+(-?[\d.]+)(.*)$")


def status() -> None:
    import psutil
    record = json.loads((OUT / "runs.json").read_text(encoding="utf-8"))
    now = time.time()
    print(f"{'run':<24} {'state':<8} {'h':>5} {'evals':>7} {'best':>10}  latest best")
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
        print(f"{r['tag']:<24} {state:<8} {(now - r['started']) / 3600:5.2f} {n:>7} "
              f"{best if best is not None else float('nan'):>10.2f}  {text.strip()[:70]}")
    print(f"\nmachine: {psutil.cpu_percent(interval=2.0):.0f}% of "
          f"{psutil.cpu_count()} threads, {psutil.virtual_memory().percent:.0f}% of memory")


def _best() -> dict:
    """Each family's best: any feasible design beats any infeasible one,
    then the score -- the ordering the search itself uses."""
    best = {}
    for m, s, tag, _ in runs():
        f = OUT / tag / "design.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        key = (bool(d.get("feasible")), float(d.get("score", -1e18)))
        if m not in best or key > best[m][0]:
            best[m] = (key, tag, d)
    return best


def pick() -> None:
    best = _best()
    for m in FAMILIES:
        if m in best:
            (ok, score), tag, d = best[m]
            print(f"{m:<18} {tag:<24} {'FEASIBLE' if ok else 'infeasible':<10} score {score:9.3f} "
                  f"| {'; '.join(d.get('reasons', []))[:120]}")
        else:
            print(f"{m:<18} no finished run")


def export() -> None:
    flags = BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 0
    for m, (_, tag, _) in _best().items():
        out = OUT / "final" / m
        out.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "run.py", "export", "--mission", m,
               "--design", str(OUT / tag / "design.json"), "--polar", POLAR,
               "--out", str(out), "--step"]
        with open(out / "export.log", "w", encoding="utf-8") as log:
            rc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                creationflags=flags,
                                env={**os.environ, "PYTHONIOENCODING": "utf-8"}).returncode
        print(f"{m:<18} {tag:<24} -> {out}  (exit {rc})")


def track() -> None:
    """Each family's winner into results/fleet, after `export`. A feasible
    winner is indexed under its own mission name; micro_fpv_winglet is a
    variant (design.MISSION_VARIANTS), indexed beside its parent, never
    over it -- its design vector means a different aircraft under the
    parent's tip-rise cap."""
    import shutil
    index_path = FLEET / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    for m, ((ok, score), tag, _) in _best().items():
        dest = FLEET / f"gen9_{tag}"
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(OUT / tag / "design.json", dest / "design.json")
        shutil.copy2(OUT / tag / "search_log.json", dest / "search_log.json")
        png = OUT / "final" / m / "design.png"
        if png.exists():
            shutil.copy2(png, dest / "design.png")
        if ok:
            index[m] = dest.name
        print(f"{m:<18} -> {dest.name}{'  (indexed)' if ok else '  (kept, not indexed: infeasible)'}")
    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    {"launch": launch, "status": status, "pick": pick,
     "export": export, "track": track}[sys.argv[1]]()
