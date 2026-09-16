#!/usr/bin/env python3
"""A dozen random aircraft from the generator, on one sheet.

The optimizer only ever shows you its winner, and a winner tells you
about the objective, not about the generator. Whether the design space
itself is made of aeroplanes is a separate question, and the quickest
honest answer is to draw random points from it: if a random draw looks
like a wing that was assembled from parts, the optimizer is searching a
space of assembled parts, and its winner will look like one too.

    python scripts/contact_sheet.py --mission trainer_v3 --out sheet.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib                                              # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402

from washout.geom import cst                                      # noqa: E402
from washout.search.design import Mission, N_DIM, build           # noqa: E402


def draw(plan, ax_top, ax_front, n: int = 120) -> None:
    eta = np.linspace(0.0, 1.0, n)
    st = [plan.at(float(e)) for e in eta]
    y = eta * plan.half_span_m * 1000
    xle = np.array([s.x_le_m for s in st]) * 1000
    xte = xle + np.array([s.chord_m for s in st]) * 1000
    for sgn in (1, -1):
        ax_top.fill(np.concatenate([sgn * y, sgn * y[::-1]]),
                    np.concatenate([xle, xte[::-1]]),
                    color="#dfe6ee", edgecolor="#1b1b1b", lw=0.8)
    ax_top.set_aspect("equal")
    ax_top.invert_yaxis()
    ax_top.axis("off")

    ef = np.linspace(0.0, 1.0, 40)
    up, lo, yy = [], [], []
    for e in ef:
        p = plan.section_3d(float(e), 61)
        up.append(p[:, 2].max()); lo.append(p[:, 2].min())
        yy.append(e * plan.half_span_m)
    up, lo, yy = (np.array(a) * 1000 for a in (up, lo, yy))
    for sgn in (1, -1):
        ax_front.fill(np.concatenate([sgn * yy, sgn * yy[::-1]]),
                      np.concatenate([up, lo[::-1]]),
                      color="#c9d6e3", edgecolor="#1b1b1b", lw=0.6)
    ax_front.set_aspect("equal")
    ax_front.axis("off")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", default="trainer_v3")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--title", default="")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    base = cst.load_selig(ROOT / "assets/mh45.dat")
    mission = getattr(Mission, a.mission)()
    rng = np.random.default_rng(a.seed)

    cols = 4
    rows = int(np.ceil(a.n / cols))
    fig = plt.figure(figsize=(4.2 * cols, 3.3 * rows), facecolor="white")
    outer = fig.add_gridspec(rows, cols, hspace=0.25, wspace=0.08)
    drawn = tries = 0
    while drawn < a.n and tries < 50 * a.n:
        tries += 1
        plan = build(rng.random(N_DIM), mission, base)
        if not plan.is_valid()[0]:
            continue
        cell = outer[drawn // cols, drawn % cols].subgridspec(
            2, 1, height_ratios=[3, 1], hspace=0.02)
        ax_t = fig.add_subplot(cell[0])
        ax_f = fig.add_subplot(cell[1])
        draw(plan, ax_t, ax_f)
        # the verdict on the sheet itself, so a flagged shape and the
        # reason it was flagged can be judged side by side -- which is how
        # a metric that counts ripple instead of shape gets caught
        from washout.geom import fairness as fz
        bad = fz.measure(plan).violations(mission.fairness)
        verdict = ("fair" if not bad else
                   "UNFAIR: " + bad[0][0].split(" (")[0][:34])
        ax_t.set_title(f"#{drawn + 1}  AR {plan.aspect_ratio:.1f}\n{verdict}",
                       fontsize=7.5, loc="left",
                       color="#1e8449" if not bad else "#c0392b")
        drawn += 1
    fig.suptitle(a.title or f"{a.mission}: {drawn} random draws from the "
                 f"generator (top and front views)", fontsize=12, x=0.02,
                 ha="left")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=110, bbox_inches="tight")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
