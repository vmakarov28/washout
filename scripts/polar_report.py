#!/usr/bin/env python3
"""Plot the measured LBM polar against the VLM's assumptions.

The point of the figure is the DISAGREEMENT. The vortex lattice is
inviscid: its lift curve is a straight line for ever and its profile
drag has no alpha dependence at all, which is exactly why the optimizer
was never charged for trimming at 9 degrees. The tunnel measures the
curve bending over and the drag rising. Where those two part company is
the honest limit of every L/D this program has quoted.

    python scripts/polar_report.py --polar out/tunnel/polar.csv \
        --design out/trainer/design.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from washout.geom import cst                                    # noqa: E402
from washout.search.design import Mission, build                # noqa: E402
from washout.aero.vlm import VLM                                # noqa: E402
from washout.aero import performance as perf                    # noqa: E402
from washout.geom import planform                               # noqa: E402

INK, ACCENT, COOL = "#1b1b1b", "#c0392b", "#2c6fbb"


def read_polar(path: Path) -> dict:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    d = {k: np.array([float(r[k]) for r in rows]) for k in
         ("alpha", "cl", "cd", "ld")}
    order = np.argsort(d["alpha"])
    return {k: v[order] for k, v in d.items()}


def stall_alpha(p: dict) -> float | None:
    """First angle where the lift curve stops rising."""
    for i in range(1, len(p["alpha"])):
        if p["cl"][i] < p["cl"][i - 1]:
            return float(p["alpha"][i - 1])
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--polar", type=Path, default=ROOT / "out/tunnel/polar.csv")
    ap.add_argument("--design", type=Path, default=None)
    ap.add_argument("--mission", default="trainer_v3")
    ap.add_argument("--airfoil", type=Path, default=ROOT / "assets/mh45.dat")
    ap.add_argument("--re", type=float, default=6.0e4)
    ap.add_argument("--out", type=Path, default=ROOT / "out/tunnel/polar.png")
    a = ap.parse_args()

    p = read_polar(a.polar)
    base = cst.load_selig(a.airfoil)

    # the VLM's view of the SAME section, as a near-2D wing
    if a.design:
        d = json.loads(a.design.read_text())
        mission = getattr(Mission, a.mission)()
        section = build(np.array(d["u"]), mission, base).at(0.5).airfoil
    else:
        section = base
    wing = planform.Planform(
        100.0, tuple(planform.Station(e, 1.0, 0.0, 0.0, 0.0, section)
                     for e in (0.0, 1.0)), "2d")
    v = VLM(wing, ns=16, nc=10)
    al_v = np.linspace(p["alpha"].min(), p["alpha"].max(), 40)
    cl_v = np.array([v.solve(float(x), 0.25).CL for x in al_v])
    drag = perf.DragModel()
    cd_v = np.full_like(al_v, drag.cf(a.re) * drag.form_factor(section.t_max)
                        * section.perimeter * drag.roughness_factor)

    a_st = stall_alpha(p)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), facecolor="white")

    ax[0].plot(p["alpha"], p["cl"], "o-", color=ACCENT, lw=2, ms=5,
               label=f"LBM, Re {a.re:.0f}")
    ax[0].plot(al_v, cl_v, "--", color=COOL, lw=1.6, label="VLM (inviscid)")
    if a_st is not None:
        ax[0].axvline(a_st, color="#888", ls=":", lw=1.4)
        ax[0].annotate(f"stall {a_st:.0f}$\\degree$", (a_st, ax[0].get_ylim()[0]),
                       xytext=(4, 8), textcoords="offset points",
                       fontsize=9, color="#555")
    ax[0].set_xlabel("alpha [deg]"); ax[0].set_ylabel("$c_l$")
    ax[0].set_title("lift: the tunnel sees the curve bend over", loc="left",
                    fontsize=10)
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.25)

    ax[1].plot(p["alpha"], p["cd"], "o-", color=ACCENT, lw=2, ms=5,
               label="LBM")
    ax[1].plot(al_v, cd_v, "--", color=COOL, lw=1.6,
               label="tier-0 model (flat, no alpha)")
    ax[1].set_xlabel("alpha [deg]"); ax[1].set_ylabel("$c_d$")
    ax[1].set_title("drag: the tier-0 model has NO alpha dependence",
                    loc="left", fontsize=10)
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.25)

    ax[2].plot(p["cd"], p["cl"], "o-", color=ACCENT, lw=2, ms=5)
    for al, cd, cl in zip(p["alpha"], p["cd"], p["cl"]):
        ax[2].annotate(f"{al:.0f}", (cd, cl), fontsize=7, color="#666",
                       xytext=(4, -2), textcoords="offset points")
    ax[2].set_xlabel("$c_d$"); ax[2].set_ylabel("$c_l$")
    ax[2].set_title("drag polar (labels = alpha)", loc="left", fontsize=10)
    ax[2].grid(alpha=0.25)

    fig.suptitle(f"washout section in windtunnel-sim  --  "
                 f"Re {a.re:.0f}, t/c {section.t_max:.3f}, "
                 f"reflex te_camber {section.te_camber:+.4f}",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=140, bbox_inches="tight")
    print(f"figure -> {a.out}")

    best = int(np.argmax(p["ld"]))
    print(f"\nmeasured section polar (Re {a.re:.0f}):")
    print(f"  best L/D {p['ld'][best]:.1f} at alpha {p['alpha'][best]:+.0f}, "
          f"cl {p['cl'][best]:.3f}, cd {p['cd'][best]:.4f}")
    print(f"  cl_max {p['cl'].max():.3f}"
          + (f" at alpha {a_st:+.0f} (stall)" if a_st is not None
             else " (no stall inside the sweep)"))
    i4 = int(np.argmin(np.abs(p["alpha"] - 4.0)))
    print(f"  cd at alpha 4 measured {p['cd'][i4]:.4f} vs tier-0 model "
          f"{cd_v[0]:.4f}  ->  model is {cd_v[0]/p['cd'][i4]:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
