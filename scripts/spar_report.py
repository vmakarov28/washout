#!/usr/bin/env python3
"""Where did the spars actually land? Reads an exported design and
reports tube placement, reach, and the joints it has to cross.

The search already fits and GATES on these (design.py), but the fits
were discarded after scoring, so the run logs could only assert that a
spar existed. This prints the geometry that was chosen.
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from washout.geom import cst
from washout.printing import vase
from washout import spars as sp
from washout.search.design import Mission, build

base = cst.load_selig(ROOT / "assets/mh45.dat")
settings = vase.PrintSettings()

for run, mname in [("trainer_v3_v62", "trainer_v3"), ("demon1_v61", "demon1"),
                   ("micro_v61", "micro"), ("micro_v60", "micro")]:
    d = ROOT / "out" / run / "design.json"
    if not d.exists():
        print(f"{run}: no design.json"); continue
    m = getattr(Mission, mname)()
    plan = build(np.array(json.loads(d.read_text())["u"]), m, base)
    joints = vase.panel_etas(plan, settings)
    fits = sp.fit_all(plan, m.spars, settings.extrusion_width_mm, joints)
    print(f"===== {run}   half-span {plan.half_span_m*1000:.0f} mm, "
          f"root chord {plan.stations[0].chord_m*1000:.0f} mm")
    print(f"      panel joints at eta: "
          + ", ".join(f"{e:.2f}" for _, e in joints))
    print(sp.report(fits, plan))
    print()
