#!/usr/bin/env python3
"""washout: search a blended wing body, print it in vase mode.

    python run.py search  [--iters 60] [--span 1.0] [--out out/run1]
    python run.py export  --design out/run1/design.json
    python run.py check                      # re-score this mission's tracked design

Everything downstream of `search` is deterministic: the design vector in
design.json plus the print settings reproduce the exact STLs, so a run is
reported by its numbers and rebuilt from them.
"""

from __future__ import annotations

# One BLAS thread per process, set before numpy is imported.
#
# The searches are run several at a time and each one's numpy tries to
# use all 24 cores for its 512x512 AIC inverse. Oversubscribed, that
# inverse took 7.05 SECONDS instead of milliseconds and was half of every
# design evaluation. Single-threaded, the processes stop fighting and the
# machine runs one search per core instead of three searches per machine.
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from washout import build_sheet
from washout import spars as sp
from washout.geom import cst
from washout.printing import elevons, stl, vase
from washout.search.design import (BOUNDS, MISSIONS, Mission,  # noqa: F401
                                choose_structure, evaluate, unit_to_physical)
from washout.search.optimize import SEEDS, run_search

ROOT = Path(__file__).resolve().parent


def print_settings(a) -> vase.PrintSettings:
    return vase.PrintSettings(
        bed_x_mm=a.bed, bed_y_mm=a.bed, bed_z_mm=a.bed_z,
        layer_h_mm=a.layer_h, extrusion_width_mm=a.width,
        nozzle_mm=a.nozzle, filament_density_gcc=a.density,
        spar_d_mm=a.spar,
        ribs=a.ribs, rib_count=a.rib_count, rib_pitch_mm=a.rib_pitch,
    )


FIN_DEFAULTS = {"fin_area_frac": 0.0, "fin_aspect": 1.3, "fin_below": 0.0}


def load_seed_physical(path: Path) -> dict:
    """A previous run's design, as a starting point for this search.

    Seeding stops a generation backsliding. gen4's micro search finished
    behind gen3's winner even though, by the gen4 search's own gates, that
    winner was feasible: differential evolution starts from random designs
    and never found its way back to it. With the old winner in the starting
    population, the best feasible design can only be at least as good --
    provided the old winner is still feasible under the new gates.

    Reads the name-keyed `physical` dict, never `u`: `u` is positional and
    changes meaning every time a design variable is added. Designs from
    before the tip-fin variables are lifted with no fins; any other missing
    variable is an error rather than a guess."""
    phys = dict(json.loads(Path(path).read_text())["physical"])
    for k, v in FIN_DEFAULTS.items():
        phys.setdefault(k, v)
    missing = [b.name for b in BOUNDS if b.name not in phys]
    if missing:
        raise SystemExit(f"{path}: design predates variables {missing}; "
                         f"cannot seed from it")
    return {b.name: float(np.clip(phys[b.name], b.lo, b.hi)) for b in BOUNDS}


def report(ev, settings: vase.PrintSettings) -> str:
    if ev.plan is None:
        return "no geometry"
    lines = [ev.plan.report(), ""]
    if ev.mass:
        lines += [ev.mass.report(), ""]
    if ev.trim:
        t = ev.trim
        lines += [
            f"flight: trim alpha {t.alpha_deg:+.2f} deg | CL {t.CL:.3f} | "
            f"cruise {t.v_ms:.1f} m/s",
            f"        L/D {t.LD:.2f}  (CD0 {t.CD0:.4f} + CDi {t.CDi:.4f})",
            f"        neutral point {t.x_np_m*1000:.1f} mm | CG "
            f"{t.x_cg_m*1000:.1f} mm | static margin {t.static_margin:+.3f}",
            f"        peak section cl {t.cl_local_max:.2f} | verdict: {t.reason}",
            "",
        ]
    if getattr(ev, "structure", None) is not None:
        lines += [ev.structure.report(), ""]
    if getattr(ev, "fins", None) is not None:
        lines += [ev.fins.report(), ""]
    if getattr(ev, "dynamics", None) is not None:
        lines += [ev.dynamics.report(), ""]
    if getattr(ev, "fairness", None) is not None:
        lines += [ev.fairness.report(getattr(ev, "fairness_limits", None)), ""]
    if getattr(ev, "lateral", None) is not None:
        lines += [ev.lateral.report(), ""]
    if getattr(ev, "aeroelastic", None) is not None:
        lines += [ev.aeroelastic.report(), ""]
    if getattr(ev, "linkage", None) is not None:
        from washout import linkage as _lkg
        lines += [_lkg.report(ev.linkage, ev.max_elevon_deflect_deg), ""]
    if ev.print_settings is not None and elevons.has_elevon(ev.print_settings):
        lines += [elevons.hinge_report(ev.plan, ev.print_settings,
                                       ev.max_elevon_deflect_deg), ""]
    if getattr(ev, "spar_fits", None):
        lines += [sp.report(ev.spar_fits, ev.plan), ""]
    return "\n".join(lines)


def do_export(ev, settings: vase.PrintSettings, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    try:
        from washout.report import figure
        fig = figure(ev, settings, out / "design.png",
                     title=f"washout {ev.plan.span_m*1000:.0f} mm BWB")
        print(f"\nfigure: {fig}")
    except Exception as e:  # noqa: BLE001
        # Any plotting failure, not just a missing matplotlib: the figure
        # is drawn BEFORE the STLs, and must never cost the parts.
        print(f"\nfigure skipped ({type(e).__name__}: {e})")
    panels = vase.build_panels(ev.plan, settings)
    # The control surfaces, as their own parts. The hinge line runs
    # spanwise and print Z is the span, so an elevon prints root-down in
    # exactly the same orientation as the wing panel it came off.
    panels = panels + elevons.build_elevons(
        ev.plan, settings, vase.panel_etas(ev.plan, settings),
        ev.max_elevon_deflect_deg + settings.hinge_margin_deg)
    total_g = total_min = 0.0
    print(f"\nprintable parts (one half wing; mirror for the other side):")
    for pan in panels:
        chk = vase.check(pan)
        rep = stl.export(pan, out / f"{pan.name}.stl")
        deg, bx, by = pan.best_bed_rotation()
        print(f"  {pan.name:<12} eta {pan.eta[0]:.2f}-{pan.eta[-1]:.2f}  "
              f"h {pan.height_mm:6.1f} mm  {pan.mass_g():5.1f} g  "
              f"{pan.print_time_min():4.0f} min  "
              f"{bx:.0f}x{by:.0f} mm @ {deg:.0f} deg  "
              f"{'OK' if chk.ok else 'CHECK: ' + ','.join(chk.failures())}")
        if not chk.ok:
            print(chk.report())
        total_g += pan.mass_g()
        total_min += pan.print_time_min()
    print(f"  {'TOTAL x2':<12} {'':22} {2*total_g:5.1f} g  {2*total_min:4.0f} min")
    fins = getattr(ev, "fins", None)
    if fins is not None:
        rep = fins.export_stl(out / f"{ev.plan.name}_tip_fin.stl")
        o = fins.outline()
        w_mm = float(o[:, 0].max() - o[:, 0].min()) * 1000.0
        h_mm = float(o[:, 1].max() - o[:, 1].min()) * 1000.0
        print(f"  {ev.plan.name + '_tip_fin':<14} flat plate {w_mm:.0f}x{h_mm:.0f} mm, "
              f"{fins.thickness_m*1000:.1f} mm -- print TWO in normal (not vase) "
              f"mode, glue to the tips  {'OK' if rep.get('watertight') else 'CHECK MESH'}")
    sheet = build_sheet.write(ev, panels, settings, out / "BUILD.md")
    print(f"\n  build sheet: {sheet}")
    # The slicer settings are in BUILD.md and nowhere else now. The
    # prose here said "1 bottom layer" while spars.report said the root
    # face needs ZERO -- and one bottom layer seals it, which makes the
    # spar corridor a closed pocket and the electronics unreachable. Two
    # places disagreeing about the single most load-bearing setting in
    # the project, one of them wrong, for fifteen commits.
    print(f"  spar: {settings.spar_d_mm} mm tube, seated against the skin "
          f"BUILD.md names.")
    print("  every other print setting is in BUILD.md: read it before "
          "slicing.")


def build_parser() -> argparse.ArgumentParser:
    """The CLI, as a value rather than a side effect of main().

    Separated so it can be TESTED. argparse used to carry its own
    hardcoded list of missions, the list had drifted from the factories in
    design.py, and `--mission fpv_1m` was duly offered by --help and by
    tab completion while raising AttributeError on every command. A
    source-text assertion cannot catch that -- the fix has to be checked
    against the parser argparse actually builds."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["search", "export", "check"])
    # Choices come from design.py, never from a second list here: the
    # hardcoded one had drifted and offered `fpv_1m`, which has no factory
    # and crashed every command that named it.
    ap.add_argument("--mission", choices=MISSIONS, default="trainer_v3")
    ap.add_argument("--polar", type=str, default=None,
                    help="measured LBM polar CSV; switches the search from "
                         "the flat tier-0 drag model to strip theory on real "
                         "data. Without it the optimizer is not charged for "
                         "flying at a high angle of attack.")
    ap.add_argument("--span", type=float, default=None, help="m; overrides the mission")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--popsize", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "out" / "run")
    ap.add_argument("--design", type=Path, default=None)
    ap.add_argument("--airfoil", type=Path, default=ROOT / "assets" / "mh45.dat")
    ap.add_argument("--bed", type=float, default=256.0, help="X1C bed")
    ap.add_argument("--bed-z", type=float, default=250.0, dest="bed_z")
    ap.add_argument("--layer-h", type=float, default=0.25, dest="layer_h")
    ap.add_argument("--width", type=float, default=0.45)
    ap.add_argument("--nozzle", type=float, default=0.4)
    ap.add_argument("--density", type=float, default=0.55,
                    help="g/cc; 0.55 Bambu PLA Aero foamed, 1.24 solid PLA")
    ap.add_argument("--spar", type=float, default=6.0)
    ap.add_argument("--ribs", action="store_true",
                    help="internal truss during the SEARCH too, so its mass "
                         "is carried from the start rather than discovered "
                         "afterwards")
    ap.add_argument("--rib-count", type=int, default=3, dest="rib_count")
    ap.add_argument("--rib-pitch", type=float, default=25.0, dest="rib_pitch")
    ap.add_argument("--no-structure", action="store_true", dest="no_structure",
                    help="skip spar/rib sizing; export a bare vase shell")
    ap.add_argument("--seed-design", type=Path, default=None, dest="seed_design",
                    help="design.json from an earlier run to put in the starting "
                         "population, so the search cannot finish behind it")
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)

    base = cst.load_selig(a.airfoil)
    mission = getattr(Mission, a.mission)()
    if a.span is not None:
        mission = Mission(**{**mission.__dict__, "span_m": a.span})
    settings = print_settings(a)
    over = {"spar_d_mm": mission.spar_d_mm} if a.spar == 6.0 else {}
    if mission.max_overhang_deg is not None:
        over["max_overhang_deg"] = mission.max_overhang_deg
    if over:
        settings = vase.PrintSettings(**{**settings.__dict__, **over})
    a.out.mkdir(parents=True, exist_ok=True)

    # Build the drag model ONCE, before the command dispatch. It used to
    # be constructed only inside the search branch, so `export` quietly
    # fell back to the flat tier-0 model and reported L/D 12.63 for a
    # design the search had scored at 7.62. Same design, two different
    # physics, and the flattering one is the number that would have been
    # handed over.
    drag = None
    if a.polar:
        from washout.aero.performance import MeasuredDrag, MultiRePolar
        specs = []
        for chunk in str(a.polar).split(","):
            if "@" in chunk:
                path, re_ = chunk.rsplit("@", 1)
                specs.append((float(re_), Path(path)))
            else:
                specs.append((6.0e4, Path(chunk)))
        drag = (MultiRePolar.from_files(specs) if len(specs) > 1
                else MeasuredDrag.from_csv(specs[0][1]))
        print("drag: measured, Re "
              + ", ".join(f"{r:.0f}" for r, _ in specs))

    if a.command == "check":
        # Re-score the design this repository says is the answer for this
        # mission, and say whether it still passes.
        #
        # This used to be `seed`, and it could not work: SEEDS is {} by
        # deliberate design ("an empty dict means start from scratch, which
        # is one less thing to keep true"), so every mission printed "no
        # seed recorded" and exited 1 -- while the README advertised it as
        # the first command to run. Pointed at results/fleet instead it
        # answers a question worth asking: does the tracked winner still
        # hold under today's gates? It did not, the first time it was run:
        # the trainer went to 355 g and outside its wing-loading gate as
        # soon as its own spars were weighed.
        index = json.loads((ROOT / "results" / "fleet" / "index.json")
                           .read_text(encoding="utf-8"))
        folder = index.get(a.mission)
        if not folder:
            print(f"no tracked design for {a.mission}; "
                  f"have {sorted(k for k in index if not k.startswith('_'))}")
            return 1
        path = ROOT / "results" / "fleet" / folder / "design.json"
        print(f"checking {path.relative_to(ROOT)}\n")
        u = np.array(json.loads(path.read_text(encoding="utf-8"))["u"])
        ev = evaluate(u, mission, base, settings, want_panels=True, drag=drag)
        print(report(ev, settings))
        if ev.reasons:
            print("issues:", "; ".join(ev.reasons))
            return 1
        print("all gates pass")
        return 0

    if a.command == "export":
        d = json.loads((a.design or (a.out / "design.json")).read_text())
        u = np.array(d["u"])
        # Spar and ribs are sized inside evaluate(), so this is the same
        # verdict the search scored -- one pipeline, not two.
        ev = evaluate(u, mission, base, settings, want_panels=True, drag=drag,
                      size_structure=not a.no_structure)
        print(report(ev, settings))
        if ev.reasons:
            print("issues:", "; ".join(ev.reasons), "\n")
        do_export(ev, ev.print_settings or settings, a.out)
        return 0

    seed_phys = (load_seed_physical(a.seed_design) if a.seed_design
                 else SEEDS.get(a.mission))
    if a.seed_design:
        print(f"seeded from {a.seed_design}")
    print(f"washout search [{a.mission}]: {mission.span_m*1000:.0f} mm span, bed "
          f"{a.bed:.0f}x{a.bed:.0f}x{a.bed_z:.0f} mm\n")
    best_u, best, log = run_search(mission, base, settings, maxiter=a.iters,
                                   popsize=a.popsize, seed=a.seed, drag=drag,
                                   out_dir=a.out, seed_physical=seed_phys)
    (a.out / "design.json").write_text(json.dumps({
        "u": list(map(float, best_u)),
        "physical": unit_to_physical(best_u),
        "score": best.score, "ld": best.ld, "v_cruise": best.v_cruise,
        "mass_kg": best.mass_kg, "static_margin": best.static_margin,
        "feasible": best.ok, "reasons": list(best.reasons),
        "seeded_from": str(a.seed_design) if a.seed_design else None,
    }, indent=2), encoding="utf-8")

    print(f"\n{'='*66}\n{log.evaluations} evaluations, "
          f"{log.feasible} feasible\n{'='*66}")
    print(report(best, settings))
    for k, v in unit_to_physical(best_u).items():
        print(f"    {k:<16} {v:8.3f}")
    if best.ok:
        # the settings the verdict was reached with: spar and ribs sized
        do_export(best, best.print_settings or settings, a.out)
    else:
        print("\nbest design still violates:", "; ".join(best.reasons))
        print("no STL written -- fix the mission or widen the bounds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
