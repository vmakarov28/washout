#!/usr/bin/env python3
"""planeforge: search a blended wing body, print it in vase mode.

    python run.py search  [--iters 60] [--span 1.0] [--out out/run1]
    python run.py export  --design out/run1/design.json
    python run.py seed                       # evaluate the seed only

Everything downstream of `search` is deterministic: the design vector in
design.json plus the print settings reproduce the exact STLs, so a run is
reported by its numbers and rebuilt from them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from planeforge.geom import cst
from planeforge.printing import stl, vase
from planeforge.search.design import (Mission, evaluate, physical_to_unit,
                                      unit_to_physical)
from planeforge.search.optimize import SEED_PHYSICAL, TRAINER_SEED, run_search

ROOT = Path(__file__).resolve().parent


def print_settings(a) -> vase.PrintSettings:
    return vase.PrintSettings(
        bed_x_mm=a.bed, bed_y_mm=a.bed, bed_z_mm=a.bed_z,
        layer_h_mm=a.layer_h, extrusion_width_mm=a.width,
        nozzle_mm=a.nozzle, filament_density_gcc=a.density,
        spar_d_mm=a.spar,
        ribs=a.ribs, rib_count=a.rib_count, rib_pitch_mm=a.rib_pitch,
    )


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
    return "\n".join(lines)


def choose_structure(ev, mission, settings: vase.PrintSettings):
    """Size the spar and the rib pitch from the trimmed flight condition,
    then hand back the print settings that follow from them.

    This is the 'pick the best settings' step and it runs AFTER the
    aerodynamic search, not inside it: the loads depend on the span
    loading of the design that won, and no earlier point in the pipeline
    knows what that is."""
    from planeforge import structure
    from planeforge.aero.vlm import VLM
    from planeforge.search.design import LATTICE_NC, LATTICE_NS

    pt = VLM(ev.plan, LATTICE_NS, LATTICE_NC).solve(ev.trim.alpha_deg,
                                                    ev.trim.x_cg_m)
    st = structure.select(ev.plan, pt, ev.mass_kg,
                          skin_t_mm=settings.extrusion_width_mm,
                          n_limit=mission.n_limit_g)
    tuned = vase.PrintSettings(**{**settings.__dict__,
                                  "spar_d_mm": st.spar.od_mm,
                                  "ribs": True,
                                  "rib_pitch_mm": st.rib_pitch_mm})
    return st, tuned


def do_export(ev, settings: vase.PrintSettings, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    try:
        from planeforge.report import figure
        fig = figure(ev, settings, out / "design.png",
                     title=f"planeforge {ev.plan.span_m*1000:.0f} mm BWB")
        print(f"\nfigure: {fig}")
    except ImportError:
        pass
    panels = vase.build_panels(ev.plan, settings)
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
    print(f"\n  slice with SPIRAL VASE / spiralize outer contour ON, 0 top "
          f"layers,\n  1 bottom layer, {settings.extrusion_width_mm} mm "
          f"extrusion width, {settings.layer_h_mm} mm layers.")
    print(f"  spar: {settings.spar_d_mm} mm tube slides into the cavity "
          f"along the print Z axis.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["search", "export", "seed"])
    ap.add_argument("--mission", choices=["fpv_1m", "trainer"], default="fpv_1m")
    ap.add_argument("--span", type=float, default=None, help="m; overrides the mission")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--popsize", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "out" / "run")
    ap.add_argument("--design", type=Path, default=None)
    ap.add_argument("--airfoil", type=Path, default=ROOT / "assets" / "mh45.dat")
    ap.add_argument("--bed", type=float, default=256.0)
    ap.add_argument("--bed-z", type=float, default=250.0, dest="bed_z")
    ap.add_argument("--layer-h", type=float, default=0.25, dest="layer_h")
    ap.add_argument("--width", type=float, default=0.45)
    ap.add_argument("--nozzle", type=float, default=0.4)
    ap.add_argument("--density", type=float, default=0.60,
                    help="g/cc; 0.60 LW-PLA foamed, 1.24 solid PLA")
    ap.add_argument("--spar", type=float, default=6.0)
    ap.add_argument("--ribs", action="store_true",
                    help="internal truss during the SEARCH too, so its mass "
                         "is carried from the start rather than discovered "
                         "afterwards")
    ap.add_argument("--rib-count", type=int, default=3, dest="rib_count")
    ap.add_argument("--rib-pitch", type=float, default=25.0, dest="rib_pitch")
    ap.add_argument("--no-structure", action="store_true", dest="no_structure",
                    help="skip spar/rib sizing; export a bare vase shell")
    a = ap.parse_args()

    base = cst.load_selig(a.airfoil)
    mission = (Mission.beginner_trainer() if a.mission == "trainer"
               else Mission.fpv_1m())
    if a.span is not None:
        mission = Mission(**{**mission.__dict__, "span_m": a.span})
    settings = print_settings(a)
    a.out.mkdir(parents=True, exist_ok=True)

    if a.command == "seed":
        seed_phys = TRAINER_SEED if a.mission == "trainer" else SEED_PHYSICAL
        ev = evaluate(physical_to_unit(seed_phys), mission, base, settings,
                      want_panels=True)
        print(report(ev, settings))
        if ev.reasons:
            print("issues:", "; ".join(ev.reasons))
        return 0

    if a.command == "export":
        d = json.loads((a.design or (a.out / "design.json")).read_text())
        u = np.array(d["u"])
        ev = evaluate(u, mission, base, settings, want_panels=True)
        print(report(ev, settings))
        if ev.trim and not a.no_structure:
            st, settings = choose_structure(ev, mission, settings)
            print(st.report())
            print()
            # the spar just got chosen, so the bore gate must be re-run
            # against the tube we actually intend to slide in
            ev = evaluate(u, mission, base, settings, want_panels=True)
            if ev.reasons:
                print("after structure sizing:", "; ".join(ev.reasons), "\n")
        do_export(ev, settings, a.out)
        return 0

    print(f"planeforge search [{a.mission}]: {mission.span_m*1000:.0f} mm span, bed "
          f"{a.bed:.0f}x{a.bed:.0f}x{a.bed_z:.0f} mm\n")
    seed_phys = TRAINER_SEED if a.mission == "trainer" else SEED_PHYSICAL
    best_u, best, log = run_search(mission, base, settings, maxiter=a.iters,
                                   popsize=a.popsize, seed=a.seed,
                                   out_dir=a.out, seed_physical=seed_phys)
    (a.out / "design.json").write_text(json.dumps({
        "u": list(map(float, best_u)),
        "physical": unit_to_physical(best_u),
        "score": best.score, "ld": best.ld, "v_cruise": best.v_cruise,
        "mass_kg": best.mass_kg, "static_margin": best.static_margin,
        "feasible": best.ok, "reasons": list(best.reasons),
    }, indent=2), encoding="utf-8")

    print(f"\n{'='*66}\n{log.evaluations} evaluations, "
          f"{log.feasible} feasible\n{'='*66}")
    print(report(best, settings))
    for k, v in unit_to_physical(best_u).items():
        print(f"    {k:<16} {v:8.3f}")
    if best.ok:
        do_export(best, settings, a.out)
    else:
        print("\nbest design still violates:", "; ".join(best.reasons))
        print("no STL written -- fix the mission or widen the bounds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
