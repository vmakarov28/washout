"""`run.py export --step`: the aircraft as STEP, and the 3D gates.

Writes, beside the STLs:

    cad/<name>_assembly.step   flight frame, mm, both halves, the spar
                               tubes, every part and face named
    cad/parts/<part>.step      each printed part alone, in its print frame,
                               root face on z = 0 -- the STL's own frame
    cad/<name>_cad.json        what the files are, where they came from,
                               and every gate below with its number

Built from the SAME evaluation the STLs are -- the settings the verdict
was reached with, the frames the panels were sliced in, the tubes the fit
found -- because a CAD file of an aircraft nobody scored is the one thing
this must never be (ROADMAP-CAD.md section 6).

## The gates

    loft fidelity     held-out layers vs the fitted surfaces        mm
    volume            solid vs the printed layers' own mesh         %
    validity          one closed, valid solid per part              --
    faces             faces per part, against a budget              --
    tube in shell     each tube's clearance to the wing's skin      mm
    size              the assembly file                             MB

The tube gate is the one a section-at-a-time pipeline could not ask. It
measures the distance from the carbon cylinder to the printed skin in 3D,
which the bore gate only approximates layer by layer.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..printing import elevons as elv
from ..printing import stl, vase
from ..printing.vase import Gate
from . import bspline as bs
from . import brep

FIDELITY_MM = 0.05
VOLUME_PCT = 0.5
FACES_PER_PART = 12
ASSEMBLY_MB = 8.0


def _tube_dims(ev, f) -> tuple[float, float]:
    """The tube that will be bought: the one the structure sized, else the
    fit's own; the bore of the stock tube with that outside diameter."""
    from .. import structure as st
    od = ev.structure.spar.od_mm if ev.structure is not None else f.spec.d_mm
    od = max(od, f.spec.d_mm)
    t = st.tube_for_od(od)
    return float(t.od_mm), float(t.id_mm)


def export_step(ev, out_dir, every_mm: float = 3.0, tol_mm: float = 0.02,
                z_step_mm: float = 0.5) -> dict:
    """Write cad/ for one evaluated design. -> the report dict, which is
    also written as JSON; its "gates" are `vase.Gate`s."""
    ps = ev.print_settings
    plan = ev.plan
    name = plan.name
    out = Path(out_dir) / "cad"
    (out / "parts").mkdir(parents=True, exist_ok=True)

    # The outer mould line of the printed parts: the same frames, the same
    # slices, no rib detours -- the ribs are the structure inside it.
    plain = replace(ps, ribs=False)
    spans = vase.panel_etas(plan, plain)
    wing = vase.build_panels(plan, plain, z_step_mm=z_step_mm)
    elev = elv.build_elevons(plan, plain, spans,
                             ev.max_elevon_deflect_deg + plain.hinge_margin_deg,
                             z_step_mm=z_step_mm)

    gates: list[Gate] = []
    parts, rows = [], []
    for part in wing + elev:
        truncated = (part.role == "wing" and elv.has_elevon(plain)
                     and part.frame.eta0 >= plain.elevon_eta - 1e-9)
        surfs, rep = bs.fit_part(part, every_mm=every_mm, tol_mm=tol_mm,
                                 truncated=truncated)
        solid, info = brep.part_solid(surfs)
        V, T = stl.skin(part)
        v_mesh = stl.volume_mm3(V, T)
        v_occ = brep.volume_mm3(solid)
        dv = 100.0 * (v_occ / v_mesh - 1.0)
        n_faces = len(info["faces"])
        dev = max(rep["fit_dev_mm"], rep["held_out_dev_mm"])
        valid = brep.is_valid(solid)
        gates += [
            Gate(f"{part.name} loft fidelity", dev <= FIDELITY_MM, dev,
                 FIDELITY_MM, "mm", f"{rep['sections']} sections, held-out layers"),
            Gate(f"{part.name} volume", abs(dv) <= VOLUME_PCT, abs(dv),
                 VOLUME_PCT, "%", f"{v_occ / 1000:.2f} cm3 vs the layers' {v_mesh / 1000:.2f}"),
            Gate(f"{part.name} valid solid", valid, float(valid), 1.0, "",
                 "closed, sewn, BRepCheck"),
            Gate(f"{part.name} faces", n_faces <= FACES_PER_PART, n_faces,
                 FACES_PER_PART, "", ", ".join(n for n, _ in info["faces"])),
        ]
        doc = brep.StepDocument(part.name)
        doc.add(part.name, solid, "vase", info["faces"])
        path = out / "parts" / f"{part.name}.step"
        doc.write(path)
        left = brep.transformed(solid, brep.print_to_flight_left(part.frame, part.origin_mm))
        right = brep.mirrored_y(left)
        names = [n for n, _ in info["faces"]]
        parts.append((part, left, right, names))
        rows.append({"part": part.name, "role": part.role, "faces": names,
                     "file": f"parts/{part.name}.step",
                     "kB": round(os.path.getsize(path) / 1000.0, 1),
                     "fit_dev_mm": round(rep["fit_dev_mm"], 4),
                     "held_out_dev_mm": round(rep["held_out_dev_mm"], 4),
                     "volume_err_pct": round(dv, 4)})

    # The joint wedge inserts: built in the flight frame, the right wing's,
    # so they share faces with the panels there by construction; the part
    # file is the left wing's, placed by the insert's own proper motion.
    for ins in getattr(ev, "inserts", None) or []:
        if ins.glue_fill:
            continue                 # filled with glue, not a part (BUILD.md)
        right, info = brep.insert_solid(ins)
        left = brep.mirrored_y(right)
        v_occ = brep.volume_mm3(right)
        dv = 100.0 * (v_occ / ins.volume_mm3 - 1.0)
        valid = brep.is_valid(right)
        names = [n for n, _ in info["faces"]]
        gates += [
            Gate(f"{ins.name} volume", abs(dv) <= 1.0, abs(dv), 1.0, "%",
                 f"{v_occ / 1000:.2f} cm3 vs the STL's {ins.volume_mm3 / 1000:.2f}"),
            Gate(f"{ins.name} valid solid", valid, float(valid), 1.0, "",
                 "ruled loft, bores cut, BRepCheck"),
            Gate(f"{ins.name} faces", len(names) <= FACES_PER_PART, len(names),
                 FACES_PER_PART, "", ", ".join(names)),
        ]
        printed = brep.transform_rt(left, ins.print_R, ins.print_t)
        doc = brep.StepDocument(ins.name)
        doc.add(ins.name, printed, "solid",
                list(zip(names, brep._shapes(printed, brep.TopAbs_FACE))))
        path = out / "parts" / f"{ins.name}.step"
        doc.write(path)
        parts.append((ins, left, right, names))
        rows.append({"part": ins.name, "role": "insert", "faces": names,
                     "file": f"parts/{ins.name}.step",
                     "kB": round(os.path.getsize(path) / 1000.0, 1),
                     "volume_err_pct": round(dv, 4)})

    # The tubes, in the flight frame: one a side, meeting at the centreline.
    tubes = []
    w = ps.extrusion_width_mm
    need = 0.5 * w + 0.5 * ps.spar_clearance_mm
    right_wings = [(p, r, names) for p, _, r, names in parts
                   if getattr(p, "role", "insert") == "wing"]
    skins = brep.compound([f for _, r, names in right_wings
                           for (n, f) in zip(names, brep._shapes(r, brep.TopAbs_FACE))
                           if n == "skin"])
    for f in ev.spar_fits:
        od, idm = _tube_dims(ev, f)
        x0, z0 = f.root_xz_mm
        sx, sz = f.slope
        right = brep.tube((x0, 0.0, z0), (sx, 1.0, sz), f.reach_mm, od, idm)
        left = brep.mirrored_y(right)
        tubes.append((f, left, right))
        gap = brep.min_distance_mm(right, skins) if right_wings else 0.0
        # Clearance alone cannot tell inside from outside, so the axis is
        # sampled too. A joint's wedge is inside the wing -- it is filled
        # when the panels are bonded -- but inside neither printed part,
        # so a point that close to a joint counts as in.
        H = plan.half_span_m * 1000.0
        joints = [(p.frame.eta0 * H, p.frame.wedge_mm) for p, _, _ in right_wings
                  if p.frame.eta0 > 0.0]
        lost = 0
        for y in np.linspace(0.05, 0.95, 10) * f.reach_y_mm:
            xm, zm = f.centre_mm(y)
            if any(brep.point_inside(r, (xm, y, zm)) for _, r, _ in right_wings):
                continue
            if any(abs(y - yj) <= w_j + od for yj, w_j in joints):
                continue
            lost += 1
        gates.append(Gate(f"{f.spec.name} tube in shell", gap >= need and lost == 0,
                          gap, need, "mm",
                          f"{od:.0f}x{idm:.0f} tube to the printed skin"
                          + ("" if lost == 0 else
                             f"; {lost} of 10 points on its axis are OUTSIDE the wing")))

    doc = brep.StepDocument(f"{name} assembly")
    for part, left, right, names in parts:
        colour = "vase" if getattr(part, "role", "insert") in ("wing", "elevon") else "solid"
        for side, shape in (("left", left), ("right", right)):
            # a copying transform keeps the faces in order, so the names
            # the part's own file carries carry over by index
            faces = list(zip(names, brep._shapes(shape, brep.TopAbs_FACE)))
            doc.add(f"{part.name} ({side})", shape, colour, faces)
    for f, left, right in tubes:
        od, idm = _tube_dims(ev, f)
        doc.add(f"{f.spec.name} {od:.0f}x{idm:.0f} (left)", left, "carbon")
        doc.add(f"{f.spec.name} {od:.0f}x{idm:.0f} (right)", right, "carbon")
    asm = out / f"{name}_assembly.step"
    doc.write(asm)
    mb = os.path.getsize(asm) / 1e6
    gates.append(Gate("assembly file", mb <= ASSEMBLY_MB, mb, ASSEMBLY_MB, "MB",
                      f"{len(parts) * 2 + len(tubes) * 2} solids"))

    report = {
        "design": name,
        "verdict": {"ok": bool(ev.ok), "reasons": list(ev.reasons)},
        "frame": "assembly: flight, mm, x aft of the root LE, y starboard, "
                 "z up; parts: their print frame, root face on z = 0, which "
                 "is the LEFT half as printed -- mirror for the right",
        "print_settings": {"extrusion_width_mm": ps.extrusion_width_mm,
                           "layer_h_mm": ps.layer_h_mm,
                           "spar_d_mm": ps.spar_d_mm},
        "surface": "outer mould line of the printed parts: the contour is "
                   "the bead's centreline (thicken_for_nozzle), and whether "
                   "the slicer lays the bead's centre or its edge on it is "
                   "unverified -- ROADMAP-CAD.md section 0.4",
        "parts": rows,
        "tubes": [{"spar": f.spec.name, "od_id_mm": list(_tube_dims(ev, f)),
                   "length_per_side_mm": round(f.reach_mm, 1),
                   "sweep_deg": round(f.sweep_deg, 2),
                   "dihedral_deg": round(f.dihedral_deg, 2)}
                  for f in ev.spar_fits],
        "assembly": {"file": asm.name, "MB": round(mb, 3)},
        "gates": [{"name": g.name, "passed": bool(g.passed), "value": float(g.value),
                   "limit": float(g.limit), "units": g.units, "detail": g.detail}
                  for g in gates],
    }
    (out / f"{name}_cad.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["_gates"] = gates
    return report
