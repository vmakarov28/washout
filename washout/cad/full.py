"""`run.py export --step-full`: the aircraft as built, every body in it.

`cad/export.py` writes each printed part as the solid its outer mould line
encloses, which is right for mating and dimensioning and wrong for seeing
the structure: a vase part is a single bead, a WALL, and its rib truss is
inside that wall. This writes the aircraft as it is assembled:

    printed wall      each wing panel's wall, 0.45 mm in from its outer
                      mould line, open where it is bonded (cad/offset.py)
    rib webs          the diagonal truss, one solid per web per panel
    battery hatch     cut through the centre body's upper skin, and the
                      piece that comes out, as the lid
    elevons           outer mould line (their wall is not resolved: see
                      the report)
    companions        joint inserts, tip fins, motor mount, control horns
    bought parts      the carbon tubes
    payload           the seated boxes of the pack, the receiver and the
                      servos -- reserved volumes, coloured as such

into cad/<name>_full.step, with cad/<name>_full.json holding every gate.
What is NOT in it, and why, is in that report too: the motor, the
propeller and the camera have no declared dimensions in this program, and
a shape invented for them would be the one thing this file must not have.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..geom import interior as it
from ..printing import elevons as elv
from ..printing import parts as pm
from ..printing import stl, vase
from ..printing.vase import Gate
from . import brep
from . import bspline as bs
from .offset import offset_stack

WALL_VOLUME_PCT = 5.0
WEB_VOLUME_PCT = 15.0
WEB_SAMPLE_LAYERS = 8          # every 8th printed layer: 2 mm at 0.25 mm


# ------------------------------------------------------------ geometry


def mesh_solid(verts: np.ndarray, tris: np.ndarray):
    """A closed triangle mesh as a B-rep solid: one planar face a triangle,
    sewn. For the companion parts, whose meshes ARE their geometry."""
    from OCP.BRepBuilderAPI import (BRepBuilderAPI_MakeFace,
                                    BRepBuilderAPI_MakePolygon,
                                    BRepBuilderAPI_MakeSolid,
                                    BRepBuilderAPI_Sewing)
    from OCP.ShapeFix import ShapeFix_Solid
    from OCP.TopAbs import TopAbs_SHELL, TopAbs_SOLID
    from OCP.TopoDS import TopoDS
    from OCP.gp import gp_Pnt

    sew = BRepBuilderAPI_Sewing(1e-4)
    for t in tris:
        poly = BRepBuilderAPI_MakePolygon()
        for i in t:
            poly.Add(gp_Pnt(*map(float, verts[i])))
        poly.Close()
        sew.Add(BRepBuilderAPI_MakeFace(poly.Wire(), True).Face())
    sew.Perform()
    shells = brep._shapes(sew.SewedShape(), TopAbs_SHELL)
    if len(shells) != 1:
        raise RuntimeError(f"mesh sewed into {len(shells)} shells")
    solid = BRepBuilderAPI_MakeSolid(TopoDS.Shell(shells[0])).Solid()
    fix = ShapeFix_Solid(solid)
    fix.Perform()
    return brep._shapes(fix.Shape(), TopAbs_SOLID)[0]


def transform_points(verts: np.ndarray, fn) -> np.ndarray:
    return np.array([fn(*v) for v in verts], dtype=float)


def rib_sections(stack, w: float) -> list:
    """Per printed layer, each rib web's section as a quad in print X/Y,
    sorted chordwise. A rib is a detour of the upper skin: the loop dives
    from the skin to two floor vertices one slit apart and climbs back
    (printing/ribs.py). The web is the two legs welded, so its section
    runs from half a bead outside the first leg to half a bead outside
    the second, and from the skin down to half a bead below the floor."""
    out = []
    h = 0.5 * w
    for k in range(len(stack.z_mm)):
        P = stack.contours[k]
        n = len(P)
        X, Y = P[:, 0], P[:, 1]
        ribs = []
        for i in range(1, n - 2):
            if (Y[i - 1] - Y[i] > 1.0 and Y[i + 2] - Y[i + 1] > 1.0
                    and abs(Y[i] - Y[i + 1]) < 0.3 and abs(X[i] - X[i + 1]) < 2.0):
                top = sorted([P[i - 1], P[i + 2]], key=lambda q: q[0])
                bot = sorted([P[i], P[i + 1]], key=lambda q: q[0])
                quad = [(top[0][0] - h, top[0][1]), (bot[0][0] - h, bot[0][1] - h),
                        (bot[1][0] + h, bot[1][1] - h), (top[1][0] + h, top[1][1])]
                ribs.append((0.5 * (bot[0][0] + bot[1][0]), quad))
        ribs.sort(key=lambda r: r[0])
        out.append([q for _, q in ribs])
    return out


def web_solids(stack, w: float, every: int = WEB_SAMPLE_LAYERS) -> tuple[list, dict]:
    """Every web of one ribbed panel as a lofted solid, in its print frame.
    -> ([solid, ...], report). Layers whose rib count differs from the
    panel's are skipped, and the report says how many."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakePolygon
    from OCP.BRepOffsetAPI import BRepOffsetAPI_ThruSections
    from OCP.gp import gp_Pnt

    secs = rib_sections(stack, w)
    counts = [len(s) for s in secs]
    n_r = max(counts) if counts else 0
    ks = sorted(set(list(range(0, len(secs), every)) + [len(secs) - 1]))
    ks = [k for k in ks if len(secs[k]) == n_r]
    solids = []
    for r in range(n_r):
        ts = BRepOffsetAPI_ThruSections(True, True, 1e-6)
        for k in ks:
            poly = BRepBuilderAPI_MakePolygon()
            for x, y in secs[k][r]:
                poly.Add(gp_Pnt(float(x), float(y), float(stack.z_mm[k])))
            poly.Close()
            ts.AddWire(poly.Wire())
        ts.Build()
        solids.append(ts.Shape())
    return solids, {"ribs": n_r, "sections": len(ks),
                    "layers_off_count": int(sum(c != n_r for c in counts))}


def _perimeter_volume_mm3(stack, w: float) -> float:
    """The printed bead's volume: each layer's loop length times the bead
    width, integrated over the print height."""
    P = stack.contours
    per = np.linalg.norm(np.roll(P, -1, 1) - P, axis=2).sum(1)
    return float(np.trapezoid(per * w, stack.z_mm))


def _skin_band(plan, x0, x1, y0, y1, which, n=7):
    half = plan.half_span_m * 1000.0
    zs = [it.skin_z_mm(plan, abs(y) / half, x, which)
          for x in np.linspace(x0, x1, n) for y in np.linspace(y0, y1, n)]
    return float(min(zs)), float(max(zs))


# ------------------------------------------------------------ the export


def export_full(ev, out_dir, mission=None, every_mm: float = 3.0,
                tol_mm: float = 0.02, z_step_mm: float = 0.5,
                log=print) -> dict:
    ps = ev.print_settings
    plan = ev.plan
    name = plan.name
    w = ps.extrusion_width_mm
    half = plan.half_span_m * 1000.0
    rc = plan.stations[0].chord_m * 1000.0
    out = Path(out_dir) / "cad"
    out.mkdir(parents=True, exist_ok=True)
    gates: list[Gate] = []
    bodies = []            # (name, shape, colour)
    notes = []

    plain = replace(ps, ribs=False)
    spans = vase.panel_etas(plan, plain)
    wing = vase.build_panels(plan, plain, z_step_mm=z_step_mm)
    elev = elv.build_elevons(plan, plain, spans,
                             ev.max_elevon_deflect_deg + plain.hinge_margin_deg,
                             z_step_mm=z_step_mm)

    def both(label, shape_left, colour):
        bodies.append((f"{label} (left)", shape_left, colour))
        bodies.append((f"{label} (right)", brep.mirrored_y(shape_left), colour))

    # --- the printed walls of the wing panels ---
    walls_left = {}
    for part in wing:
        truncated = (elv.has_elevon(plain) and part.frame.eta0 >= plain.elevon_eta - 1e-9)
        so, _ = bs.fit_part(part, every_mm=every_mm, tol_mm=tol_mm, truncated=truncated)
        inner = replace(part, contours=offset_stack(part.contours, w))
        si, rep_i = bs.fit_part(inner, every_mm=every_mm, tol_mm=tol_mm, truncated=truncated)
        wall = None
        try:
            wall, _ = brep.wall_solid(so, si)
        except Exception as e:                       # noqa: BLE001
            notes.append(f"{part.name}: wall not built ({type(e).__name__}: {e})")
        est = _perimeter_volume_mm3(part, w)
        ok = wall is not None and brep.is_valid(wall)
        dv = 100.0 * (brep.volume_mm3(wall) / est - 1.0) if ok else 999.0
        gates.append(Gate(f"{part.name} wall valid solid", ok, float(ok), 1.0, "",
                          "outer mould line to one bead in, open where bonded"))
        gates.append(Gate(f"{part.name} wall volume", ok and abs(dv) <= WALL_VOLUME_PCT,
                          abs(dv), WALL_VOLUME_PCT, "%",
                          f"{(brep.volume_mm3(wall) if ok else 0)/1000:.2f} cm3 vs "
                          f"the layers' bead {est/1000:.2f}"))
        if not ok:
            outer, _ = brep.part_solid(so)
            wall = outer
            notes.append(f"{part.name}: exported as its outer mould line, the wall failed")
        left = brep.transformed(wall, brep.print_to_flight_left(part.frame, part.origin_mm))
        walls_left[part.name] = left
        log(f"  wall  {part.name}: {'valid' if ok else 'FAILED'}, {dv:+.1f}% vs bead")

    # --- the rib truss, from the panels as printed ---
    plain_real = {p.name: p for p in vase.build_panels(plan, plain, z_step_mm=None)}
    for part in ev.panels:
        if not part.has_ribs:
            notes.append(f"{part.name}: carries no rib truss")
            continue
        solids, rep = web_solids(part, w)
        trsf = brep.print_to_flight_left(part.frame, part.origin_mm)
        v_web = sum(brep.volume_mm3(s) for s in solids)
        pl = plain_real.get(part.name)
        extra = (_perimeter_volume_mm3(part, w) - _perimeter_volume_mm3(pl, w)
                 if pl is not None else 0.0)
        valid = all(brep.is_valid(s) for s in solids)
        dv = 100.0 * (v_web / extra - 1.0) if extra > 0 else 999.0
        gates.append(Gate(f"{part.name} rib webs valid", valid, float(valid), 1.0, "",
                          f"{rep['ribs']} webs lofted through {rep['sections']} layers"))
        gates.append(Gate(f"{part.name} rib web volume", abs(dv) <= WEB_VOLUME_PCT,
                          abs(dv), WEB_VOLUME_PCT, "%",
                          f"{v_web/1000:.2f} cm3 vs the bead the ribs add, "
                          f"{extra/1000:.2f} (the web also fills its slit)"))
        for i, s in enumerate(solids):
            both(f"{part.name} rib {i + 1}", brep.transformed(s, trsf), "vase")
        log(f"  webs  {part.name}: {rep['ribs']} ribs, {dv:+.1f}% vs bead")

    # --- the battery hatch, cut through the bonded centre body ---
    from .. import build_sheet as bsheet
    h = bsheet.hatch_template(ev)
    root_panel = wing[0].name
    if h is not None and root_panel in walls_left:
        zu_lo, zu_hi = _skin_band(plan, h["x0"], h["x1"], 0.0, h["y"], it.UPPER)
        zl_lo, zl_hi = _skin_band(plan, h["x0"], h["x1"], 0.0, h["y"], it.LOWER)
        z_cut = zu_lo - 2.0 * w - 1.0
        clear = z_cut - (zl_hi + 2.0 * w)
        gates.append(Gate("hatch cuts the upper skin only", clear > 0.0, clear, 0.0,
                          "mm", "cut plane to the lower skin's inside"))
        cutter = brep.box(h["x0"], h["x1"], -h["y"], h["y"], z_cut, zu_hi + 5.0)
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut
        wl = walls_left[root_panel]
        wr = brep.mirrored_y(wl)
        cut_l = BRepAlgoAPI_Cut(wl, cutter).Shape()
        cut_r = BRepAlgoAPI_Cut(wr, cutter).Shape()
        lid = brep.compound([BRepAlgoAPI_Common(wl, cutter).Shape(),
                             BRepAlgoAPI_Common(wr, cutter).Shape()])
        ok = brep.is_valid(cut_l) and brep.is_valid(cut_r) and brep.is_valid(lid)
        v_lid = brep.volume_mm3(lid)
        gates.append(Gate("hatch cut valid", ok, float(ok), 1.0, "",
                          f"x {h['x0']:.0f}-{h['x1']:.0f} mm, y +/-{h['y']:.0f} mm; "
                          f"lid {v_lid/1000:.2f} cm3"))
        ae = getattr(ev, "aeroelastic", None)
        if ae is not None and ae.hatch_eta > 0.0 and mission is not None:
            lim = mission.min_aeroelastic_margin
            gates.append(Gate("hatch stiffness margin", ae.margin_hatch >= lim,
                              ae.margin_hatch, lim, "x",
                              f"reversal {ae.v_rev_hatch_ms:.0f} m/s with it cut"))
        walls_left.pop(root_panel)
        bodies.append((f"{root_panel} wall (left)", cut_l, "vase"))
        bodies.append((f"{root_panel} wall (right)", cut_r, "vase"))
        bodies.append(("battery hatch lid", lid, "vase"))
        log(f"  hatch: {'valid' if ok else 'FAILED'}, lid {v_lid/1000:.2f} cm3")
    centre_wall = (bodies[-3][1] if h is not None and root_panel not in walls_left
                   else brep.mirrored_y(walls_left[root_panel]))
    for pname, left in walls_left.items():
        both(f"{pname} wall", left, "vase")

    # --- the elevons: outer mould line ---
    for part in elev:
        so, _ = bs.fit_part(part, every_mm=every_mm, tol_mm=tol_mm)
        solid, _ = brep.part_solid(so)
        both(part.name, brep.transformed(solid, brep.print_to_flight_left(
            part.frame, part.origin_mm)), "vase")
    notes.append("elevons: outer mould line, not their wall -- their thin, cut "
                 "nose collapses the inside offset at the tip, and the wall "
                 "fitted there is not a trustworthy solid")

    # --- joint inserts and tubes ---
    for ins in getattr(ev, "inserts", None) or []:
        if ins.glue_fill:
            continue
        right, _ = brep.insert_solid(ins)
        bodies.append((f"{ins.name} (right)", right, "solid"))
        bodies.append((f"{ins.name} (left)", brep.mirrored_y(right), "solid"))
    from .export import _tube_dims
    for f in ev.spar_fits:
        od, idm = _tube_dims(ev, f)
        x0, z0 = f.root_xz_mm
        sx, sz = f.slope
        right = brep.tube((x0, 0.0, z0), (sx, 1.0, sz), f.reach_mm, od, idm)
        bodies.append((f"{f.spec.name} {od:.0f}x{idm:.0f} (right)", right, "carbon"))
        bodies.append((f"{f.spec.name} {od:.0f}x{idm:.0f} (left)",
                       brep.mirrored_y(right), "carbon"))

    # --- tip fins: flat plates, outboard of the tip ---
    fins = getattr(ev, "fins", None)
    if fins is not None:
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon
        from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
        from OCP.gp import gp_Pnt, gp_Vec
        poly = BRepBuilderAPI_MakePolygon()
        y_tip = fins.y_m * 1000.0
        for x, z in fins.outline() * 1000.0:
            poly.Add(gp_Pnt(float(x), y_tip, float(z)))
        poly.Close()
        face = BRepBuilderAPI_MakeFace(poly.Wire(), True).Face()
        fin = BRepPrimAPI_MakePrism(face, gp_Vec(0.0, fins.thickness_m * 1000.0, 0.0)).Shape()
        bodies.append(("tip fin (right)", fin, "solid"))
        bodies.append(("tip fin (left)", brep.mirrored_y(fin), "solid"))

    # --- the motor mount: its part frame is (span, up, forward of its aft face) ---
    pt = getattr(ev, "powertrain", None)
    if pt is not None:
        mount, _ = pm.mount_for(plan, ps, pt, f"{name}_motor_mount")
        x_aft = mount.x_web_mm + pm.WEB_T_MM
        V = transform_points(mount.verts, lambda u, v, ww: (x_aft - ww, u, v))
        ms = mesh_solid(V, mount.tris)
        bodies.append(("motor mount", ms, "solid"))
        gap = brep.min_distance_mm(ms, centre_wall)
        gates.append(Gate("motor mount placed on the trailing edge", gap <= 1.0,
                          gap, 1.0, "mm", "mount to the centre body's wall"))

    # --- control horns: a plate in the section plane, at the horn station ---
    if ev.linkage is not None and ev.horn_eta > 0.0:
        rod_n = (mission.servo_stall_nmm / max(mission.servo_arm_mm, 1e-9)
                 if mission is not None else 0.0)
        horn = pm.horn_for(plan, ev.linkage, ev.horn_eta, w, f"{name}_horn",
                           rod_n * ev.linkage.horn_arm_mm)
        x_mid, _, _, _, length = pm.horn_geometry(plan, ev.linkage, ev.horn_eta, w,
                                                  rod_n * ev.linkage.horn_arm_mm)
        st = plan.at(float(ev.horn_eta))
        c, x_le = st.chord_m * 1000.0, st.x_le_m * 1000.0
        x0 = x_mid - 0.5 * length
        f0 = float(np.clip((x0 - x_le) / c, 0.0, 1.0))
        y0 = float(st.airfoil.y_upper(np.array([f0]))[0]) * c
        ang = np.radians(-st.twist_deg)
        ca, sa = np.cos(ang), np.sin(ang)
        wz = horn.verts[:, 2]
        wmid = 0.5 * (wz.min() + wz.max())
        y_st = ev.horn_eta * half

        def place(u, v, ww):
            px, py = (x0 + u - x_le) - 0.25 * c, y0 + v
            return (x_le + 0.25 * c + px * ca - py * sa, y_st + (ww - wmid),
                    st.z_le_m * 1000.0 + px * sa + py * ca)
        hs = mesh_solid(transform_points(horn.verts, place), horn.tris)
        bodies.append(("control horn (right)", hs, "solid"))
        bodies.append(("control horn (left)", brep.mirrored_y(hs), "solid"))

    # --- payload: the seated boxes, reserved volume ---
    for v in getattr(ev, "bays", ()):
        x0, x1 = v.x0 * rc, v.x1 * rc
        y0, y1 = v.eta0 * half, v.eta1 * half
        zu_lo, _ = _skin_band(plan, x0, x1, y0, y1, it.UPPER)
        zl_lo, zl_hi = _skin_band(plan, x0, x1, y0, y1, it.LOWER)
        if v.anchor == it.UPPER:
            top = zu_lo - v.offset_mm
            bot = top - v.height_mm
        else:
            bot = zl_hi + v.offset_mm
            top = bot + v.height_mm
        if y0 <= 1e-9:
            bodies.append((f"{v.name} (reserved)", brep.box(x0, x1, -y1, y1, bot, top), "payload"))
        else:
            b = brep.box(x0, x1, y0, y1, bot, top)
            bodies.append((f"{v.name} (reserved, right)", b, "payload"))
            bodies.append((f"{v.name} (reserved, left)", brep.mirrored_y(b), "payload"))
    notes.append("not modelled: the motor, the propeller and the FPV camera -- "
                 "their dimensions are not declared anywhere in this program")
    notes.append("the outer surface is the bead's CENTRELINE (thicken_for_nozzle), "
                 "as in the part files; the wall is drawn one bead inward of it")

    invalid = [n for n, s, _ in bodies if not brep.is_valid(s)]
    gates.append(Gate("every body a valid solid", not invalid, float(len(invalid)), 0.0,
                      "", ", ".join(invalid) or f"{len(bodies)} bodies"))

    doc = brep.StepDocument(f"{name} as built")
    for n_, s, col in bodies:
        doc.add(n_, s, col)
    path = out / f"{name}_full.step"
    doc.write(path)
    mb = os.path.getsize(path) / 1e6
    report = {
        "design": name, "file": path.name, "MB": round(mb, 2),
        "bodies": [n_ for n_, _, _ in bodies],
        "frame": "flight, mm: x aft of the root LE, y starboard, z up",
        "gates": [{"name": g.name, "passed": bool(g.passed), "value": float(g.value),
                   "limit": float(g.limit), "units": g.units, "detail": g.detail}
                  for g in gates],
        "notes": notes,
    }
    (out / f"{name}_full.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["_gates"] = gates
    return report
