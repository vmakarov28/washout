#!/usr/bin/env python3
"""Tier 2: the whole aircraft in the D3Q19 tunnel, with smoke.

    python scripts/tunnel3d.py --mission micro_fpv \
        --design results/fleet/<folder>/design.json --alpha trim --re 15000

Everything else in this project judges a three-dimensional aeroplane with
two-dimensional evidence or an inviscid lattice. This runs the real thing,
at a Reynolds number a 16 GB card can hold, and shows it: a three-colour
smoke rake upstream of the wing and a smoke source at each tip, rendered
as a wind-tunnel photograph and exported for the browser viewer in the
tunnel repo (viewer3d/), where the flow can be walked around.

It follows the project's rule for the tunnel: WRITE A SCENE AND CALL IT.
washout builds the voxel mask, the smooth body mesh and the smoke rake,
writes a scenes3d-format YAML in physical units, and runs windtunnel's
`run3d.py` inside WSL -- under a systemd memory cap, at low priority, with
a CUDA allocator cap -- because the VM that owns the GPU also runs a live
process that a hung VM would kill (ROADMAP.md, constraints). An earlier
version of this script imported the solver into a generated runner; that
is gone.

THE REYNOLDS COMPROMISE, stated plainly. The aircraft flies at about
Re 1.4e5 on its mean chord. This runs at 1.5e4 -- ten times lower -- with
~68 cells on the mean chord. Tip-vortex structure, the spanwise load
distribution and where the flow first separates are governed by geometry
and are only weakly Reynolds-sensitive at a fixed angle of attack. Section
drag, the stall angle and every absolute coefficient are strongly
Reynolds-sensitive, and are NOT to be read off this. The output says so.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from washout.geom import cst                                       # noqa: E402
from washout.search.design import (MISSION_VARIANTS, MISSIONS, Mission,  # noqa: E402
                                   build, evaluate, unit_to_physical)

# The companion windtunnel repo: a sibling of this one unless WASHOUT_TUNNEL_DIR says otherwise.
WIN_TUNNEL = Path(os.environ.get("WASHOUT_TUNNEL_DIR", ROOT.parent / "windtunnel")).resolve()
WSL_PY = "~/venvs/windtunnel/bin/python"
NU_AIR = 1.5e-5


def wsl_path(p: Path) -> str:
    s = str(Path(p).resolve()).replace("\\", "/")
    return "/mnt/" + s[0].lower() + s[2:]


WSL_TUNNEL = wsl_path(WIN_TUNNEL)


# ------------------------------------------------------------- geometry


def _placed(st, loop):
    """A unit loop at one span station -> flight (x, z) metres: twist
    about the quarter chord, scaled, at the station's leading edge."""
    p = loop - np.array([0.25, 0.0])
    t = np.radians(-st.twist_deg)
    ct, s_ = np.cos(t), np.sin(t)
    r = np.stack([p[:, 0] * ct - p[:, 1] * s_, p[:, 0] * s_ + p[:, 1] * ct], 1)
    return r * st.chord_m + np.array([st.x_le_m + 0.25 * st.chord_m, st.z_le_m])


def _rot(p, a_deg):
    """Nose up by alpha, about the origin: (x, z) -> (x', z')."""
    a = np.radians(a_deg)
    ca, sa = np.cos(a), np.sin(a)
    return np.stack([p[:, 0] * ca + p[:, 1] * sa, -p[:, 0] * sa + p[:, 1] * ca], 1)


class Box:
    """The tunnel grid in flight metres: solver x = flight x (downstream),
    solver y = flight z (up), solver z = flight y (span)."""

    def __init__(self, plan, alpha, cells_per_mac, up_mac=1.5, down_mac=3.0,
                 height_mac=3.5, side_mac=1.3):
        mac = plan.mac_m
        self.dx = mac / cells_per_mac
        pts = []
        for e in np.linspace(0.0, 1.0, 21):
            st = plan.at(float(e))
            pts.append(_rot(_placed(st, st.airfoil.coords(41)), alpha))
        pts = np.concatenate(pts)
        x_lo, x_hi = pts[:, 0].min(), pts[:, 0].max()
        z_lo, z_hi = pts[:, 1].min(), pts[:, 1].max()
        self.x0 = x_lo - up_mac * mac
        self.nx = int(round((x_hi - x_lo + (up_mac + down_mac) * mac) / self.dx))
        zc = 0.5 * (z_lo + z_hi)
        self.ny = int(round(height_mac * mac / self.dx))
        self.y0 = zc - 0.5 * self.ny * self.dx
        self.nz = int(round((plan.span_m + 2.0 * side_mac * mac) / self.dx))
        self.z0 = -0.5 * self.nz * self.dx

    def cells(self, x, z_up, y_span):
        """Flight metres -> lattice cell coordinates (cell centres at +0.5)."""
        return ((np.asarray(x) - self.x0) / self.dx - 0.5,
                (np.asarray(z_up) - self.y0) / self.dx - 0.5,
                (np.asarray(y_span) - self.z0) / self.dx - 0.5)


def voxelize(plan, box: Box, alpha, fins=None) -> np.ndarray:
    """Boolean (nx, ny, nz) mask: both halves of the loft, and the fins.

    A span cell at a time: at one station the section is a graph over the
    chord -- two single-valued surfaces -- so a cell is inside when its
    height lies between them, one interpolation per streamwise column."""
    nx, ny, nz, dx = box.nx, box.ny, box.nz, box.dx
    mask = np.zeros((nx, ny, nz), dtype=bool)
    xs = box.x0 + (np.arange(nx) + 0.5) * dx
    ys = box.y0 + (np.arange(ny) + 0.5) * dx
    half = plan.half_span_m
    for k in range(nz):
        yspan = box.z0 + (k + 0.5) * dx
        eta = abs(yspan) / half
        if eta > 1.0:
            continue
        st = plan.at(float(eta))
        loop = st.airfoil.coords(161)
        n = (len(loop) + 1) // 2
        up, lo = loop[:n][::-1], loop[n - 1:]
        pu, pl = _rot(_placed(st, up), alpha), _rot(_placed(st, lo), alpha)
        x_lo = max(pu[:, 0].min(), pl[:, 0].min())
        x_hi = min(pu[:, 0].max(), pl[:, 0].max())
        col = (xs >= x_lo) & (xs <= x_hi)
        if not col.any():
            continue
        xc = xs[col]
        ou, ol = np.argsort(pu[:, 0]), np.argsort(pl[:, 0])
        zu = np.interp(xc, pu[ou, 0], pu[ou, 1])
        zl = np.interp(xc, pl[ol, 0], pl[ol, 1])
        inside = ((ys[None, :] >= np.minimum(zl, zu)[:, None])
                  & (ys[None, :] <= np.maximum(zl, zu)[:, None]))
        mask[np.flatnonzero(col), :, k] = inside
    if fins is not None:
        from matplotlib.path import Path as MPath
        poly = _rot(fins.outline(), alpha)
        path = MPath(poly)
        X, Yv = np.meshgrid(xs, ys, indexing="ij")
        inside = path.contains_points(np.stack([X.ravel(), Yv.ravel()], 1)).reshape(nx, ny)
        t_cells = max(1, int(round(fins.thickness_m / dx)))
        for side in (1.0, -1.0):
            kc = (side * fins.y_m - box.z0) / dx - 0.5
            k0 = int(round(kc - 0.5 * (t_cells - 1)))
            for k in range(k0, k0 + t_cells):
                if 0 <= k < nz:
                    mask[:, :, k] |= inside
    return mask


def body_mesh(plan, box: Box, alpha, fins=None, n_span=121, n_pts=121):
    """The smooth surface -- the loft both sides, tips capped, and the
    fins as thin plates -- in lattice cells, for the viewer."""
    etas = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n_span)))
    rings = []
    for e in np.concatenate([-etas[::-1], etas[1:]]):
        st = plan.at(float(abs(e)))
        pz = _rot(_placed(st, st.airfoil.coords(n_pts)), alpha)
        rings.append(np.stack(box.cells(pz[:, 0], pz[:, 1],
                                        np.full(len(pz), e * plan.half_span_m)), 1))
    R = np.stack(rings)                                    # (S, P, 3)
    S, P = R.shape[:2]
    V = R.reshape(-1, 3)
    T = []
    for i in range(S - 1):
        for j in range(P - 1):
            a, b = i * P + j, i * P + j + 1
            c, d = (i + 1) * P + j + 1, (i + 1) * P + j
            T += [(a, b, c), (a, c, d)]
    for i, flip in ((0, True), (S - 1, False)):            # tip caps
        c0 = len(V)
        V = np.vstack([V, R[i].mean(0)[None]])
        for j in range(P - 1):
            tri = (c0, i * P + j, i * P + j + 1)
            T.append(tri[::-1] if flip else tri)
    if fins is not None:
        poly = _rot(fins.outline(), alpha)
        h = 0.5 * fins.thickness_m
        for side in (1.0, -1.0):
            base = len(V)
            ring = []
            for dy in (-h, h):
                ring.append(np.stack(box.cells(poly[:, 0], poly[:, 1],
                                               np.full(len(poly), side * fins.y_m + dy)), 1))
            V = np.vstack([V] + ring)
            m = len(poly)
            for j in range(1, m - 1):
                T += [(base, base + j, base + j + 1),
                      (base + m, base + m + j + 1, base + m + j)]
            for j in range(m):
                j2 = (j + 1) % m
                T += [(base + j, base + m + j, base + m + j2),
                      (base + j, base + m + j2, base + j2)]
    return V.astype(np.float32), np.array(T, dtype=np.int64)


def smoke_and_seeds(plan, box: Box, alpha, mac_cells: float):
    """The smoke rake and the streamline seeds, in cells.

    Rake: a row of sources a quarter-chord ahead of the leading edge at
    each of 13 span stations, one just below the leading edge and one just
    above, coloured by where they are -- red outboard of 80% semi-span,
    green in the outer wing, blue in front of the centre body -- plus a
    red source just off each tip's trailing edge, where the tip vortex
    forms. Seeds: the same rake, denser, and a ring round each tip."""
    sources, seeds = [], []
    r = max(1.8, 0.03 * mac_cells)
    half = plan.half_span_m
    for e in np.linspace(-0.95, 0.95, 13):
        st = plan.at(float(abs(e)))
        le = _rot(np.array([[st.x_le_m, st.z_le_m]]), alpha)[0]
        ch = 0 if abs(e) > 0.8 else (1 if abs(e) > 0.35 else 2)
        for dz in (-0.06, 0.06):
            x = le[0] - 0.25 * plan.mac_m
            z = le[1] + dz * plan.mac_m
            cx, cy, cz = box.cells(x, z, e * half)
            sources.append([float(cx), float(cy), float(cz), float(r), int(ch)])
    tip = plan.at(1.0)
    te = _rot(_placed(tip, np.array([[1.0, 0.0]])), alpha)[0]
    for side in (1.0, -1.0):
        cx, cy, cz = box.cells(te[0] + 0.02 * plan.mac_m, te[1],
                               side * (half + 0.03 * plan.mac_m))
        sources.append([float(cx), float(cy), float(cz), float(r), 0])
    for e in np.linspace(-0.98, 0.98, 29):
        st = plan.at(float(abs(e)))
        le = _rot(np.array([[st.x_le_m, st.z_le_m]]), alpha)[0]
        for dz in (-0.12, -0.05, 0.05, 0.12):
            seeds.append(box.cells(le[0] - 0.4 * plan.mac_m, le[1] + dz * plan.mac_m,
                                   e * half))
    for side in (1.0, -1.0):
        for th in np.linspace(0.0, 2 * np.pi, 16, endpoint=False):
            rr = 0.08 * plan.mac_m
            seeds.append(box.cells(tip.x_le_m - 0.4 * plan.mac_m,
                                   tip.z_le_m + rr * np.sin(th),
                                   side * half + rr * np.cos(th)))
    return sources, np.array(seeds, dtype=np.float32)


# --------------------------------------------------------------- the run


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", type=Path, required=True)
    ap.add_argument("--mission", default="micro_fpv",
                    choices=MISSIONS + tuple(MISSION_VARIANTS))
    ap.add_argument("--airfoil", type=Path, default=ROOT / "assets/mh45.dat")
    ap.add_argument("--alpha", default="trim",
                    help="degrees, or 'trim' for the tier-0 trim angle")
    ap.add_argument("--re", type=float, default=1.5e4)
    ap.add_argument("--cells-per-mac", type=int, default=68, dest="cpm")
    ap.add_argument("--u-lat", type=float, default=0.08, dest="u_lat")
    ap.add_argument("--transits", type=float, default=4.0,
                    help="domain flow-throughs to run")
    ap.add_argument("--average-transits", type=float, default=1.5,
                    help="the last this-many flow-throughs are averaged")
    ap.add_argument("--frame-every", type=int, default=100)
    ap.add_argument("--viewer-frames", type=int, default=72)
    ap.add_argument("--mem-gb", type=int, default=24,
                    help="systemd MemoryMax for the tunnel process (VM RAM)")
    ap.add_argument("--vram-gb", type=float, default=11.0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--analyse-only", action="store_true", dest="analyse_only",
                    help="re-read a finished run in --out and redo the comparison")
    a = ap.parse_args()

    base = cst.load_selig(a.airfoil)
    mission = getattr(Mission, a.mission)()
    d = json.loads(a.design.read_text(encoding="utf-8"))
    u = np.array(d["u"])
    plan = build(u, mission, base)
    from washout.printing import vase
    ev = evaluate(u, mission, base, vase.PrintSettings(
        filament_density_gcc=0.55, spar_d_mm=mission.spar_d_mm))
    fins = ev.fins
    alpha = (float(ev.trim.alpha_deg) if a.alpha == "trim" and ev.trim is not None
             else float(a.alpha))

    tag = f"{plan.name}_a{alpha:.1f}_re{a.re/1000:.0f}k"
    out = a.out or ROOT / "out" / "tunnel3d" / tag
    out.mkdir(parents=True, exist_ok=True)
    box = Box(plan, alpha, a.cpm)
    cells = box.nx * box.ny * box.nz
    tau = 0.5 + 3.0 * a.u_lat * a.cpm / a.re
    print(f"aircraft : {plan.name}, span {plan.span_m*1000:.0f} mm, MAC "
          f"{plan.mac_m*1000:.1f} mm, alpha {alpha:.2f} deg"
          f"{' (tier-0 trim)' if a.alpha == 'trim' else ''}, fins "
          f"{'yes' if fins is not None else 'none'}")
    print(f"lattice  : {box.nx} x {box.ny} x {box.nz} = {cells/1e6:.1f} Mcells, "
          f"dx {box.dx*1000:.2f} mm, {a.cpm} cells/MAC")
    print(f"physics  : Re {a.re:.0f} on the MAC (flight ~"
          f"{ev.v_cruise * plan.mac_m / NU_AIR:.0f}), u_lat {a.u_lat}, tau {tau:.5f}")
    if tau < 0.501:
        raise SystemExit("tau under the SGS floor 0.501: raise cells or lower Re")
    if a.analyse_only:
        return analyse(out, plan, mission, base, u, alpha, box, a, ev)

    t0 = time.perf_counter()
    mask = voxelize(plan, box, alpha, fins)
    solid = int(mask.sum())
    vol_vox = solid * box.dx ** 3
    print(f"voxels   : {solid:,} solid ({100*solid/cells:.3f}%), "
          f"{vol_vox*1e6:.0f} cm3 vs the loft's {plan.volume_m3()*1e6:.0f} cm3 "
          f"({100*vol_vox/plan.volume_m3():.0f}%)  [{time.perf_counter()-t0:.1f} s]")
    np.save(out / "mask.npy", mask)
    V, T = body_mesh(plan, box, alpha, fins)
    np.savez(out / "body.npz", V=V, T=T)
    sources, seeds = smoke_and_seeds(plan, box, alpha, a.cpm)
    np.save(out / "seeds.npy", seeds)

    transit = box.nx / a.u_lat
    steps = int(a.transits * transit)
    avg_from = int(steps - a.average_transits * transit)
    viewer_every = max(int((steps - int(1.0 * transit)) / a.viewer_frames), 50)
    area_cells = plan.area_m2 / box.dx ** 2
    scene = {
        "name": tag,
        "description": (f"washout {plan.name} at alpha {alpha:.2f} deg, Re {a.re:.0f} "
                        f"on the MAC; voxel mask from washout/scripts/tunnel3d.py"),
        "physical": {"char_length_m": float(plan.mac_m),
                     "velocity_ms": float(a.re * NU_AIR / plan.mac_m),
                     "nu_m2s": NU_AIR},
        "lattice": {"cells_per_char": a.cpm, "u_lat": a.u_lat},
        "domain": {"length_chars": box.nx / a.cpm, "height_chars": box.ny / a.cpm,
                   "span_chars": box.nz / a.cpm},
        "sgs": True,
        "boundaries": {"inlet": "equilibrium_ramp", "outlet": "anechoic_sponge",
                       "top_bottom": "periodic", "ramp_steps": 2000,
                       "sponge_fraction": 0.08},
        "obstacle": {"type": "voxel_mask", "file": wsl_path(out / "mask.npy")},
        "dye": {"sources": sources, "every": 2, "density": 0.30,
                # full resolution for the photographs when the grid is the
                # real one: the outer wing is ~4 cells thick at 68 cells a
                # chord, and a pooled grid shades it in stripes
                "render_downsample": 1 if a.cpm >= 48 else 2},
        "viewer": {"aircraft": plan.name, "mission": a.mission,
                   "alpha_deg": alpha, "re_mac": a.re,
                   "re_flight_estimate": float(ev.v_cruise * plan.mac_m / NU_AIR),
                   "dx_m": box.dx, "chord_cells": float(a.cpm),
                   "ref_area_cells": float(area_cells),
                   "span_cells": float(plan.span_m / box.dx),
                   "caveat": ("Re is ~10x below flight: read the flow topology "
                              "(tip vortices, spanwise flow, where it separates), "
                              "not the coefficients.")},
    }
    import yaml
    (out / "scene.yaml").write_text(yaml.safe_dump(scene, sort_keys=False),
                                    encoding="utf-8")
    print(f"run      : {steps} steps ({a.transits:g} flow-throughs), averaged "
          f"from {avg_from}, viewer frame every {viewer_every}, "
          f"{len(sources)} smoke sources, {len(seeds)} streamline seeds")
    print(f"scene    : {out / 'scene.yaml'}")
    if a.dry_run:
        return 0

    tun = out / "tunnel"
    cmd = (f"cd {WSL_TUNNEL} && systemd-run --user --scope -q "
           f"-p MemoryMax={a.mem_gb}G -p MemorySwapMax=0 "
           # Expandable segments: the gen9 micro_fpv (48.5 Mcells) died
           # at the first average with 0.7 GB reserved but unusable.
           f"env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
           f"nice -n 10 {WSL_PY} run3d.py --scene {wsl_path(out / 'scene.yaml')} "
           f"--seed 0 --steps {steps} --solver fused --preset dye "
           f"--frame-every {a.frame_every} --guard-every 100 --measure-force "
           f"--force-bins 96 --average-from {avg_from} "
           f"--viewer-every {viewer_every} --body-mesh {wsl_path(out / 'body.npz')} "
           f"--seeds {wsl_path(out / 'seeds.npy')} --no-final-checkpoint "
           f"--vram-cap-gb {a.vram_gb} --out {wsl_path(tun)}")
    (out / "command.sh").write_text(cmd + "\n", encoding="utf-8")
    print(f"\ntunnel   : {cmd}\n", flush=True)
    t0 = time.perf_counter()
    with open(out / "tunnel.log", "w", encoding="utf-8") as log:
        p = subprocess.run(["wsl", "-d", "Ubuntu", "-e", "bash", "-lc", cmd],
                           stdout=log, stderr=subprocess.STDOUT)
    print(f"tunnel exit {p.returncode} after {(time.perf_counter()-t0)/60:.1f} min "
          f"(log {out / 'tunnel.log'})")
    if p.returncode != 0:
        return p.returncode
    return analyse(out, plan, mission, base, u, alpha, box, a, ev)


def circulation_loading(npz_path: Path, pad_mac: float, mac_cells: float):
    """Lift per unit span from the circulation round each section.

    At every spanwise station of the averaged, pooled velocity field: a
    rectangle in the (x, y) plane, `pad_mac` chords clear of the section
    all round, and the circulation round it, counter-clockwise,

        G = sum u_x dx (bottom) + u_y dy (right) - u_x dx (top) - u_y dy (left)

    Kutta-Joukowski gives the lift per unit span, l = -rho U G (a lifting
    section in flow along +x circulates clockwise). The rectangle crosses
    the wake behind the trailing edge, whose trailing vorticity is
    streamwise and passes straight through it; what it encloses is the
    section's bound circulation. -> (z index in cells, l per cell of span,
    U), both in lattice units."""
    z = np.load(npz_path)
    uu = z["u"].astype(np.float32)                 # (3, X, Y, Z) pooled
    solid = z["solid"]
    P = int(z["pool"])
    U = float(z["u_char"])
    X, Y, Z = solid.shape
    m = max(2, int(round(pad_mac * mac_cells / P)))
    zs, lift = [], []
    for k in range(Z):
        sl = solid[:, :, k]
        if not sl.any():
            continue
        xs, ys = np.nonzero(sl)
        x0, x1 = max(xs.min() - m, 1), min(xs.max() + m, X - 2)
        y0, y1 = max(ys.min() - m, 1), min(ys.max() + m, Y - 2)
        ux, uy = uu[0, :, :, k], uu[1, :, :, k]
        G = (ux[x0:x1, y0].sum() + uy[x1, y0:y1].sum()
             - ux[x0:x1, y1].sum() - uy[x0, y0:y1].sum()) * P
        zs.append((k + 0.5) * P)                   # pooled cell k -> cells
        lift.append(-U * G)
    return np.array(zs), np.array(lift), U


def analyse(out, plan, mission, base, u, alpha, box, a, ev) -> int:
    """Tunnel lift against the vortex lattice at the same angle: total, and
    strip by strip across the span."""
    from washout.aero.vlm import VLM
    import csv
    tun = out / "tunnel"
    rows = list(csv.DictReader(open(tun / "forces.csv", encoding="utf-8")))
    steps = np.array([int(r["step"]) for r in rows])
    fx = np.array([float(r["fx"]) for r in rows])
    fy = np.array([float(r["fy"]) for r in rows])
    q = 0.5 * a.u_lat ** 2
    A = plan.area_m2 / box.dx ** 2
    transit = box.nx / a.u_lat
    tail = steps >= steps.max() - a.average_transits * transit
    CL, CD = fy[tail].mean() / (q * A), fx[tail].mean() / (q * A)
    CL_sd = fy[tail].std() / (q * A)

    vlm = VLM(plan, ns=40, nc=6, fins=ev.fins)
    pt = vlm.solve(alpha, ev.trim.x_cg_m if ev.trim else 0.0)

    # spanwise: lift per unit span, tunnel bins vs lattice strips, both as
    # the local lift coefficient times chord over the MAC, c_l c / c_bar
    span_rows = [r for r in csv.reader(open(tun / "forces_span.csv", encoding="utf-8"))
                 if r and not r[0].startswith("#")][1:]
    zb = np.array([0.5 * (float(r[1]) + float(r[2])) for r in span_rows])
    width = np.array([float(r[2]) - float(r[1]) for r in span_rows])
    fyb = np.array([float(r[4]) for r in span_rows])
    # The tunnel's bins are ~4 cells wide, the scale of the voxel staircase,
    # and momentum exchange on a staircase is large and alternating from
    # step to step -- the 96-bin loading scattered +/-0.8 either side of
    # the lattice's curve. Summed in groups to ~16 cells, the staircase
    # averages out and what is left is the flow.
    g = max(1, int(round(16.0 / float(width.mean()))))
    n = len(fyb) // g * g
    zb = zb[:n].reshape(-1, g).mean(1)
    width = width[:n].reshape(-1, g).sum(1)
    fyb = fyb[:n].reshape(-1, g).sum(1)
    # a bin holds cells [z_lo, z_hi); cell c is centred at z0 + (c + 1/2) dx,
    # so the bin's cells average to z0 + (z_lo + z_hi)/2 dx
    y_span = box.z0 + zb * box.dx
    ccl_tunnel = fyb / (q * width) / (plan.mac_m / box.dx)
    order = np.argsort(pt.y_strip)
    ccl_vlm = (pt.cl_local * pt.chord_strip)[order] / plan.mac_m
    y_vlm = pt.y_strip[order]

    report = {"aircraft": plan.name, "mission": a.mission, "alpha_deg": alpha,
              "re_mac": a.re, "cells_per_mac": a.cpm,
              "tunnel": {"CL": float(CL), "CL_sd": float(CL_sd), "CD": float(CD)},
              "tier0": {"CL": float(pt.CL), "CDi": float(pt.CDi),
                        "e": float(pt.e_oswald)},
              "caveat": ("Re ~10x below flight: coefficients are not flight "
                         "values. Compare SHAPES: the spanwise loading, and "
                         "where it falls away.")}

    # the spanwise loading from circulation, where the averaged velocity
    # was saved; checked against the momentum-exchange total
    vel = tun / "velocity_avg_pool2.npz"
    y_kj = ccl_kj = None
    if vel.exists():
        zc, l_cell, U = circulation_loading(vel, 0.6, a.cpm)
        pool = int(np.load(vel)["pool"])
        # l is per CELL of span; each pooled station stands for `pool` cells
        L_kj = float(l_cell.sum()) * pool
        CL_kj = L_kj / (q * A)
        y_kj = box.z0 + zc * box.dx
        ccl_kj = l_cell / q / (plan.mac_m / box.dx)
        half = plan.half_span_m
        inner = np.abs(y_kj) <= 0.5 * half
        outer = np.abs(y_kj) > 0.8 * half
        share = lambda msk, y, l: float(l[msk].sum() / l.sum())      # noqa: E731
        vi = np.abs(y_vlm) <= 0.5 * half
        vo = np.abs(y_vlm) > 0.8 * half
        l_vlm = ccl_vlm * np.gradient(y_vlm)            # per strip: times its width
        report["tunnel"]["CL_circulation"] = CL_kj
        report["tunnel"]["circulation_vs_force"] = CL_kj / CL if CL else None
        report["span_share"] = {
            "tunnel_inner_half": share(inner, y_kj, l_cell),
            "tunnel_outer_20pct": share(outer, y_kj, l_cell),
            "vlm_inner_half": share(vi, y_vlm, l_vlm),
            "vlm_outer_20pct": share(vo, y_vlm, l_vlm),
            "note": ("fraction of the total lift carried inboard of half "
                     "semi-span, and outboard of 80%")}
    (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    viewer = tun / "viewer"
    if viewer.exists():                  # the browser viewer shows it too
        (viewer / "report.json").write_text(json.dumps(report, indent=1),
                                            encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.2))
    half = plan.half_span_m
    ax.plot(y_vlm / half, ccl_vlm, "-", lw=2, label=f"vortex lattice (inviscid), CL {pt.CL:.3f}")
    if y_kj is not None:
        ax.plot(y_kj / half, ccl_kj, "-", lw=1.6, color="C3",
                label=f"LBM tunnel, circulation (Kutta-Joukowski), CL {report['tunnel']['CL_circulation']:.3f}")
        # the same loading scaled to the lattice's total: the SHAPE, which
        # is what a ten-times-low Reynolds number can still be trusted for
        ax.plot(y_kj / half, ccl_kj * pt.CL / max(report["tunnel"]["CL_circulation"], 1e-9),
                "--", lw=1.2, color="C3", alpha=0.6,
                label="  the same, scaled to the lattice's CL (shape only)")
    ax.plot(y_span / half, ccl_tunnel, ".", ms=3, color="0.6",
            label=f"LBM tunnel, surface force binned (staircase-noisy), CL {CL:.3f}")
    ax.set_xlim(-1.3, 1.3)
    ax.axvline(-1, color="0.7", lw=0.8)
    ax.axvline(1, color="0.7", lw=0.8)
    ax.set_xlabel("y / semi-span")
    ax.set_ylabel("c_l c / MAC  (lift per unit span)")
    ax.set_title(f"{plan.name} at {alpha:.1f} deg: where the lift is carried")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "span_loading.png", dpi=120)
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
