#!/usr/bin/env python3
"""Tier 2: the whole aircraft in the D3Q19 tunnel.

Everything in this project so far has judged a three-dimensional
aeroplane with two-dimensional evidence. The vortex lattice is inviscid,
so it cannot stall; the 2D tunnel measures a section in isolation, so it
knows nothing about the tip. Between them they assert -- and cannot
check -- the single most important safety property the trainer has: that
the ROOT stalls before the tips.

This runs the real thing. It builds a voxel mask of the entire aircraft
and hands it to windtunnel-sim's D3Q19 solver as a library, which is
legitimate use rather than modification: the Solver already accepts an
obstacle_mask, so nothing in that repo has to change.

THE REYNOLDS COMPROMISE, stated plainly. A full aircraft at flight
Reynolds number is out of reach and not by a little:

    Re 90k, MAC-resolved     ~9.3 billion cells    ~1.4 TB
    Re 15k, MAC-resolved      ~45 million cells      ~7 GB   <- this

Six times too low. What that buys and what it costs is worth being
precise about: tip-vortex structure, spanwise load distribution and the
ORDER in which sections give up are governed by geometry and are only
weakly Reynolds-sensitive. Section drag and the exact stall angle are
strongly Reynolds-sensitive and should NOT be read off this. So the
number to take from a run here is where the flow separates first along
the span, not what the drag coefficient is.

    python scripts/tunnel3d.py --design out/trainer_v3_v50/design.json \
        --mission trainer_v3 --alpha 8 --re 15000
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from washout.geom import cst                                       # noqa: E402
from washout.search.design import Mission, build                   # noqa: E402

WSL_TUNNEL = "/mnt/c/Users/aipla/Desktop/windtunnel"
WSL_ROOT = "/mnt/c/Users/aipla/Desktop/washout"
WSL_PY = "~/wt-venv/bin/python"
NU_AIR = 1.5e-5


def voxelize(plan, nx, ny, nz, dx_m, origin, alpha_deg=0.0):
    """Boolean (nx, ny, nz) mask of the aircraft. Solver axes are
    (streamwise, vertical, spanwise); the aircraft's y is the span.

    Done a span slice at a time, and that is what makes it tractable. A
    general point-in-solid test over 45 million cells against a few
    hundred polygon edges is billions of operations; but at any span
    station the section is a GRAPH over the chord -- two single-valued
    surfaces -- so a cell is inside exactly when its height lies between
    the two, which is one interpolation per streamwise column.
    """
    mask = np.zeros((nx, ny, nz), dtype=bool)
    x0, y0, z0 = origin
    half = plan.half_span_m
    xs = x0 + (np.arange(nx) + 0.5) * dx_m           # streamwise, metres
    ys = y0 + (np.arange(ny) + 0.5) * dx_m           # vertical, metres
    a = np.radians(alpha_deg)
    ca, sa = np.cos(a), np.sin(a)

    for k in range(nz):
        span = z0 + (k + 0.5) * dx_m
        eta = abs(span) / half
        if eta > 1.0:
            continue
        st = plan.at(eta)
        loop = st.airfoil.coords(161)
        n = (len(loop) + 1) // 2
        up, lo = loop[:n][::-1], loop[n - 1:]        # LE -> TE
        # place the section: twist about quarter chord, scale, sweep
        def place(sec):
            p = sec - np.array([0.25, 0.0])
            t = np.radians(-st.twist_deg)
            ct, stt = np.cos(t), np.sin(t)
            r = np.stack([p[:, 0] * ct - p[:, 1] * stt,
                          p[:, 0] * stt + p[:, 1] * ct], 1)
            return (r * st.chord_m
                    + np.array([st.x_le_m + 0.25 * st.chord_m, st.z_le_m]))
        pu, pl = place(up), place(lo)
        # rotate the whole aircraft to angle of attack about the origin
        def rot(p):
            return np.stack([p[:, 0] * ca + p[:, 1] * sa,
                             -p[:, 0] * sa + p[:, 1] * ca], 1)
        pu, pl = rot(pu), rot(pl)

        lo_x, hi_x = pu[:, 0].min(), pu[:, 0].max()
        col = (xs >= lo_x) & (xs <= hi_x)
        if not col.any():
            continue
        xc = xs[col]
        zu = np.interp(xc, pu[:, 0], pu[:, 1], left=np.nan, right=np.nan)
        zl = np.interp(xc, pl[:, 0], pl[:, 1], left=np.nan, right=np.nan)
        good = ~(np.isnan(zu) | np.isnan(zl))
        if not good.any():
            continue
        inside = ((ys[None, :] >= np.minimum(zl, zu)[good][:, None])
                  & (ys[None, :] <= np.maximum(zl, zu)[good][:, None]))
        idx = np.flatnonzero(col)[good]
        mask[idx, :, k] = inside
    return mask


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", type=Path, required=True)
    ap.add_argument("--mission", default="trainer_v3")
    ap.add_argument("--airfoil", type=Path, default=ROOT / "assets/mh45.dat")
    ap.add_argument("--alpha", type=float, default=8.0)
    ap.add_argument("--re", type=float, default=1.5e4)
    ap.add_argument("--cells-per-mac", type=int, default=60, dest="cpm")
    ap.add_argument("--u-lat", type=float, default=0.10, dest="u_lat")
    ap.add_argument("--conv-times", type=float, default=18.0, dest="conv")
    ap.add_argument("--out", type=Path, default=ROOT / "out/tunnel3d")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    base = cst.load_selig(a.airfoil)
    mission = getattr(Mission, a.mission)()
    plan = build(np.array(json.loads(a.design.read_text())["u"]), mission, base)

    mac = plan.mac_m
    dx = mac / a.cpm
    tau = 3.0 * a.u_lat * a.cpm / a.re + 0.5
    if tau < 0.501:
        raise SystemExit(f"tau {tau:.5f} below the SGS floor -- raise "
                         f"cells-per-mac or u_lat")

    nx = int(round(7.0 * mac / dx))
    ny = int(round(4.5 * mac / dx))
    nz = int(round((plan.span_m + 1.2 * mac) / dx))
    cells = nx * ny * nz
    gb = cells * 19 * 4 * 2 / 1e9

    print(f"aircraft: span {plan.span_m*1000:.0f} mm, MAC {mac*1000:.1f} mm, "
          f"AR {plan.aspect_ratio:.2f}")
    print(f"lattice : {nx} x {ny} x {nz} = {cells/1e6:.1f} Mcells "
          f"({gb:.2f} GB f-populations)")
    print(f"physics : Re {a.re:.0f}, tau {tau:.5f}, u_lat {a.u_lat}, "
          f"dx {dx*1000:.2f} mm")
    steps = int(a.conv * a.cpm / a.u_lat)
    print(f"run     : {steps} steps ({a.conv:g} convective times)")
    if gb > 13.0:
        print("  WARNING: over the 16 GB card once fields and rendering "
              "buffers are added")

    origin = (-2.0 * mac, -2.25 * mac, -0.5 * (plan.span_m + 1.2 * mac))
    print("\nvoxelising...", flush=True)
    mask = voxelize(plan, nx, ny, nz, dx, origin, a.alpha)
    solid = int(mask.sum())
    print(f"  {solid:,} solid cells ({100*solid/cells:.3f}% of the domain)")
    vol_vox = solid * dx**3 * 1e6
    print(f"  voxel volume {vol_vox:.0f} cm^3 vs planform integral "
          f"{plan.volume_m3()*1e6:.0f} cm^3  "
          f"({100*vol_vox/(plan.volume_m3()*1e6):.0f}%)")

    a.out.mkdir(parents=True, exist_ok=True)
    npy = a.out / "mask.npy"
    np.save(npy, mask)
    meta = {"nx": nx, "ny": ny, "nz": nz, "tau": tau, "u_lat": a.u_lat,
            "steps": steps, "alpha": a.alpha, "re": a.re, "dx_m": dx,
            "mac_m": mac, "span_m": plan.span_m, "area_m2": plan.area_m2,
            "cells_per_mac": a.cpm}
    (a.out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"  mask -> {npy}")
    if a.dry_run:
        return 0

    runner = a.out / "_run3d.py"
    runner.write_text(f'''"""Generated by washout/scripts/tunnel3d.py -- drives lbm3d as a library."""
import json, sys, time
import numpy as np, torch
sys.path.insert(0, "{WSL_TUNNEL}")
from lbm3d.solver import Solver
from lbm3d import render as R

meta = json.load(open("{WSL_ROOT}/out/tunnel3d/meta.json"))
mask = torch.from_numpy(np.load("{WSL_ROOT}/out/tunnel3d/mask.npy"))
s = Solver(meta["nx"], meta["ny"], meta["nz"], meta["tau"], meta["u_lat"],
           obstacle_mask=mask, sgs=True, ramp_steps=2000,
           scene_name="washout_aircraft")
s.measure_force = True
q_dyn = 0.5 * meta["u_lat"] ** 2
area_lat = meta["area_m2"] / (meta["dx_m"] ** 2)
hist = []
t0 = time.perf_counter()
for i in range(meta["steps"]):
    s.step()
    if i % 200 == 0 and s.last_force is not None:
        f = s.last_force.cpu().numpy()
        hist.append([i, float(f[0]), float(f[1]), float(f[2])])
        if i % 2000 == 0:
            mlups = meta["nx"]*meta["ny"]*meta["nz"]*(i+1)/1e6/(time.perf_counter()-t0)
            print(f"  step {{i}}  {{mlups:.0f}} MLUPS  "
                  f"CD {{f[0]/(q_dyn*area_lat):+.4f}}  "
                  f"CL {{f[1]/(q_dyn*area_lat):+.4f}}", flush=True)
np.savetxt("{WSL_ROOT}/out/tunnel3d/forces3d.csv", np.array(hist),
           delimiter=",", header="step,fx,fy,fz", comments="")
tail = np.array(hist)[len(hist)//2:]
print("MEAN  CD %+.5f  CL %+.5f  Cy %+.5f" % (
    tail[:,1].mean()/(q_dyn*area_lat), tail[:,2].mean()/(q_dyn*area_lat),
    tail[:,3].mean()/(q_dyn*area_lat)))
torch.save(s.f.cpu(), "{WSL_ROOT}/out/tunnel3d/final.pt")
''', encoding="utf-8")

    cmd = f"cd {WSL_TUNNEL} && {WSL_PY} {WSL_ROOT}/out/tunnel3d/_run3d.py"
    print(f"\nrunning on the GPU...\n")
    p = subprocess.run(["wsl", "-d", "Ubuntu", "-e", "bash", "-lc", cmd],
                       capture_output=True, text=True, timeout=36000)
    print(p.stdout[-3000:] if p.returncode == 0 else p.stderr[-3000:])
    return 0 if p.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
