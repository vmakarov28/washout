"""One figure per design: planform, sections, span loading, print stack.

The numbers in design.json say whether it flies. This says whether it
looks like an aeroplane -- which catches a different class of mistake,
and catches it in one glance. A planform with the CG behind the neutral
point, a span loading spiking at the tip, or a print stack whose layers
march sideways off the bed are all obvious here and all invisible in a
table of coefficients.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .aero.vlm import VLM
from .geom.planform import Planform
from .printing import vase

INK = "#1b1b1b"
ACCENT = "#c0392b"
COOL = "#2c6fbb"


def figure(ev, settings: vase.PrintSettings, path: str | Path,
           title: str = "planeforge") -> Path:
    plan: Planform = ev.plan
    fig = plt.figure(figsize=(13.5, 8.5), facecolor="white")
    gs = fig.add_gridspec(2, 3, hspace=0.30, wspace=0.26)

    # ---------------------------------------------------- planform, top view
    ax = fig.add_subplot(gs[0, :2])
    etas = np.linspace(0, 1, 160)
    le = np.array([[plan.at(e).x_le_m, e * plan.half_span_m] for e in etas])
    te = np.array([[plan.at(e).x_le_m + plan.at(e).chord_m, e * plan.half_span_m]
                   for e in etas])
    outline = np.vstack([le, te[::-1]])
    for sign in (1, -1):
        ax.fill(outline[:, 1] * sign * 1000, outline[:, 0] * 1000,
                color="#dfe6ee", edgecolor=INK, lw=1.2, zorder=2)

    for a, b in vase.panel_etas(plan, settings):
        for sign in (1, -1):
            y = b * plan.half_span_m * sign * 1000
            st = plan.at(b)
            ax.plot([y, y], [st.x_le_m * 1000, (st.x_le_m + st.chord_m) * 1000],
                    color=ACCENT, lw=1.4, ls="--", zorder=3)

    mac_x = plan.x_mac_le_m * 1000
    ax.plot([0, 0], [mac_x, mac_x + plan.mac_m * 1000], color=COOL, lw=2.5,
            zorder=4, label=f"MAC {plan.mac_m*1000:.0f} mm")
    if ev.trim:
        ax.scatter([0], [ev.trim.x_np_m * 1000], marker="o", s=70,
                   color=COOL, zorder=6, label="neutral point")
        ax.scatter([0], [ev.trim.x_cg_m * 1000], marker="v", s=90,
                   color=ACCENT, zorder=6,
                   label=f"CG (SM {ev.static_margin:+.3f})")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("span [mm]")
    ax.set_ylabel("x, aft [mm]")
    ax.set_title(f"{title}  --  dashed = printed panel joins", loc="left")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.25)

    # ------------------------------------------------------------- summary
    ax = fig.add_subplot(gs[0, 2])
    ax.axis("off")
    t = ev.trim
    rows = [
        ("span", f"{plan.span_m*1000:.0f} mm"),
        ("area", f"{plan.area_m2*1e4:.0f} cm^2"),
        ("aspect ratio", f"{plan.aspect_ratio:.2f}"),
        ("sweep c/4", f"{plan.sweep_quarter_chord_deg():.1f} deg"),
        ("mass", f"{ev.mass_kg*1000:.0f} g"),
        ("wing loading",
         f"{ev.mass_kg*1000/plan.area_m2:.1f} g/dm^2"
         if plan.area_m2 else "-"),
        ("", ""),
        ("trim alpha", f"{t.alpha_deg:+.2f} deg" if t else "-"),
        ("CL trim", f"{ev.cl_trim:.3f}"),
        ("cruise", f"{ev.v_cruise:.1f} m/s"),
        ("L/D", f"{ev.ld:.2f}"),
        ("CD0 / CDi", f"{t.CD0:.4f} / {t.CDi:.4f}" if t else "-"),
        ("static margin", f"{ev.static_margin:+.3f}"),
    ]
    y = 0.98
    for k, v in rows:
        if k:
            ax.text(0.0, y, k, fontsize=9, color="#5a5a5a", va="top")
            ax.text(1.0, y, v, fontsize=9, color=INK, va="top", ha="right",
                    fontweight="bold")
        y -= 0.072
    ax.text(0.0, y - 0.02, "OK" if ev.ok else "INFEASIBLE", fontsize=11,
            color="#1e8449" if ev.ok else ACCENT, fontweight="bold", va="top")
    if not ev.ok:
        ax.text(0.0, y - 0.09, "\n".join(ev.reasons), fontsize=7,
                color=ACCENT, va="top", wrap=True)

    # -------------------------------------------------------- section stack
    ax = fig.add_subplot(gs[1, 0])
    for e, lab in ((0.0, "root"), (0.5, "mid"), (1.0, "tip")):
        st = plan.at(e)
        p = st.airfoil.coords(160) * st.chord_m * 1000
        ax.plot(p[:, 0], p[:, 1], lw=1.3, label=f"{lab}  c={st.chord_m*1000:.0f} mm")
    ax.set_aspect("equal")
    ax.set_title("sections, true scale [mm]", loc="left", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    # --------------------------------------------------------- span loading
    ax = fig.add_subplot(gs[1, 1])
    if t:
        v = VLM(plan, ns=30, nc=6)
        pt = v.solve(t.alpha_deg, t.x_cg_m)
        order = np.argsort(pt.y_strip)
        y = pt.y_strip[order] / plan.half_span_m
        cl = pt.cl_local[order]
        ax.plot(y, cl, color=COOL, lw=1.8, label="local cl")
        ell = pt.CL * 4 / np.pi * np.sqrt(np.maximum(1 - y**2, 0))
        ax.plot(y, ell * np.trapezoid(cl, y) / max(np.trapezoid(ell, y), 1e-9),
                color="#999", lw=1.0, ls="--", label="elliptic")
        ax.axhline(1.05, color=ACCENT, lw=1.0, ls=":", label="cl_max")
        ax.set_ylim(0, max(1.25, float(np.nanmax(cl)) * 1.15))
        ax.legend(fontsize=7)
    ax.set_xlabel("y / semi-span")
    ax.set_title(f"span loading at trim (e = {pt.e_oswald:.3f})" if t
                 else "span loading", loc="left", fontsize=10)
    ax.grid(alpha=0.25)

    # ---------------------------------------------------------- print stack
    ax = fig.add_subplot(gs[1, 2])
    panels = ev.panels or vase.build_panels(plan, settings, z_step_mm=2.0)
    pan = panels[-1]
    step = max(len(pan.contours) // 26, 1)
    n = len(pan.contours[::step])
    for i, c in enumerate(pan.contours[::step]):
        ax.plot(c[:, 0], c[:, 1], lw=0.7,
                color=plt.cm.viridis(i / max(n - 1, 1)), alpha=0.9)
    bx, by = settings.bed_x_mm, settings.bed_y_mm
    deg, w, h = pan.best_bed_rotation()
    ax.set_aspect("equal")
    ax.set_title(f"{pan.name}: {len(pan.contours)} layers, "
                 f"{pan.height_mm:.0f} mm tall\n"
                 f"{w:.0f}x{h:.0f} mm at {deg:.0f} deg on the bed",
                 loc="left", fontsize=9)
    ax.grid(alpha=0.25)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path
