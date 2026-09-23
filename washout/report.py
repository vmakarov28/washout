"""One figure per design: planform, front view, 3D, sections, loading, print.

The numbers in design.json say whether it flies. This says whether it
looks like an aeroplane -- which catches a different class of mistake,
and catches it in one glance. A planform with the CG behind the neutral
point, a span loading spiking at the tip, or a print stack whose layers
march sideways off the bed are all obvious here and all invisible in a
table of coefficients.

The front and 3D views were added after a whole generation shipped with
winglets nobody could see: a top view is blind to dihedral, so a 200 mm
upturned tip and a flat wing drew identically.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .aero.vlm import VLM
from .geom import fairness as fz
from .geom.planform import Planform
from .printing import vase

INK = "#1b1b1b"
ACCENT = "#c0392b"
COOL = "#2c6fbb"
SKIN = "#dfe6ee"


def figure(ev, settings: vase.PrintSettings, path: str | Path,
           title: str = "washout") -> Path:
    """Draw the design sheet. NEVER raises.

    do_export draws this BEFORE it writes the STL parts, and it used to
    catch only ImportError -- so any plotting bug at the end of a two-hour
    search would have taken the printable parts down with it. The figure
    is a picture of the result, not the result. If the 3D panel fails the
    sheet is redrawn without it; if that fails too, the export carries on
    without a figure."""
    path = Path(path)
    for with_3d in (True, False):
        try:
            return _figure(ev, settings, path, title, with_3d)
        except Exception as e:                      # noqa: BLE001
            plt.close("all")
            print(f"  figure: {'3D view' if with_3d else 'sheet'} failed "
                  f"({type(e).__name__}: {e})"
                  + ("; redrawing without it" if with_3d else "; skipped"))
    return path


def _front_outline(plan: Planform, n: int = 90):
    """Upper and lower extent of the section at each span station, in the
    aircraft frame: the silhouette seen from dead ahead."""
    eta = np.linspace(0.0, 1.0, n)
    up, lo = [], []
    for e in eta:
        p = plan.section_3d(float(e), 81)
        up.append(p[:, 2].max())
        lo.append(p[:, 2].min())
    return (eta * plan.half_span_m * 1000, np.array(up) * 1000,
            np.array(lo) * 1000)


def _draw_3d(ax, plan: Planform, fins=None) -> None:
    """Both halves as a shaded surface at TRUE scale.

    The first version set a box aspect from the data ranges but left the
    axis limits to autoscale, so the box and the data disagreed and the
    aircraft came out as a dark sliver in a mostly empty panel. Limits and
    box are now set together from one set of numbers, with a floor on the
    vertical extent so a flat wing is not drawn as a line."""
    ne, nc = 30, 37
    grid = np.array([plan.section_3d(float(e), nc)
                     for e in np.linspace(0.0, 1.0, ne)]) * 1000   # (ne,nc,3)
    X, Y, Z = grid[:, :, 0], grid[:, :, 1], grid[:, :, 2]
    for sign in (1, -1):
        ax.plot_surface(sign * Y, X, Z, color="#b9c9da", linewidth=0,
                        antialiased=False, shade=True)
    z_lo, z_hi = float(Z.min()), float(Z.max())
    if fins is not None:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        o = fins.outline() * 1000
        z_lo, z_hi = min(z_lo, float(o[:, 1].min())), max(z_hi, float(o[:, 1].max()))
        for sign in (1, -1):
            yv = np.full(len(o), sign * fins.y_m * 1000)
            ax.add_collection3d(Poly3DCollection(
                [list(zip(yv, o[:, 0], o[:, 1]))], facecolor="#5d7189",
                edgecolor="#1b1b1b", linewidths=0.5))
    # 6% of span either side, so a tip fin is not clipped at the limit
    half = 1.06 * float(Y.max())
    x0, x1 = float(X.min()), float(X.max())
    if fins is not None:
        x0, x1 = min(x0, float(o[:, 0].min())), max(x1, float(o[:, 0].max()))
    zc = 0.5 * (z_lo + z_hi)
    h = max(z_hi - z_lo, 0.12 * 2.0 * half)
    ax.set_xlim(-half, half)
    ax.set_ylim(x0, x1)
    ax.set_zlim(zc - 0.5 * h, zc + 0.5 * h)
    ax.set_box_aspect((2.0 * half, x1 - x0, h))
    # Orthographic, from ahead of the nose and above. Chosen by rendering
    # candidates, not guessed: in perspective every oblique camera
    # foreshortened the near half of a 46 degree swept wing into a tall
    # wedge, so a symmetric aircraft drew as one fin and one wing, and
    # zoom 1.3 clipped it besides. Orthographic removes the distortion,
    # and a camera on the centreline keeps the halves visibly mirror
    # images -- the property this panel exists to show.
    #
    # EXCEPT when there are tip fins. They stand in the fore-aft plane, so
    # from dead ahead they are exactly edge-on: rendered, the first finned
    # sheet showed two hairlines where 109 mm plates should be. Twenty
    # degrees off the centreline shows them as plates and costs only a
    # slight lean, which is the better trade on the one kind of design
    # where there is something at the tips to see.
    ax.set_proj_type("ortho")
    if fins is not None:
        ax.view_init(elev=35, azim=-75)
    else:
        ax.view_init(elev=40, azim=-90)
    ax.set_axis_off()


def _figure(ev, settings: vase.PrintSettings, path: Path, title: str,
            with_3d: bool) -> Path:
    plan: Planform = ev.plan
    fig = plt.figure(figsize=(14.5, 12.5), facecolor="white")
    gs = fig.add_gridspec(3, 3, height_ratios=[1.15, 0.85, 1.0],
                          hspace=0.34, wspace=0.26)

    # ---------------------------------------------------- planform, top view
    ax = fig.add_subplot(gs[0, :2])
    etas = np.linspace(0, 1, 160)
    le = np.array([[plan.at(e).x_le_m, e * plan.half_span_m] for e in etas])
    te = np.array([[plan.at(e).x_le_m + plan.at(e).chord_m, e * plan.half_span_m]
                   for e in etas])
    outline = np.vstack([le, te[::-1]])
    for sign in (1, -1):
        ax.fill(outline[:, 1] * sign * 1000, outline[:, 0] * 1000,
                color=SKIN, edgecolor=INK, lw=1.2, zorder=2)

    joins = vase.panel_etas(plan, settings)
    for a, b in joins:
        for sign in (1, -1):
            y = b * plan.half_span_m * sign * 1000
            st = plan.at(b)
            ax.plot([y, y], [st.x_le_m * 1000, (st.x_le_m + st.chord_m) * 1000],
                    color=ACCENT, lw=1.4, ls="--", zorder=3)

    fins = getattr(ev, "fins", None)
    if fins is not None:                    # a vertical plate, seen edge-on
        o = fins.outline()
        for sign in (1, -1):
            y = sign * fins.y_m * 1000
            ax.plot([y, y], [o[:, 0].min() * 1000, o[:, 0].max() * 1000],
                    color=INK, lw=3.2, solid_capstyle="butt", zorder=5)

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
    lat = getattr(ev, "lateral", None)
    dynm = getattr(ev, "dynamics", None)
    fair = getattr(ev, "fairness", None) or fz.measure(plan)
    lim = getattr(ev, "fairness_limits", None) or fz.Limits()
    unfair = fair.violations(lim)
    rows = [
        ("span", f"{plan.span_m*1000:.0f} mm"),
        ("area", f"{plan.area_m2*1e4:.0f} cm^2"),
        ("aspect ratio", f"{plan.aspect_ratio:.2f}"),
        ("sweep c/4", f"{plan.sweep_quarter_chord_deg():.1f} deg"),
        ("mass", f"{ev.mass_kg*1000:.0f} g"),
        # area_m2 * 100 is dm^2. The first version divided grams by m^2
        # and labelled it g/dm^2, so every aircraft read 100x too heavy.
        ("wing loading",
         f"{ev.mass_kg*1000/(plan.area_m2*100):.1f} g/dm^2"
         if plan.area_m2 else "-"),
        ("trim alpha", f"{t.alpha_deg:+.2f} deg" if t else "-"),
        ("CL trim", f"{ev.cl_trim:.3f}"),
        ("cruise", f"{ev.v_cruise:.1f} m/s"),
        ("L/D", f"{ev.ld:.2f}"),
        ("CD0 / CDi", f"{t.CD0:.4f} / {t.CDi:.4f}" if t else "-"),
        ("static margin", f"{ev.static_margin:+.3f}"),
        ("Cn_beta", f"{lat.cn_beta:+.4f} /rad" if lat else "-"),
        ("Dutch roll zeta",
         "-" if dynm is None else
         ("overdamped" if dynm.zeta_dr is None else f"{dynm.zeta_dr:+.3f}")),
        ("tip fins",
         "none" if fins is None else
         f"2x {fins.area_m2*1e4:.0f} cm^2, {fins.height_m*1000:.0f} mm tall"),
        ("tip rise", f"{100*fair.tip_rise_frac:.0f}% semi-span"),
        ("root t/c", f"{fair.root_t_over_c:.3f}"),
        ("LE / TE turns", f"{fair.le_inflections} / {fair.te_inflections}"),
        ("surface", "fair" if not unfair else "UNFAIR"),
    ]
    y = 0.99
    for k, v in rows:
        ax.text(0.0, y, k, fontsize=9, color="#5a5a5a", va="top")
        ax.text(1.0, y, v, fontsize=9, color=INK, va="top", ha="right",
                fontweight="bold")
        y -= 0.046
    ax.text(0.0, y - 0.005, "OK" if ev.ok else "INFEASIBLE", fontsize=11,
            color="#1e8449" if ev.ok else ACCENT, fontweight="bold", va="top")
    if not ev.ok:
        # capped: an uncapped list ran straight down into the panel below
        shown = [r[:58] for r in ev.reasons[:3]]
        more = len(ev.reasons) - len(shown)
        if more > 0:
            shown.append(f"... and {more} more")
        ax.text(0.0, y - 0.065, "\n".join(shown), fontsize=7, color=ACCENT,
                va="top")

    # ---------------------------------------------------------- front view
    ax = fig.add_subplot(gs[1, :2] if with_3d else gs[1, :])
    yy, up, lo = _front_outline(plan)
    for sign in (1, -1):
        ax.fill(np.concatenate([sign * yy, sign * yy[::-1]]),
                np.concatenate([up, lo[::-1]]),
                color=SKIN, edgecolor=INK, lw=1.1, zorder=2)
    for a, b in joins:
        yj = b * plan.half_span_m * 1000
        k = int(np.argmin(np.abs(yy - yj)))
        for sign in (1, -1):
            ax.plot([sign * yy[k]] * 2, [lo[k], up[k]], color=ACCENT, lw=1.2,
                    ls="--", zorder=3)
    ax.set_aspect("equal")
    ax.set_xlabel("span [mm]")
    ax.set_ylabel("z, up [mm]")
    if fins is not None:
        o = fins.outline()
        for sign in (1, -1):
            yf = sign * fins.y_m * 1000
            ax.plot([yf, yf], [o[:, 1].min() * 1000, o[:, 1].max() * 1000],
                    color=INK, lw=3.2, solid_capstyle="butt", zorder=5)
    ax.set_title("front view  --  dihedral, winglets, thickness", loc="left",
                 fontsize=10)
    ax.grid(alpha=0.25)

    # ------------------------------------------------------------- 3D view
    if with_3d:
        ax = fig.add_subplot(gs[1, 2], projection="3d")
        _draw_3d(ax, plan, getattr(ev, "fins", None))

    # -------------------------------------------------------- section stack
    ax = fig.add_subplot(gs[2, 0])
    for e, lab in ((0.0, "root"), (0.5, "mid"), (1.0, "tip")):
        st = plan.at(e)
        p = st.airfoil.coords(160) * st.chord_m * 1000
        ax.plot(p[:, 0], p[:, 1], lw=1.3, label=f"{lab}  c={st.chord_m*1000:.0f} mm")
    ax.set_aspect("equal")
    ax.set_title("sections, true scale [mm]", loc="left", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    # --------------------------------------------------------- span loading
    ax = fig.add_subplot(gs[2, 1])
    if t:
        v = VLM(plan, ns=30, nc=6, fins=fins)
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
    ax = fig.add_subplot(gs[2, 2])
    panels = ev.panels or vase.build_panels(plan, settings, z_step_mm=2.0)
    pan = panels[-1]
    step = max(len(pan.contours) // 26, 1)
    n = len(pan.contours[::step])
    for i, c in enumerate(pan.contours[::step]):
        ax.plot(c[:, 0], c[:, 1], lw=0.7,
                color=plt.cm.viridis(i / max(n - 1, 1)), alpha=0.9)
    deg, w, h = pan.best_bed_rotation()
    ax.set_aspect("equal")
    ax.set_title(f"{pan.name}: {len(pan.contours)} layers, "
                 f"{pan.height_mm:.0f} mm tall\n"
                 f"{w:.0f}x{h:.0f} mm at {deg:.0f} deg on the bed",
                 loc="left", fontsize=9)
    ax.grid(alpha=0.25)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path
