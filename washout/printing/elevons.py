"""The elevon as its own printed part, and the pocket it hinges in.

The realisation that makes control surfaces tractable: **the hinge line
runs spanwise, and print Z is the span.** So an elevon is a vase-mode
part in exactly the same orientation as the wing panel it came off -- a
small tapered wedge, printed root-down, one closed contour per layer. It
needs no new print mode and no supports.

What it does need is for the cut to land on a panel joint. The elevon
begins at a spanwise station, so the wing's trailing edge has to
disappear there; and a surface normal to the span is a ROOF in this
orientation, which spiralize cannot print. Truncating half way up a
panel is therefore impossible. `vase.panel_etas` is told to break at
`elevon_eta`, and then every panel is either wholly plain or wholly
truncated and the transition is a print joint rather than a roof.

Three parts come out of one chord station:

    wing          upper surface from the hinge line forward, round the
                  leading edge, back along the lower surface to the hinge
                  line, closed by the cut face. Still one simple loop,
                  still 2n-1 points, still paired upper-to-lower at
                  matching x -- so the rib inserter and the wall
                  separation gate keep working untouched.

    elevon        the section aft of the hinge line, with its nose
                  CHAMFERED so it can deflect without striking the cut
                  face. The chamfer angle is the deflection the mission
                  asks for, plus margin: it is not a styling choice, it
                  is what `max_elevon_deflect_deg` costs in geometry.

    the gap       `hinge_gap_mm` between them. A top-surface tape hinge
                  needs the upper corners nearly touching and the lower
                  ones open, which is exactly what a chamfered elevon
                  nose against a square cut face gives.

## Why the chamfer is computed and not chosen

Hinge the elevon at its upper surface and deflect it down by delta. The
nose's lower corner, a depth t below the axis, swings forward by
t*sin(delta). On the trainer's inboard elevon t is 10.8 mm and delta is
12 degrees, so the corner wants 2.25 mm of wing that is still there. Open
the gap that far and there is a 2.25 mm slot in the aerofoil at every
station; chamfer the nose instead and the slot stays one bead wide. The
chamfer is therefore sized from the deflection, and the residual gap is
gated rather than hoped for.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .bays import BaySpec, POINTS_PER_BAY, insert_detours
from .vase import (LayerStack, PrintSettings, arc_length_mm, skin_drift_abs,
                   solve_ramp_mm, thicken_for_nozzle)


def hinge_x(settings: PrintSettings) -> float:
    """Chord fraction of the hinge line."""
    return float(np.clip(1.0 - settings.elevon_chord, 0.05, 0.98))


def has_elevon(settings: PrintSettings) -> bool:
    return settings.elevon_chord > 1e-6 and settings.elevon_eta < 1.0


def truncate_loop(loop_unit: np.ndarray, x_hinge: float) -> np.ndarray:
    """The wing's loop with everything aft of the hinge line removed.

    The point count and the upper/lower pairing are PRESERVED: the same
    number of stations, redistributed over the shorter chord, upper
    surface from `x_hinge` to 0 and lower from 0 to `x_hinge`. Those two
    invariants are load-bearing -- the STL skinner joins layer k index i
    to layer k+1 index i, and `wall_separation_mm` pairs index i against
    its mirror -- so a truncation that dropped points would shear the
    mesh and blind the thickness gate at once.

    The cut face is not a set of vertices. It is the closing segment of
    the loop, from the last lower point back to the first upper point,
    which a closed contour already has. Adding a ladder of points across
    it would only give the slicer somewhere to round off.
    """
    n = (len(loop_unit) + 1) // 2
    upper = loop_unit[:n][::-1]                  # LE -> TE
    lower = loop_unit[n - 1:]                    # LE -> TE
    xs_old = upper[:, 0]
    # cosine spacing again over the new chord, so the leading edge stays
    # dense where the curvature is
    t = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n)))
    xs = x_hinge * t
    up = np.stack([xs, np.interp(xs, xs_old, upper[:, 1])], 1)
    lo = np.stack([xs, np.interp(xs, xs_old, lower[:, 1])], 1)
    return np.concatenate([up[::-1], lo[1:]], 0)


_MIN_TE_MM = 1.0


def settings_min_te_frac(chord_mm: float, min_te_mm: float = _MIN_TE_MM) -> float:
    """The nose's residual flat, as a chord fraction. A bevel that runs
    all the way to the hinge axis is a feather edge, thinner than the
    nozzle, and vase mode would cross its own bead."""
    return float(min_te_mm / max(chord_mm, 1e-9))


def elevon_loop(loop_unit: np.ndarray, x_hinge: float, chord_mm: float,
                gap_mm: float, chamfer_deg: float, n: int) -> np.ndarray:
    """The section aft of the hinge line, nose chamfered for deflection.

    Hinged at the UPPER surface -- which is where a tape hinge goes and
    where a pinned eye would sit -- so the chamfer is cut from the upper
    nose corner downward and aft at `chamfer_deg`. Deflecting the elevon
    down then rotates the chamfer face toward the wing's cut face and
    they stay parallel instead of colliding.
    """
    m = (len(loop_unit) + 1) // 2
    upper = loop_unit[:m][::-1]
    lower = loop_unit[m - 1:]
    xs_old = upper[:, 0]

    x0 = x_hinge + gap_mm / max(chord_mm, 1e-9)
    t = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n)))
    xs = x0 + (1.0 - x0) * t
    up = np.stack([xs, np.interp(xs, xs_old, upper[:, 1])], 1)
    lo = np.stack([xs, np.interp(xs, xs_old, lower[:, 1])], 1)

    # --- the bevel, derived rather than chosen ---
    #
    # Hinge axis at the upper surface, x = x0. A nose-face point a depth d
    # below the axis on a face bevelled beta from vertical sits at
    # (d tan(beta), -d). Rotating the elevon trailing-edge-DOWN by delta
    # takes it to
    #
    #     x' = d (tan(beta) cos(delta) - sin(delta))
    #
    # and the point stays out of the wing while x' >= 0, which reduces to
    #
    #     tan(beta) >= tan(delta)   =>   beta >= delta
    #
    # So a bevel at the deflection angle exactly suffices, and the margin
    # in `hinge_margin_deg` is what makes it reach full travel rather than
    # arrive at the geometry's limit. A bevel that meets the axis exactly
    # would come to a feather edge, so the nose keeps `min_te_mm` of
    # thickness and the bevel runs aft from there: that residual corner is
    # only 1 mm deep, so it swings 1 mm * sin(beta) -- under 0.3 mm --
    # which the hinge gap covers.
    y_axis = float(up[0, 1])
    tb = max(np.tan(np.radians(chamfer_deg)), 1e-3)
    y_floor = y_axis - (xs - x0) / tb
    t_min = settings_min_te_frac(chord_mm)
    lo[:, 1] = np.minimum(np.maximum(lo[:, 1], y_floor), up[:, 1] - t_min)

    # TWO blunt faces and no shared vertex. The wing's loop shares its
    # leading-edge point because the surfaces genuinely meet there; an
    # elevon's nose is a cut and they do not, so sharing a vertex would
    # put a feather edge on the part -- which the first version of this
    # function duly did, and the thickness gate read 0.039 mm. The nose
    # face is the closing segment between index n-1 and n; the trailing
    # edge is the one between 2n-1 and 0.
    return np.concatenate([up[::-1], lo], 0)


def nose_swing_mm(depth_mm: float, deflect_deg: float) -> float:
    """How far the un-chamfered lower nose corner swings into the wing."""
    return float(depth_mm * np.sin(np.radians(abs(deflect_deg))))


def build_elevon(plan, settings: PrintSettings, eta0: float, eta1: float,
                 name: str, chamfer_deg: float,
                 z_step_mm: float | None = None, sockets=()) -> LayerStack:
    """One elevon panel, printed root-down like the wing it came off.

    `sockets` are `BaySpec`s in this panel's own z: the horn's socket, a
    pocket in the UPPER skin cut by the same detour machinery as a bay,
    so the tongue is held in bearing by the skin and the glue only keeps
    it there. It ramps in and out like everything else."""
    tip, root = plan.at(eta1), plan.at(eta0)
    dy = (eta1 - eta0) * plan.half_span_m
    dz = tip.z_le_m - root.z_le_m
    panel_len_mm = float(np.hypot(dy, dz)) * 1000.0

    step = z_step_mm or settings.layer_h_mm
    n_layers = max(int(round(panel_len_mm / step)), 2)
    z = np.arange(n_layers) * step
    z = z[z <= panel_len_mm + 1e-9]
    eta = eta0 + (z / panel_len_mm) * (eta1 - eta0)

    sockets = tuple(sockets)
    n_pts = 2 * settings.contour_points + POINTS_PER_BAY * len(sockets)
    contours = np.empty((len(z), n_pts, 2))
    xh = hinge_x(settings)
    for k, e in enumerate(eta):
        st = plan.at(float(e))
        chord_mm = st.chord_m * 1000.0
        loop = thicken_for_nozzle(st.airfoil.coords(settings.contour_points),
                                  chord_mm, settings)
        loop = elevon_loop(loop, xh, chord_mm, settings.hinge_gap_mm,
                           chamfer_deg, settings.contour_points)
        if sockets:
            loop = insert_detours(loop, chord_mm, float(z[k]), sockets, None,
                                  min_groove_mm=settings.extrusion_width_mm,
                                  x_le_mm=st.x_le_m * 1000.0)
        p = loop - np.array([0.25, 0.0])
        a = np.radians(-st.twist_deg)
        ca, sa = np.cos(a), np.sin(a)
        rot = np.stack([p[:, 0] * ca - p[:, 1] * sa,
                        p[:, 0] * sa + p[:, 1] * ca], 1)
        contours[k] = rot * chord_mm + np.array(
            [st.x_le_m * 1000.0 + 0.25 * chord_mm, 0.0])

    flat = contours.reshape(-1, 2)
    origin = 0.5 * (flat.min(0) + flat.max(0))
    contours -= origin
    return LayerStack(z_mm=z, eta=eta, contours=contours, settings=settings,
                      name=name, z_step_mm=step, has_ribs=False,
                      has_cuts=bool(sockets), bay_specs=sockets,
                      origin_mm=(float(origin[0]), float(origin[1])),
                      role="elevon",
                      n_upper=settings.contour_points + POINTS_PER_BAY * len(sockets))


SOCKET_RUN_CLEAR_MM = 0.15
"""Fit clearance each side of the horn's tongue in the socket's
full-depth run, so the socket is the tongue's thickness plus twice this
along the span."""


def socket_for(plan, settings: PrintSettings, panel_etas, eta_h: float,
               x_abs_mm: float, length_mm: float, depth_mm: float,
               tongue_t_mm: float, chamfer_deg: float, margin: float,
               z_step_mm: float | None = None):
    """The horn's socket as a `BaySpec` in the z of the elevon panel that
    holds station `eta_h`. -> (panel index among the elevons, spec) or
    (None, None) if no elevon holds that station or the socket cannot
    ramp within the panel.

    The ramps are measured on the BARE elevon at the socket's absolute
    station, on the upper skin, with the same drift profile and the same
    solve a bay gets. A socket whose ramp would run off the panel's end
    is reported as unmakeable rather than cut short."""
    if not has_elevon(settings):
        return None, None
    lim = np.tan(np.radians(settings.max_overhang_deg))
    j = 0
    for a, b in panel_etas:
        if a < settings.elevon_eta - 1e-9:
            continue
        if a - 1e-9 <= eta_h <= b + 1e-9:
            bare = build_elevon(plan, settings, a, b, "bare", chamfer_deg,
                                z_step_mm=z_step_mm)
            z_h = arc_length_mm(plan, a, eta_h)
            run = tongue_t_mm + 2.0 * SOCKET_RUN_CLEAR_MM
            zs, drift = skin_drift_abs(bare, x_abs_mm - 0.5 * length_mm,
                                       x_abs_mm + 0.5 * length_mm, side="upper")
            ramps = []
            for sign in (+1.0, -1.0):
                edge = z_h + sign * 0.5 * run
                sel = (zs - edge) * sign > 0.0
                d = np.abs(zs[sel] - edge)
                order = np.argsort(d)
                ramps.append(solve_ramp_mm(d[order], drift[sel][order],
                                           depth_mm, lim, margin))
            ramp_out, ramp_in = ramps
            if not (np.isfinite(ramp_out) and np.isfinite(ramp_in)):
                return j, None
            spec = BaySpec("horn socket", 0.0, 0.0,
                           z_h - 0.5 * run, z_h + 0.5 * run, ramp_out, depth_mm,
                           settings.extrusion_width_mm, open_from="upper",
                           ledge_mm=0.0, ramp_in_mm=ramp_in,
                           length_mm=length_mm, x_abs_mm=x_abs_mm)
            return j, spec
        j += 1
    return None, None


def build_elevons(plan, settings: PrintSettings, panel_etas,
                  chamfer_deg: float,
                  z_step_mm: float | None = None,
                  sockets: dict | None = None) -> list[LayerStack]:
    """Every elevon part, split at the same joints as the wing panels.

    Sharing the wing's joint stations is not tidiness: each elevon panel
    then sits against exactly one wing panel, one hinge rod or one strip
    of tape serves both, and the parts cannot be assembled in the wrong
    order.

    `sockets` maps an elevon index to the `BaySpec`s cut into it."""
    if not has_elevon(settings):
        return []
    sockets = sockets or {}
    out = []
    j = 0
    for a, b in panel_etas:
        if a < settings.elevon_eta - 1e-9:
            continue
        out.append(build_elevon(plan, settings, a, b,
                                f"{plan.name}_elevon{j}", chamfer_deg,
                                z_step_mm=z_step_mm,
                                sockets=tuple(sockets.get(j, ()))))
        j += 1
    return out


def hinge_report(plan, settings: PrintSettings, deflect_deg: float) -> str:
    xh = hinge_x(settings)
    rows = ["elevons:"]
    for e in (settings.elevon_eta, 0.5 * (settings.elevon_eta + 1.0), 0.98):
        st = plan.at(float(np.clip(e, 0.0, 1.0)))
        c_mm = st.chord_m * 1000.0
        t = float(st.airfoil.thickness(np.array([xh]))[0]) * c_mm
        rows.append(f"  eta {e:.2f}: chord {c_mm:5.1f} mm | hinge at "
                    f"{xh:.2f}c | section {t:5.2f} mm | elevon chord "
                    f"{settings.elevon_chord * c_mm:5.1f} mm | nose swings "
                    f"{nose_swing_mm(t, deflect_deg):4.2f} mm at "
                    f"{deflect_deg:.0f} deg")
    rows.append(f"  hinge: {settings.hinge_gap_mm:.1f} mm gap, nose chamfered "
                f"{deflect_deg + settings.hinge_margin_deg:.0f} deg, "
                f"tape the UPPER surface")
    return "\n".join(rows)
