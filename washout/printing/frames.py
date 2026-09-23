"""Where each printed panel sits in the aircraft, and how a layer is cut.

## What was wrong

A printed layer used to be the loft's section at one span station --
the `y = const` section, placed with its leading-edge x, its chord and
its twist -- stacked straight up the print Z axis. The station's `z_le`
was never used. So every panel printed STRAIGHT, and the dihedral the
lattice scored as a smooth curve arrived as a polyline with a kink at
every joint and nothing to say how big: 4.1 mm off the loft on micro_fpv
with a panel per control station, 9.0 mm with the split kept minimal,
and joint kinks up to 21 degrees that no build sheet mentioned.

And the sections were the wrong sections. A `y = const` cut through a
wing with dihedral delta is 1/cos(delta) thicker than the cut square to
the wing, so a layer that IS the `y = const` section, stood square to a
tilted panel, prints a wing 1/cos(phi) too thick when it is assembled:
+13% on the trainer's 28 degree tip panel.

## The construction

Each panel has an AXIS in the span-height plane, at angle `phi` above the
span direction, and a layer at print height `s` is the loft cut by the
plane square to that axis at distance `s` from the panel's root pivot.
The panel therefore follows the dihedral curve inside itself -- the
curve's departure from the axis is a lean in print Y, and the overhang
gate charges it by the same quadrature law it charges sweep with -- and
its sections are the true square sections.

The CENTRE BODY's axis is horizontal. Its root face is then the symmetry
plane itself, and the two halves mate flat. Any other angle would leave
them meeting in a V.

The other panels' axes run between JOINT PIVOTS, and the pivot is where
the design choice is. Two adjacent panels have end faces square to two
different axes, so they cannot both be the same plane. Pivoted about the
chord line, they would interpenetrate on one side of it and gape on the
other, and a printed part cannot interpenetrate. Pivoted about the joint
section's TOP point when the dihedral steepens outboard (its BOTTOM when
it flattens), every point of both parts lies on its own side and the
joint opens as a wedge on the far skin instead. That wedge is glue, and
its angle and its opening in millimetres go on the build sheet.

## The slice

Each point of the layer's reference section is carried along the loft's
local span direction until it reaches the layer plane. The reference
station is the one whose chord line lies in that plane, so the points
that move are the ones above and below the chord line, and they move by

    lambda = -(delta_s) / (cos(phi) + sin(phi) dz/dy)

along the span, where delta_s is how far the point sits off the plane.
The span direction carries the chord, sweep, dihedral and twist rates of
the loft at that station and the aerofoil's own rate of change of shape,
so what is left is second order in the distance travelled: the
direction's own curvature over a few millimetres.
`test_a_layer_is_the_loft_cut_square_to_its_panel` measures it against
the analytic surface.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geom.planform import Planform


@dataclass(frozen=True)
class PanelFrame:
    """One printed panel's place in the aircraft.

    Flight frame, millimetres: x aft, y starboard, z up. Print frame: Z
    is `s`, the distance along the axis from the root pivot; X is flight
    x; Y is height square to the axis. Bed centring then subtracts
    `LayerStack.origin_mm` from X and Y."""

    eta0: float
    eta1: float
    phi_deg: float
    """The axis' angle above the span direction: the panel's dihedral."""
    origin_yz_mm: tuple
    """(y, z) of the root pivot, where s = 0."""
    length_mm: float
    kink_deg: float = 0.0
    """How much steeper this panel's axis is than the one inboard of it.
    0 for the centre body, which meets its own mirror image."""
    wedge_mm: float = 0.0
    """How far the root joint gapes at the far skin: the section's depth
    there times the change in the axes' slopes."""
    pivot: str = "centre"
    """Which point of the root joint the two faces hinge about:
    "centre" (the symmetry plane), "upper", "lower", "chord", or "flat"
    -- the height inside the section at which the joint does not turn."""

    @property
    def cos(self) -> float:
        return float(np.cos(np.radians(self.phi_deg)))

    @property
    def sin(self) -> float:
        return float(np.sin(np.radians(self.phi_deg)))

    def s_of(self, y_mm, z_mm):
        """Print height of flight points."""
        ay, az = self.origin_yz_mm
        return (np.asarray(y_mm) - ay) * self.cos + (np.asarray(z_mm) - az) * self.sin

    def to_flight(self, X, Y, s) -> np.ndarray:
        """Print (X, Y, s) before bed centring -> flight (x, y, z), mm."""
        ay, az = self.origin_yz_mm
        X, Y, s = (np.asarray(v, dtype=float) for v in (X, Y, s))
        y = ay + s * self.cos - Y * self.sin
        z = az + s * self.sin + Y * self.cos
        return np.stack(np.broadcast_arrays(X, y, z), axis=-1)

    def to_print(self, xyz) -> np.ndarray:
        """Flight (x, y, z), mm -> print (X, Y, s) before bed centring."""
        p = np.asarray(xyz, dtype=float)
        ay, az = self.origin_yz_mm
        dy, dz = p[..., 1] - ay, p[..., 2] - az
        return np.stack([p[..., 0], -dy * self.sin + dz * self.cos,
                         dy * self.cos + dz * self.sin], axis=-1)


def _section_z_extent(plan: Planform, eta: float, settings) -> tuple:
    """(z_bottom, z_chord, z_top) of the placed section at eta, flight mm,
    trailing edge thickened as it will be printed."""
    from .vase import thicken_for_nozzle
    st = plan.at(float(eta))
    c = st.chord_m * 1000.0
    loop = thicken_for_nozzle(st.airfoil.coords(settings.contour_points), c,
                              settings)
    a = np.radians(-st.twist_deg)
    pz = c * ((loop[:, 0] - 0.25) * np.sin(a) + loop[:, 1] * np.cos(a))
    z0 = st.z_le_m * 1000.0
    return z0 + float(pz.min()), z0, z0 + float(pz.max())


def panel_frames(plan: Planform, spans, settings) -> list[PanelFrame]:
    """A frame for every panel in `spans` ((eta0, eta1), root outboard).

    The centre body's axis is horizontal; every other axis runs from its
    root joint's pivot to its tip joint's (the tip's is its chord line).
    Pivots are chosen from the sign of the kink, which needs the axes, so
    this is solved twice: once on chord-line pivots to find which way each
    joint turns, then on the real pivots. A joint whose turn changes sign
    between the two is put at the height where it does not turn at all,
    if its section contains one, and on its chord line if not."""
    spans = [(float(a), float(b)) for a, b in spans]
    H = plan.half_span_m * 1000.0
    n = len(spans)
    joints = [b for _, b in spans[:-1]]
    ext = [_section_z_extent(plan, e, settings) for e in joints]
    tip = (H, plan.at(1.0).z_le_m * 1000.0)

    def axes(pivots):
        phis = [0.0]
        for i in range(1, n):
            q0 = pivots[i - 1]
            q1 = pivots[i] if i < n - 1 else tip
            phis.append(float(np.degrees(np.arctan2(q1[1] - q0[1], q1[0] - q0[0]))))
        return phis

    chord = [(e * H, zc) for e, (_, zc, _) in zip(joints, ext)]
    phis0 = axes(chord)
    kinds, pivots = [], []
    for j, (e, (zb, zc, zt)) in enumerate(zip(joints, ext)):
        turn = phis0[j + 1] - phis0[j]
        if abs(turn) < 0.05:
            kinds.append("chord")
            pivots.append((e * H, zc))
        else:
            kinds.append("upper" if turn > 0.0 else "lower")
            pivots.append((e * H, zt if turn > 0.0 else zb))
    phis = axes(pivots)
    # A joint whose turn changes sign once the pivots move is a joint that
    # can be made NOT to turn: some height inside its section puts it on
    # the line through its neighbours' pivots (for the first joint, level
    # with the next, the centre body's axis being horizontal), and there
    # the two faces are one plane -- no overlap and no wedge. The
    # gen7 trainer's first joint flipped from +10 to -0.3 degrees when both
    # joints moved to their tops; falling back to the chord line then put
    # a 10-degree turn about the chord line back, and p0 and p1 shared
    # 4.3 mm of material above it. The chord line is kept only where no
    # such height exists.
    for _ in range(len(joints)):
        changed = False
        for j in range(len(joints)):
            turn0, turn1 = phis0[j + 1] - phis0[j], phis[j + 1] - phis[j]
            if kinds[j] in ("chord", "flat") or turn0 * turn1 > 0.0:
                continue
            y = joints[j] * H
            nxt = pivots[j + 1] if j + 1 < len(joints) else tip
            if j == 0:
                z = nxt[1]
            else:
                prv = pivots[j - 1]
                z = prv[1] + (nxt[1] - prv[1]) * (y - prv[0]) / (nxt[0] - prv[0])
            zb, _, zt = ext[j]
            if zb <= z <= zt:
                kinds[j], pivots[j] = "flat", (y, float(z))
            else:
                kinds[j], pivots[j] = "chord", chord[j]
            phis = axes(pivots)
            changed = True
        if not changed:
            break

    out = []
    for i, (a, b) in enumerate(spans):
        if i == 0:
            z_axis = pivots[0][1] if pivots else plan.at(0.0).z_le_m * 1000.0
            length = (pivots[0][0] if pivots else H)
            out.append(PanelFrame(a, b, 0.0, (0.0, z_axis), float(length)))
            continue
        q0 = pivots[i - 1]
        q1 = pivots[i] if i < n - 1 else tip
        zb, _, zt = ext[i - 1]
        slope = abs(np.tan(np.radians(phis[i])) - np.tan(np.radians(phis[i - 1])))
        out.append(PanelFrame(
            a, b, phis[i], (float(q0[0]), float(q0[1])),
            float(np.hypot(q1[0] - q0[0], q1[1] - q0[1])),
            kink_deg=float(phis[i] - phis[i - 1]),
            wedge_mm=float((zt - zb) * slope),
            pivot=kinds[i - 1]))
    return out


def standalone(plan: Planform, eta0: float, eta1: float,
               settings) -> PanelFrame:
    """The frame one panel gets when it is built on its own: the frame it
    would have in a split with joints at exactly its two ends."""
    spans = []
    if eta0 > 1e-9:
        spans.append((0.0, float(eta0)))
    spans.append((float(eta0), float(eta1)))
    if eta1 < 1.0 - 1e-9:
        spans.append((float(eta1), 1.0))
    return panel_frames(plan, spans, settings)[1 if eta0 > 1e-9 else 0]


def max_lean_rate(plan: Planform, frame: PanelFrame, n: int = 9) -> float:
    """Steepest print-Y motion of the chord line per mm of print Z: the
    tangent of the largest angle between the loft's local dihedral and
    the panel axis, anywhere along the panel."""
    e = np.linspace(frame.eta0, frame.eta1, n)
    z = np.array([plan.at(float(v)).z_le_m for v in e])
    y = e * plan.half_span_m
    delta = np.arctan(np.gradient(z, y)) if np.ptp(y) > 0 else np.zeros(n)
    return float(np.abs(np.tan(delta - np.radians(frame.phi_deg))).max())


def reference_etas(plan: Planform, frame: PanelFrame, s_mm: np.ndarray,
                   n_table: int = 400) -> np.ndarray:
    """The span station whose chord line crosses each layer plane.

    s of the chord-line point rises monotonically with eta for any panel
    whose axis is within 90 degrees of the local dihedral, so a table
    inverts it. It is padded past both ends of the panel, because a
    pivot off the chord line puts the chord line's crossing of the end
    planes just outside [eta0, eta1]."""
    H = plan.half_span_m * 1000.0
    pad = (40.0 + 0.5 * frame.length_mm * abs(frame.sin)) / H
    e_tab = np.linspace(frame.eta0 - pad, frame.eta1 + pad, n_table)
    z_tab = np.array([plan.at(float(e)).z_le_m for e in e_tab]) * 1000.0
    s_tab = frame.s_of(e_tab * H, z_tab)
    return np.interp(np.asarray(s_mm, dtype=float), s_tab, e_tab)


def slice_layers(plan: Planform, frame: PanelFrame, z_mm: np.ndarray,
                 unit_loop) -> tuple[np.ndarray, np.ndarray]:
    """Cut every layer of one panel from the loft.

    `unit_loop(station, chord_mm, s_mm, min_te_mm_scale)` returns the
    layer's section at unit chord, in the station's own frame -- trailing
    edge thickened, hinge cut, rib slits in -- exactly as it would have
    been stacked before. `min_te_mm_scale` is the factor the trailing
    edge's minimum thickness has to be raised by in the `y = const`
    section so it survives the tilt into the layer plane.

    -> (contours (L, N, 2) print X/Y before bed centring, eta (L,)).

    The span rates are differences across the layers asked for, so a call
    for fewer than three layers is padded with a symmetric stencil of
    half-millimetre neighbours and only the requested layers returned. A
    SINGLE layer used to get zero rates: every point then slid along y
    alone, and a joint face cut that way sat 4 mm aft of the loft on
    micro_fpv's swept wing -- invisible to a distance-to-surface check,
    because a chordwise slide along a nearly flat crest barely leaves it."""
    z_mm = np.asarray(z_mm, dtype=float)
    if len(z_mm) < 3:
        pad = np.concatenate([z_mm[:1] - 1.0, z_mm[:1] - 0.5, z_mm,
                              z_mm[-1:] + 0.5, z_mm[-1:] + 1.0])
        c, e = slice_layers(plan, frame, pad, unit_loop)
        return c[2:2 + len(z_mm)], e[2:2 + len(z_mm)]
    H = plan.half_span_m * 1000.0
    etas = reference_etas(plan, frame, z_mm)
    st = [plan.at(float(np.clip(e, 0.0, 1.0))) for e in etas]
    y = etas * H
    chord = np.array([s_.chord_m for s_ in st]) * 1000.0
    x_le = np.array([s_.x_le_m for s_ in st]) * 1000.0
    z_le = np.array([s_.z_le_m for s_ in st]) * 1000.0
    tw = np.radians(np.array([s_.twist_deg for s_ in st]))
    # Second-order differences at the ends too: the root layer is where
    # points travel furthest (a pivot on the top skin puts the chord line's
    # crossing outboard of it), and a one-sided first-order rate there left
    # the gen7 trainer's tip panel 0.064 mm off the loft at its root.
    eo = 2 if len(y) >= 3 else 1
    if len(y) >= 2 and np.ptp(y) > 1e-9:
        d = lambda v: np.gradient(v, y, edge_order=eo)                     # noqa: E731
        d_c, d_xle, d_zle, d_a = d(chord), d(x_le), d(z_le), -d(tw)
    else:
        d_c = d_xle = d_zle = d_a = np.zeros_like(y)
    ca, sa = frame.cos, frame.sin
    ay, az = frame.origin_yz_mm
    phi = np.radians(frame.phi_deg)
    delta = np.arctan(d_zle)
    te_scale = np.cos(delta - phi) / np.cos(delta)
    # The aerofoil's own change of shape along the span, on a grid in
    # t = sqrt(x) -- the variable in which CST is a polynomial, so the grid
    # is as fine at the nose as the nose needs. Carried per point below by
    # interpolating between the two skins at the point's own height, which
    # serves the rib slits too: a slit's floor sits a bead above the lower
    # skin and moves with it. Without this the slice is first order in the
    # distance travelled, and at micro's p1 root, where points travel
    # 5.6 mm back into the body blend, that was 0.14 mm off the loft.
    tg = np.linspace(0.0, 1.0, 61)
    ug = tg * tg
    vu = np.array([s_.airfoil.y_upper(ug) for s_ in st])
    vl = np.array([s_.airfoil.y_lower(ug) for s_ in st])
    if len(y) >= 2 and np.ptp(y) > 1e-9:
        dvu, dvl = np.gradient(vu, y, axis=0, edge_order=eo), np.gradient(vl, y, axis=0, edge_order=eo)
    else:
        dvu = dvl = np.zeros_like(vu)

    out = None
    for k in range(len(z_mm)):
        loop = unit_loop(st[k], float(chord[k]), float(z_mm[k]), float(te_scale[k]))
        a = -tw[k]
        px_u = loop[:, 0] - 0.25
        rx = px_u * np.cos(a) - loop[:, 1] * np.sin(a)
        rz = px_u * np.sin(a) + loop[:, 1] * np.cos(a)
        px = x_le[k] + 0.25 * chord[k] + chord[k] * rx
        pz = chord[k] * rz
        z = z_le[k] + pz
        ti = np.sqrt(np.clip(loop[:, 0], 0.0, 1.0))
        up_i, lo_i = np.interp(ti, tg, vu[k]), np.interp(ti, tg, vl[k])
        w = np.clip((loop[:, 1] - lo_i) / np.maximum(up_i - lo_i, 1e-12), 0.0, 1.0)
        dv = np.interp(ti, tg, dvl[k]) + w * (np.interp(ti, tg, dvu[k])
                                               - np.interp(ti, tg, dvl[k]))
        dpx = (d_xle[k] + 0.25 * d_c[k] + d_c[k] * rx - pz * d_a[k]
               - chord[k] * np.sin(a) * dv)
        dz = (d_zle[k] + d_c[k] * rz + chord[k] * rx * d_a[k]
              + chord[k] * np.cos(a) * dv)
        ds = (y[k] - ay) * ca + (z - az) * sa - z_mm[k]
        lam = -ds / (ca + dz * sa)
        X = px + lam * dpx
        Y = -(y[k] + lam - ay) * sa + (z + lam * dz - az) * ca
        if out is None:
            out = np.empty((len(z_mm), len(loop), 2))
        out[k, :, 0] = X
        out[k, :, 1] = Y
    return out, etas


def joint_report(frames) -> list[str]:
    """One line per joint, root outboard, for the build sheet."""
    rows = []
    for i, f in enumerate(frames[1:], start=1):
        if f.pivot == "flat":
            rows.append(f"p{i - 1}/p{i} at eta {f.eta0:.3f}: does not turn; "
                        f"the faces mate flat")
            continue
        rows.append(
            f"p{i - 1}/p{i} at eta {f.eta0:.3f}: turns {f.kink_deg:+.1f} deg, "
            f"faces hinge about the {f.pivot} point and open "
            f"{f.wedge_mm:.1f} mm at the far skin")
    return rows
