"""The rib slits, cut into the shell as detours of the skin loop.

## A detour is the only way a single loop can have a feature

The loop has to go somewhere. To put a web across the section the
contour runs along the upper skin, **dives** to one bead above the lower
skin, crosses a **floor** one slit wide, **climbs** back, and carries
on. The two walls are laid one `rib_clearance_factor` apart so the beads
weld into a single web -- the same trick the lid uses -- and the closed
cell either side of it is untouched.

    upper  ------------.  .------------------
                       |  |                        <- dives, floors, climbs
                       |  |                        <- two beads, welded
    lower  ------------------------------------

This module owns the machinery that makes any such detour safe: holding
its vertices strictly inside their band and a minimum step apart
(`_strictly_inside`), placing them without collision (`pack_detours`),
and reparametrising the skin exactly ONCE per layer so no detour is
re-interpolated by a later one (`rebuild_skin`).

It used to own the payload bays too, which were recesses cut into the
same loop on the same machinery. Those are gone: an opening in vase mode
is sealed from the cavity, so nothing could be routed between two of
them, and the wing is loaded through the open ends of its panels before
they are bonded instead. What is left here is the rib web, which never
needed an opening in the first place.
"""

from __future__ import annotations

import numpy as np

_NEIGHBOURHOOD = 0.05
"""How far outside a bay's own band the floor's clearance is measured,
as a chord fraction. The clearance rule is vertex-to-SEGMENT, so the
opposite skin just beyond a corner is part of that corner's
neighbourhood whether or not it is inside the band: on the trainer's
root section the lower surface just aft of the pack's trailing edge is
0.4 mm higher than anything inside the band, and a floor placed against
the in-band maximum left its corner 0.255 mm from a segment it was never
measured against."""

UPPER = "upper"
LOWER = "lower"


BAND_MARGIN = 0.02
"""How close to the leading or trailing edge, as a chord fraction, an
opening's wall may come. Inside that the skin is turning through the
nose or thinning to the blunt edge, and a detour there is a detour with
nowhere to be."""

def split_skins(loop_unit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(upper LE->TE, lower LE->TE), the leading edge found by GEOMETRY.

    The LE is the vertex of minimum x. Assuming it sits at index n-1 is
    true only of a loop nothing has been inserted into."""
    i_le = int(np.argmin(loop_unit[:, 0]))
    return loop_unit[:i_le + 1][::-1], loop_unit[i_le:]


def _strictly_inside(pts: np.ndarray, x0: float, x1: float,
                     step: float) -> np.ndarray:
    """Nudge a detour's vertices so the contour stays a SIMPLE polygon.

    Every vertex strictly inside (x0, x1) and at least `step` from the
    one before it, so no edge of the detour can be zero-length and
    neither end can coincide with the skin vertex the run beside it ends
    on. Order and y are untouched; only x moves, by at most a few times
    `step`.

    The groove used to provide this for free: while a detour was always
    a bead below the skin its vertices differed in y and nothing could
    coincide. Once a detour kept its vertices where it scribes nothing
    -- which is what stops the skin being reparametrised layer to layer
    -- the y differences went away, an ear-clipper fell back to a fan,
    and six non-manifold edges came out of `stl.export`, which is the
    only thing in the program that checks.
    Guaranteeing it in x costs a few hundredths of a millimetre of
    opening and cannot be defeated by any combination of fades.
    """
    x = np.asarray(pts[:, 0], dtype=float).copy()
    lo, hi = x0 + step, x1 - step
    x[0] = min(max(x[0], lo), hi)
    for i in range(1, len(x)):
        x[i] = min(max(x[i], x[i - 1] + step), hi)
    # a band too narrow to hold them all (it never is: MIN_BAND_MM is
    # hundreds of steps wide) would stack them on `hi`; spread back down
    for i in range(len(x) - 2, -1, -1):
        x[i] = min(x[i], x[i + 1] - step)
    out = pts.copy()
    out[:, 0] = x
    return out


def rebuild_skin(near: np.ndarray, detours) -> np.ndarray:
    """One skin, LE -> TE, resampled around its detours -> (M, 2).

    `detours` are (x_a, x_b, points) with x_a < x_b in the skin's own
    chord fractions and `points` the detour's vertices, forward to aft,
    which are kept EXACTLY. Only the free skin between detours is
    resampled: it is divided into segments with FIXED vertex counts, so
    the total per layer is constant and the vertices inside a segment
    slide smoothly as its ends move. Snapping a detour onto existing
    vertices would quantise it to the contour grid, and a corner that
    jumps a grid interval as the depth ramps is a step in a wall the
    printer builds in mid-air.

    This is the one function every detour goes through. Bays and ribs
    used to have one each, applied in sequence, and the second
    re-interpolated a skin that already carried the first's vertical
    walls: on the trainer's centre body a ledge vertex dropped 7.2 mm to
    the floor between two layers a millimetre apart, 72.8 degrees of
    overhang from a bay whose ramp had been budgeted at 50.

    The skin's own ends bound the segments -- a panel truncated at the
    hinge line ends at x_hinge, not 1.0, and walking to 1.0 clamped the
    last segment's vertices onto the cut face in a pile."""
    sx, sy = near[:, 0], near[:, 1]
    n_pts = len(near)
    detours = sorted(detours, key=lambda d: d[0])
    for (a0, b0, _), (a1, b1, _) in zip(detours, detours[1:]):
        if a1 < b0 - 1e-12:
            raise ValueError(f"detours overlap: [{a0:.4f}, {b0:.4f}] and "
                             f"[{a1:.4f}, {b1:.4f}]")
    edges = [float(sx[0])]
    for a, b, _ in detours:
        edges += [a, b]
    edges += [float(sx[-1])]
    segs = [(edges[2 * i], edges[2 * i + 1]) for i in range(len(detours) + 1)]
    widths = np.array([max(b - a, 1e-6) for a, b in segs])
    counts = np.maximum(np.round(n_pts * widths / widths.sum()), 2).astype(int)
    while counts.sum() > n_pts:
        counts[int(np.argmax(counts))] -= 1
    while counts.sum() < n_pts:
        counts[int(np.argmax(widths / counts))] += 1
    out = []
    for k, ((a, b), cnt) in enumerate(zip(segs, counts)):
        t = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, cnt)))
        xs = a + (b - a) * t
        out.append(np.stack([xs, np.interp(xs, sx, sy)], 1))
        if k < len(detours):
            out.append(np.asarray(detours[k][2], dtype=float))
    return np.concatenate(out, 0)


def pack_detours(detours, near: np.ndarray, lo_lim: float, hi_lim: float,
                 sep: float = 0.0):
    """Slide CLOSED detours aside so no two slots overlap -> new list.

    Every bay keeps its six vertices for the whole panel, so a bay whose
    band has run off the front of the section still holds a slot, and two
    of those pile up against the leading edge: on the trainer the
    electronics bay, long closed and clamped to a sliver at the nose, met
    the pack's band exactly.

    Where a bay is closed its vertices lie on the skin and scribe
    nothing, so their chordwise position is free and they may be moved.
    Where it is OPEN the position IS the opening and may not move; two
    open detours wanting the same chord is a real conflict that the seat
    solver gates, and the geometry still refuses it rather than quietly
    sliding an opening somewhere else.

    Entries are (x0, x1, points, movable). The immovable ones are laid
    down first and keep exactly the chord they asked for; the movable
    ones are then fitted into the gaps that remain, each as near its own
    station as it can get, and re-seated on `near` so its vertices still
    lie on the skin after the slide.

    Placing them in one ordered sweep instead is not enough: a closed bay
    clamped to the nose and an open one clamped beside it both start at
    the same limit, and whichever the sort happened to put first won.
    """
    fixed = [(a, b, q) for a, b, q, m in detours if not m]
    free = [(a, b, q) for a, b, q, m in detours if m]
    fixed.sort(key=lambda d: d[0])
    for (a0, b0, _), (a1, b1, _) in zip(fixed, fixed[1:]):
        if a1 < b0 - 1e-9:
            raise ValueError(f"open detours overlap: [{a0:.4f}, {b0:.4f}] "
                             f"and [{a1:.4f}, {b1:.4f}]")
    sx, sy = near[:, 0], near[:, 1]
    placed = list(fixed)
    for a, b, q in sorted(free, key=lambda d: d[0] - d[1]):   # widest first
        w = b - a
        # every gap is shrunk by `sep` at each end, so a parked detour
        # never abuts its neighbour exactly: touching slots put one
        # detour's last vertex on the next one's first, and both lie on
        # the skin, which is a zero-length edge again by another route
        gaps, cur = [], lo_lim
        for oa, ob, _ in sorted(placed, key=lambda d: d[0]):
            if (oa - sep) - (cur + sep) >= w:
                gaps.append((cur + sep, oa - sep))
            cur = max(cur, ob)
        if (hi_lim - sep) - (cur + sep) >= w:
            gaps.append((cur + sep, hi_lim - sep))
        if not gaps:
            # Nothing wide enough: park it SHRUNK in the widest gap there
            # is. Its vertices lie on the skin, so the only thing its
            # width buys is a comfortable spread; a narrow slot costs
            # nothing but the spacing rule, which is checked below. The
            # alternative was refusing the layer, and once the wiring
            # channels joined the payload bays in competing for one
            # skin's chord that happened on ordinary designs.
            all_gaps, cur = [], lo_lim
            for oa, ob, _ in sorted(placed, key=lambda d: d[0]):
                if oa - cur > 0.0:
                    all_gaps.append((cur, oa))
                cur = max(cur, ob)
            if hi_lim - cur > 0.0:
                all_gaps.append((cur, hi_lim))
            if not all_gaps:
                raise ValueError(f"no chord left at all to park a closed "
                                 f"bay's {len(q)} vertices")
            g0, g1 = max(all_gaps, key=lambda g: g[1] - g[0])
            gaps, w = [(g0 + sep, g1 - sep)], max(g1 - g0 - 2.0 * sep, 0.0)
            if w <= 0.0:
                raise ValueError(f"no chord left at all to park a closed "
                                 f"bay's {len(q)} vertices")
            q = np.asarray(q, dtype=float).copy()
            span = float(q[-1, 0] - q[0, 0])
            if span > 1e-12:
                q[:, 0] = g0 + sep + (q[:, 0] - q[0, 0]) * (w / span)
            a, b = g0 + sep, g0 + sep + w
        na = min((min(max(a, g0), g1 - w) for g0, g1 in gaps),
                 key=lambda x: abs(x - a))
        q = np.asarray(q, dtype=float).copy()
        q[:, 0] += na - a
        # back onto the skin: a closed detour's vertices ARE the skin, and
        # translating them in x alone would leave them off the surface
        q[:, 1] = np.interp(q[:, 0], sx, sy)
        placed.append((na, na + w, q))
    return sorted(placed, key=lambda d: d[0])


def rib_detours(upper: np.ndarray, lower: np.ndarray, chord_mm: float,
                z_mm: float, spec, slit_mm: float, gap_mm: float) -> list:
    """Every rib slit on the upper skin at this height, as detours.

    A rib is a chordwise web from the upper skin to one bead above the
    lower: the detour is two vertices on the floor, `slit` apart, and
    the walls are the segments the closed contour already has. The floor
    clears the lower skin over a NEIGHBOURHOOD of the slit, not at three
    sample points inside it, because the clearance rule is
    vertex-to-segment and the skin just outside the slit is part of the
    floor corners' neighbourhood."""
    if spec is None or not spec.enabled or spec.n_ribs <= 0:
        return []
    slit = slit_mm / chord_mm
    gap = gap_mm / chord_mm
    x_te = float(max(upper[:, 0].max(), lower[:, 0].max()))
    xr = np.sort(np.clip(spec.stations(z_mm, chord_mm),
                         spec.x_clip[0],
                         min(spec.x_clip[1], x_te - 1.5 * slit)))
    out = []
    for x in xr:
        xa, xb = x - 0.5 * slit, x + 0.5 * slit
        pad = 2.0 * slit
        xs_f = np.linspace(xa - pad, xb + pad, 9)
        floor = float(np.max(np.interp(xs_f, lower[:, 0], lower[:, 1]))) + gap
        out.append((float(xa), float(xb),
                    np.array([[xa, floor], [xb, floor]])))
    return out



def insert_detours(loop_unit: np.ndarray, chord_mm: float, z_mm: float,
                   rib_spec=None, slit_mm: float = 0.5, gap_mm: float = 0.5,
                   min_groove_mm: float = 0.45) -> np.ndarray:
    """Every rib slit on this layer, in ONE pass -> longer contour.

    The slits are detours of the UPPER skin, inserted by a single
    reparametrisation so none of them is re-interpolated by a later one.
    Inserting them one at a time handed the second call a skin the first
    had already moved, and the vertices no longer lay where the clearance
    had been measured. The loop comes back in Selig order: TE over the
    upper to the LE, then the lower to the TE, sharing the LE vertex
    exactly once.
    """
    upper, lower = split_skins(loop_unit)
    up = [(a, b, pts, False) for a, b, pts in
          rib_detours(upper, lower, chord_mm, z_mm, rib_spec, slit_mm, gap_mm)]
    if not up:
        return loop_unit
    lim = (float(upper[0, 0]) + BAND_MARGIN, float(upper[-1, 0]) - BAND_MARGIN)
    sep = min_groove_mm / 20.0 / chord_mm
    new_up = rebuild_skin(upper, pack_detours(up, upper, *lim, sep=sep))
    return np.concatenate([new_up[::-1], lower[1:]], 0)
