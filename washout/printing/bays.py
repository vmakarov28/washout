"""The payload bays, cut into the shell as detours of the skin loop.

Until now a `Bay` was reserved *volume*: a box checked against the
section's depth, against the spars and against the print joints, and then
not cut. The battery went in through a hole someone made with a knife.

## An opening in vase mode is a recess, never a hole

The loop has to go somewhere. To open the top of a bay the contour runs
along the upper skin, **dives** at one wall, runs across a **floor**,
**climbs** the other wall, and carries on. The floor sits exactly deep
enough to hold the box, and the closed cell UNDER it is where the spar
lives and where most of the torsion box survives.

    upper  -----.  .____________.  .--------      <- ledge for the lid
                |__|            |__|
                |     the bay      |              <- dives, floors, climbs
                '------------------'
                     closed cell
    lower  ----------------------------------

A pocket that opens DOWNWARD -- a servo's, whose arm has to reach the
horn under the wing -- is the same detour on the lower skin, mirrored.

Topologically the section is still one simple closed curve. It is the
rib detour of `printing/ribs.py` widened from a slit to a chord band, and
it inherits that module's two invariants: the point count per layer is
CONSTANT, and every inserted vertex stays at least one extrusion width
from every non-adjacent part of the contour.

## Every bay is cut in ONE pass

All of a skin's bays are inserted by a single reparametrisation, exactly
as `insert_ribs` inserts all the ribs. The first version cut them one at
a time, and the second call assumed the loop it was handed still had its
leading edge at index n-1 -- which a loop that already carries one bay
does not. It split one vertex early, duplicated the leading edge, and the
zero-length edge that produced stalled the ear-clipper into an invalid
fan: six non-manifold edges on both caps of the trainer's centre body.
The same index assumption that broke `spar_fit` on ribbed panels,
arriving by a different route. The leading edge is now found by
geometry -- it is the vertex of minimum x -- and never assumed.

## The bay fades in and out, and a root bay only has to fade once

A wall normal to the span is a roof in this print orientation and
spiralize cannot build one, so a bay's depth is a function of Z and the
rate of change is an overhang (`vase.ramp_budget`). A bay at the
centreline starts at the root face -- which is open anyway, because the
spar has to get in and the halves join there -- so it needs only ONE
ramp, at its outboard end. A bay out in the wing needs two. Treating a
servo pocket as if it started at the root cut it into the centre body at
full depth, which is how that was found.

Where a bay has closed the detour does not vanish: it becomes a one-bead
groove. That keeps the point count constant without coincident vertices,
and the groove scribes the hatch rim on the finished part.

## What it costs, and the cost is the point

The upper skin is the compression member and the closed cell is the
torsion box. Cutting the bay open removes both over the bay's span:
`aeroelastic.gj_open_nmm2` is one to two orders below the closed value.
That is not a reason not to cut the bay -- you cannot load a battery
through an unbroken skin -- it is a reason the cut has to be paid for in
the same evaluation that scores the aircraft.
"""

from __future__ import annotations

from dataclasses import dataclass

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

POINTS_PER_BAY = 6
"""Per bay, per layer, always: at each wall a ledge-outer and a
ledge-inner vertex, and between them the floor's two corners. The walls
are the segments a closed contour already has. Six whether the bay is
open, closing, a pocket with no ledge, or a groove -- and never
coincident with the skin's own vertices, so no zero-length edge can
reach the ear-clipper."""

UPPER = "upper"
LOWER = "lower"


def clamp_band(mid: float, half: float, chord_mm: float,
               skin_x0: float = 0.0, skin_x1: float = 1.0
               ) -> tuple[float, float]:
    """An opening's (x0, x1) in chord fractions: centred on `mid`, half
    width `half`, clipped into [skin_x0, skin_x1] less `BAND_MARGIN`,
    and never narrower than `MIN_BAND_MM`.

    The one place that decides where an opening's walls are. `BaySpec`
    cuts to it and `vase.cut_budget_profile` budgets the ramp against it;
    when the budget had its own clipping rule it declared stations
    unusable that the cutter was happy to cut, and the trainer's
    electronics bay was given 70 mm of span to close in when it has 453.
    """
    lo_lim, hi_lim = skin_x0 + BAND_MARGIN, skin_x1 - BAND_MARGIN
    want = max(MIN_BAND_MM / max(chord_mm, 1e-9), 1e-6)
    lo, hi = max(mid - half, lo_lim), min(mid + half, hi_lim)
    if hi - lo >= want:
        return (lo, hi)
    # squeezed by the section's edge: hold the minimum width and slide it
    # inside, so the band moves continuously instead of vanishing
    if lo <= lo_lim + 1e-12:
        lo = lo_lim
        hi = min(lo + want, hi_lim)
    else:
        hi = hi_lim
        lo = max(hi - want, lo_lim)
    return (lo, hi)

BAND_MARGIN = 0.02
"""How close to the leading or trailing edge, as a chord fraction, an
opening's wall may come. Inside that the skin is turning through the
nose or thinning to the blunt edge, and a detour there is a detour with
nowhere to be."""

MIN_BAND_MM = 4.0
"""The narrowest an opening's band may become. It is not a threshold for
dropping the detour -- a detour is never dropped, because removing one
reparametrises the whole skin in a single layer -- it is the width the
band is held at when the section's edge would otherwise squeeze it to
nothing. Below about this the six vertices would sit closer together
than the lip between them."""

GROOVE_TAIL_MM = 2.0
"""Span over which a closed bay's groove fades from one bead to nothing."""

FIT_MM = 0.6
"""Chordwise clearance between a box and the walls of its pocket, total.

A pocket cut to exactly the box's length has the box's length between
its two bead CENTRES, so the walls intrude half a bead each and the box
does not go in. The cut is therefore the box plus two beads plus this,
which is a declared fit allowance -- 0.3 mm a side -- not a measured one.
The ramps give a pocket all the spanwise slack it needs; chordwise there
is nothing else to give it."""


@dataclass(frozen=True)
class BaySpec:
    """One bay to cut, in the loop's own chord fractions."""

    name: str
    x0: float                   # forward edge
    x1: float                   # aft edge
    start_mm: float             # z at which full depth BEGINS (0 = root face)
    span_mm: float              # z at which full depth ENDS
    ramp_mm: float              # span over which it CLOSES, outboard
    depth_mm: float             # how deep the floor sits below the opened skin
    gap_mm: float               # clearance the floor keeps from the far skin
    open_from: str = UPPER
    ledge_mm: float = 0.0       # lid thickness; 0 means a pocket, no ledge
    ramp_in_mm: float = 0.0     # span over which it OPENS, inboard; 0 = a root bay
    length_mm: float | None = None
    """The opening's chordwise extent in MILLIMETRES. When set, `x0` and
    `x1` only give the centre and the band is recomputed at every layer
    from that layer's chord, because a pocket is a box and a box does
    not get shorter as the chord tapers: cut as a fraction, the trainer's
    servo pocket was 17.8 mm wide at its inboard end and 15.5 at its
    outboard, for a 23 mm servo."""
    x_abs_mm: float | None = None
    """The opening's centre as an absolute station, mm aft of the ROOT
    leading edge. A box is rigid: its centre does not slide with the
    chord fraction as the wing sweeps under it. With a physical width on
    a fractional centre, the trainer's two root bays overlapped where
    their grooves ran out. Needs the layer's own `x_le_mm` to convert."""

    def band(self, chord_mm: float, x_le_mm: float = 0.0,
             skin_x0: float = 0.0, skin_x1: float = 1.0
             ) -> tuple[float, float]:
        """(x0, x1) in this layer's own chord fractions, CLIPPED to the
        skin's extent less `BAND_MARGIN`, and never narrower than
        `MIN_BAND_MM`.

        The clipping is what lets a root bay on a swept body close at
        all: its box is rigid and sits at an absolute station, but the
        ramp that closes the opening is not a box, and as the leading
        edge comes round outboard the recess simply narrows with it.

        It is CLAMPED rather than refused, and slid inside the section
        rather than dropped, because a detour that disappears takes its
        segment with it and the skin either side is reparametrised in one
        layer. Where the bay is closed these six vertices lie on the skin
        and cost nothing; what they buy is a point distribution that
        varies continuously from root to tip."""
        if self.length_mm is None:
            return (self.x0, self.x1)
        h = 0.5 * self.length_mm / max(chord_mm, 1e-9)
        if self.x_abs_mm is not None:
            mid = (self.x_abs_mm - x_le_mm) / max(chord_mm, 1e-9)
        else:
            mid = 0.5 * (self.x0 + self.x1)
        return clamp_band(mid, h, chord_mm, skin_x0, skin_x1)

    def groove_frac(self, z_mm: float) -> float:
        """How much of the one-bead groove remains at this height.

        A closed bay used to leave its groove all the way to the tip: a
        scribe line down the whole wing, 0.45 mm deep, for the sake of a
        constant point count. It now fades to nothing over `GROOVE_TAIL_MM`
        past each ramp, at a rate far inside the overhang limit, and a
        bay that has faded out is not a detour at all."""
        r_out = max(self.ramp_mm, 1e-9)
        end = self.span_mm + r_out
        if z_mm > end:
            return float(max(0.0, 1.0 - (z_mm - end) / GROOVE_TAIL_MM))
        if self.ramp_in_mm > 0.0 and self.start_mm > 0.0:
            begin = self.start_mm - max(self.ramp_in_mm, 1e-9)
            if z_mm < begin:
                return float(max(0.0, 1.0 - (begin - z_mm) / GROOVE_TAIL_MM))
        elif self.start_mm > 0.0 and z_mm < self.start_mm:
            return 0.0
        return 1.0

    def present(self, z_mm: float) -> bool:
        return self.groove_frac(z_mm) > 0.0

    def depth_frac(self, z_mm: float) -> float:
        """How open the bay is at this height: a trapezoid in Z.

        Ramps in from `start - ramp_in` to `start`, full to `span`, ramps
        out to `span + ramp`. A root bay has start = 0 and no ramp-in --
        one ramp, as the root face is open anyway. The two ramps are
        sized SEPARATELY, because they cross different panels: sizing the
        ramp-in from the outboard walk that sized the ramp-out let the
        trainer's servo pocket open at 1.04 mm/mm through a panel whose
        budget there was 0.98. Linear because the overhang limit is a
        limit on the RATE, so the cheapest profile that respects it has
        constant rate."""
        r_out = max(self.ramp_mm, 1e-9)
        r_in = max(self.ramp_in_mm, 1e-9)
        if z_mm > self.span_mm + r_out:
            return 0.0
        if z_mm >= self.start_mm:
            return 1.0 if z_mm <= self.span_mm else float(
                1.0 - (z_mm - self.span_mm) / r_out)
        if self.ramp_in_mm <= 0.0:
            return 1.0 if self.start_mm <= 0.0 else 0.0
        if z_mm < self.start_mm - r_in:
            return 0.0
        return float((z_mm - (self.start_mm - r_in)) / r_in)


def split_skins(loop_unit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(upper LE->TE, lower LE->TE), the leading edge found by GEOMETRY.

    The LE is the vertex of minimum x. Assuming it sits at index n-1 is
    true only of a loop nothing has been inserted into."""
    i_le = int(np.argmin(loop_unit[:, 0]))
    return loop_unit[:i_le + 1][::-1], loop_unit[i_le:]


def floor_limits_skins(near: np.ndarray, far: np.ndarray, x0: float,
                       x1: float, chord_mm: float, depth_mm: float,
                       gap_mm: float, min_groove_mm: float,
                       open_from: str) -> tuple[float, float] | None:
    """(y at fully closed, y at fully open), in chord units.

    `near` is the skin being opened, `far` the one the floor must clear.
    Returns None only when not even the one-bead groove fits. Everything
    else is CLAMPED: the floor is as deep as the box needs where the
    section allows and as deep as the section allows where it does not,
    so the geometry can never self-intersect; whether the box fits is
    `Volume.fits`'s question, reported as a gate with the shortfall.
    Declining instead left the layer short of vertices and the contour
    array would not broadcast.
    """
    pad = _NEIGHBOURHOOD
    sel_n = (near[:, 0] >= x0) & (near[:, 0] <= x1)
    sel_f = (far[:, 0] >= x0 - pad) & (far[:, 0] <= x1 + pad)
    if sel_n.sum() < 2 or sel_f.sum() < 2:
        return None
    if open_from == UPPER:                       # deeper is DOWN
        y_top = float(near[sel_n, 1].min()) - min_groove_mm / chord_mm
        y_lim = float(far[sel_f, 1].max()) + gap_mm / chord_mm
        if y_top <= y_lim:
            return None
        return y_top, max(y_top - depth_mm / chord_mm, y_lim)
    y_top = float(near[sel_n, 1].max()) + min_groove_mm / chord_mm     # UP
    y_lim = float(far[sel_f, 1].min()) - gap_mm / chord_mm
    if y_top >= y_lim:
        return None
    return y_top, min(y_top + depth_mm / chord_mm, y_lim)


def floor_limits(loop_unit: np.ndarray, x0: float, x1: float,
                 chord_mm: float, depth_mm: float, gap_mm: float,
                 min_groove_mm: float, open_from: str = UPPER
                 ) -> tuple[float, float] | None:
    """`floor_limits_skins` for a whole loop."""
    upper, lower = split_skins(loop_unit)
    near, far = (upper, lower) if open_from == UPPER else (lower, upper)
    return floor_limits_skins(near, far, x0, x1, chord_mm, depth_mm,
                              gap_mm, min_groove_mm, open_from)


def _strictly_inside(pts: np.ndarray, x0: float, x1: float,
                     step: float) -> np.ndarray:
    """Nudge a detour's vertices so the contour stays a SIMPLE polygon.

    Every vertex strictly inside (x0, x1) and at least `step` from the
    one before it, so no edge of the detour can be zero-length and
    neither end can coincide with the skin vertex the run beside it ends
    on. Order and y are untouched; only x moves, by at most a few times
    `step`.

    This is the invariant `POINTS_PER_BAY` claims and the groove used to
    provide: while the detour was always a bead below the skin its
    vertices differed in y and nothing could coincide. Once a closed bay
    kept its vertices -- which is what stops the skin being
    reparametrised layer to layer -- the y differences went away, an
    ear-clipper fell back to a fan, and six non-manifold edges came out
    of `stl.export`, which is the only thing in the program that checks.
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


def _detour(spec: BaySpec, near: np.ndarray, far: np.ndarray,
            chord_mm: float, z_mm: float, min_groove_mm: float,
            x_le_mm: float = 0.0) -> np.ndarray:
    """The six detour vertices for one bay, forward wall to aft wall."""
    sx, sy = near[:, 0], near[:, 1]
    sign = 1.0 if spec.open_from == UPPER else -1.0   # +1: deeper is down
    groove = min_groove_mm / chord_mm * spec.groove_frac(z_mm)
    x0, x1 = spec.band(chord_mm, x_le_mm, float(sx[0]), float(sx[-1]))
    lip = 0.04 * (x1 - x0)                            # the lid's bearing
    lim = floor_limits_skins(near, far, x0, x1, chord_mm,
                             spec.depth_mm, spec.gap_mm, min_groove_mm,
                             spec.open_from)
    if lim is None:
        # not even a groove fits: six distinct vertices a groove inside
        # the skin, so the count holds and nothing is coincident
        xs = np.linspace(x0, x1, POINTS_PER_BAY)
        ys = np.interp(xs, sx, sy) - sign * groove
        return _strictly_inside(np.stack([xs, ys], 1), x0, x1,
                                min_groove_mm / 20.0 / chord_mm)

    y_closed, y_open = lim
    d = spec.depth_frac(z_mm)

    # The floor follows the SKIN when closed and is flat only when open.
    # A box is flat, so an open bay's floor is one horizontal line; a
    # scribe line is not, and a horizontal groove at the band's lowest
    # skin point is a 2-3 mm step on a cambered surface -- which is what
    # the rendered section showed on both skins. Blended linearly in the
    # depth fraction, so the corners move smoothly through the ramp.
    def floor_at(x):
        hug = float(np.interp(x, sx, sy)) - sign * groove
        return (1.0 - d) * hug + d * y_open

    # How far the six vertices pull INSIDE the band, and how far the
    # ledge's two split apart. Zero while the bay is open, so the walls
    # stand exactly at x0 and x1 and the ledge face is vertical; growing
    # as the groove fades, because once the detour's y values have all
    # reached the skin its vertices would otherwise be coincident with
    # each other and with the skin vertex each neighbouring run ends on.
    #
    # `POINTS_PER_BAY` says this module never lets a zero-length edge
    # reach the ear-clipper, and while there was always a groove the y
    # difference guaranteed it. Once a closed bay learned to keep its
    # vertices -- which is what stops the skin being reparametrised -- the
    # guarantee had to move into x. Six non-manifold edges on the
    # trainer's centre body, from an ear-clipper that fell back to a fan,
    # found by `stl.export` and by nothing else: watertightness is not
    # one of the printability gates.
    g = spec.groove_frac(z_mm)
    e = 0.5 * lip * (1.0 - g)
    # The ledge face is a slope over `q` rather than a vertical segment
    # between two vertices sharing an x. Unconditional, because the pair
    # coincides whenever the floor is no deeper than the ledge -- which
    # is a nearly-closed bay, DEPTH near zero, and its groove may still
    # be full, so a separation keyed on the groove does not see it. A
    # hundredth of the band's width: the trainer's 73 mm hatch gets a
    # 0.7 mm slope, the 4 mm minimum band 0.04 mm.
    q = 0.25 * lip
    # Vertex separation: a twentieth of a bead, which is far below
    # anything the printer resolves and far above float noise.
    sep = min_groove_mm / 20.0 / chord_mm

    if spec.ledge_mm <= 0.0:
        # a pocket: the floor's corners, and two more spread along it
        xs = np.array([x0 + e, x0 + lip, x0 + 2 * lip,
                       x1 - 2 * lip, x1 - lip, x1 - e])
        pts = np.stack([xs, [floor_at(x) for x in xs]], 1)
        return _strictly_inside(pts, x0, x1, sep)

    # a ledge one lid-thickness inside the skin at each wall, but never
    # deeper than the floor itself -- on a nearly closed bay the ledge
    # simply IS the floor.
    #
    # It fades with the groove, and did not until a test asked whether a
    # closed bay's vertices lie on the skin: they did not, because the
    # depth faded and the groove after it while the LEDGE was cut at full
    # thickness regardless. A lidded bay therefore scribed a 0.9 mm notch
    # along its whole panel, outboard of any lid there was to rest in it.
    ledge = spec.ledge_mm / chord_mm * spec.groove_frac(z_mm)

    def ledge_at(x):
        """The lid's seat: one lid-thickness inside the skin, FOLLOWING
        it, and never deeper than the floor under it.

        Per station, not one height carried across the shelf's two
        vertices. A flat shelf on a curved skin left its outer vertex
        1.0 mm proud of the surface, and where the bay had closed -- so
        the seat should have been the skin itself -- that bump ran the
        whole panel. The lid is a lens built from the same station loops,
        so a seat that follows the skin is also the seat it wants."""
        y = float(np.interp(x, sx, sy)) - sign * ledge
        f = floor_at(x)
        return max(y, f) if sign > 0 else min(y, f)

    pts = np.array([[x0 + e, ledge_at(x0 + e)],
                    [x0 + lip, ledge_at(x0 + lip)],
                    [x0 + lip + q, floor_at(x0 + lip + q)],
                    [x1 - lip - q, floor_at(x1 - lip - q)],
                    [x1 - lip, ledge_at(x1 - lip)],
                    [x1 - e, ledge_at(x1 - e)]])
    return _strictly_inside(pts, x0, x1, sep)


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
                   bay_specs=(), rib_spec=None, slit_mm: float = 0.5,
                   gap_mm: float = 0.5, min_groove_mm: float = 0.45,
                   x_le_mm: float = 0.0) -> np.ndarray:
    """Every detour on both skins, in ONE pass -> longer contour.

    Bays open from whichever skin they were declared on; ribs are slits
    in the upper skin. All of a skin's detours are inserted by a single
    reparametrisation (`rebuild_skin`), so none of them is ever
    re-interpolated by a later one. The loop comes back in Selig order:
    TE over the upper to the LE, then the lower to the TE, sharing the LE
    vertex exactly once.
    """
    bay_specs = list(bay_specs)
    upper, lower = split_skins(loop_unit)
    # EVERY bay keeps its six vertices, at every height. Where it is
    # closed they lie on the skin and scribe nothing; dropping them
    # instead reparametrised the skin either side in a single layer and
    # the overhang gate read the jump as material in mid-air.
    up_b = [b for b in bay_specs if b.open_from == UPPER]
    lo_b = [b for b in bay_specs if b.open_from == LOWER]
    def shut(b):
        """Closed AND unscribed: the six points are plain skin."""
        return b.depth_frac(z_mm) <= 0.0 and b.groove_frac(z_mm) <= 0.0

    up = [(*b.band(chord_mm, x_le_mm, float(upper[0, 0]), float(upper[-1, 0])),
           _detour(b, upper, lower, chord_mm, z_mm, min_groove_mm, x_le_mm),
           shut(b))
          for b in up_b]
    up += [(a, b_, pts, False) for a, b_, pts in
           rib_detours(upper, lower, chord_mm, z_mm, rib_spec, slit_mm, gap_mm)]
    lo = [(*b.band(chord_mm, x_le_mm, float(lower[0, 0]), float(lower[-1, 0])),
           _detour(b, lower, upper, chord_mm, z_mm, min_groove_mm, x_le_mm),
           shut(b))
          for b in lo_b]
    up_lim = (float(upper[0, 0]) + BAND_MARGIN, float(upper[-1, 0]) - BAND_MARGIN)
    lo_lim = (float(lower[0, 0]) + BAND_MARGIN, float(lower[-1, 0]) - BAND_MARGIN)
    sep = min_groove_mm / 20.0 / chord_mm
    new_up = (rebuild_skin(upper, pack_detours(up, upper, *up_lim, sep=sep))
              if up else upper)
    new_lo = (rebuild_skin(lower, pack_detours(lo, lower, *lo_lim, sep=sep))
              if lo else lower)
    return np.concatenate([new_up[::-1], new_lo[1:]], 0)


def insert_bays(loop_unit: np.ndarray, chord_mm: float, z_mm: float,
                specs, min_groove_mm: float) -> np.ndarray:
    """The bays alone. `insert_detours` with no ribs."""
    specs = list(specs)
    if not specs:
        return loop_unit
    return insert_detours(loop_unit, chord_mm, z_mm, specs, None,
                          min_groove_mm=min_groove_mm)


def bay_point_budget(n_bays: int) -> int:
    return POINTS_PER_BAY * max(int(n_bays), 0)
