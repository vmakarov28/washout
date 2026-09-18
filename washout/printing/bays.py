"""The payload bay, cut into the shell as a detour of the skin loop.

Until now a `Bay` was reserved *volume*: a box checked against the
section's depth, against the spars and against the print joints, and then
not cut. The battery went in through a hole someone made with a knife.

## An opening in vase mode is a recess, never a hole

The loop has to go somewhere. To open the top of a bay the contour runs
aft-to-forward along the upper skin, **dives** at the bay's trailing
edge, runs forward along a **floor** one bead above the lower skin,
**climbs** back at the bay's leading edge, and carries on to the nose.
Later in the same loop it comes back along the real lower skin, one bead
under that floor, and the two weld.

    upper  -----.                    .--------
                |                    |
                |    the bay         |          <- dives, floors, climbs
                |                    |
    floor       '--------------------'
    lower  ------------------------------------  (one bead below: welds)

Topologically still one simple closed curve. It is the rib detour of
`printing/ribs.py` widened from a slit to a chord band, and it inherits
that module's two invariants: the point count per layer is CONSTANT, and
every inserted vertex stays at least one extrusion width from every
non-adjacent part of the contour.

## The bay fades out, and it only has to do it once

A wall normal to the span is a roof in this print orientation and
spiralize cannot build one, so the bay's depth is a function of Z and the
rate of change is an overhang (`vase.ramp_budget`). But the root face is
open anyway -- the spar has to get in and the two halves join there -- so
a bay that starts at the centreline needs only **one** ramp, at its
outboard end. That halves the span it costs.

Where the bay has closed, the detour does not vanish: it becomes a
one-bead groove in the upper skin. That keeps the point count constant
without coincident vertices, and the groove is useful in its own right --
it scribes the hatch rim on the finished part.

## What it costs, and the cost is the point

The upper skin is the compression member and the closed cell is the
torsion box. Cutting the bay open removes both, over the bay's span:
`aeroelastic.gj_open_nmm2` is one to two orders below the closed value,
which takes demon1's divergence from 122 m/s to 72. That is not a reason
not to cut the bay -- you cannot load a battery through an unbroken skin
-- it is a reason the cut has to be paid for in the same evaluation that
scores the aircraft.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_NEIGHBOURHOOD = 0.05
"""How far outside the bay's own band the floor's clearance is measured,
as a chord fraction. See `floor_limits`: the clearance rule is
vertex-to-segment, so the lower skin just beyond a corner is part of the
corner's neighbourhood whether or not it is inside the band."""

POINTS_PER_BAY = 2
"""Two vertices: the floor's aft corner and its forward corner.

The dive and the climb are the segments BETWEEN those corners and the
upper-surface points either side of them, which a closed contour already
has. Adding a ladder of points down each wall would only give the slicer
somewhere to round the corner off."""


@dataclass(frozen=True)
class BaySpec:
    """One bay to cut, in the loop's own chord fractions."""

    name: str
    x0: float                   # forward edge
    x1: float                   # aft edge
    span_mm: float              # full-depth run from the root face
    ramp_mm: float              # span over which it closes
    depth_mm: float             # how deep the floor sits below the upper skin
    floor_gap_mm: float         # clearance the floor must keep from the lower skin

    def depth_frac(self, z_mm: float) -> float:
        """How open the bay is at this height. 1 at the root, 0 outboard.

        Full depth from the root face to `span_mm`, then a linear ramp to
        closed over `ramp_mm`. Linear because the overhang limit is a
        limit on the RATE, so the cheapest profile that respects it is
        the one with constant rate."""
        if z_mm <= self.span_mm:
            return 1.0
        if self.ramp_mm <= 0.0:
            return 0.0
        return float(np.clip(1.0 - (z_mm - self.span_mm) / self.ramp_mm,
                             0.0, 1.0))


def floor_limits(loop_unit: np.ndarray, x0: float, x1: float,
                 chord_mm: float, depth_mm: float, gap_mm: float,
                 min_groove_mm: float) -> tuple[float, float] | None:
    """(y at fully closed, y at fully open) for the floor, in chord units.

    Returns None only when not even the one-bead groove fits between the
    skins, which on a real aerofoil means the band runs off the trailing
    edge. Everything else is CLAMPED rather than refused -- see below.

    The floor sits exactly deep enough to hold the box and **no deeper**,
    and the difference matters more than it looks.

    Cutting to one bead above the lower skin -- the first version -- opens
    the section over the whole of the bay's chord band, which destroys the
    very solution the spar seat solver had found: on the trainer the TE
    spar clears the pack by taking the upper skin while the pack sits on
    the lower one, and a full-depth cut removes the upper region the tube
    was seated in. A real bay has a FLOOR with closed section underneath
    it, and that residual cell is where the spar lives and most of the
    torsion box survives.

    The floor is one horizontal line across the whole band, so it is
    measured from the LOWEST point of the upper skin in the band, and it
    must clear the HIGHEST point of the lower skin. Taking those extremes
    rather than the values at the band's centre is what stops the floor
    poking through a cambered surface near the nose.

    The band is WIDENED before taking those extremes, and that is not
    padding for luck. The clearance rule is a distance from a vertex to a
    SEGMENT, and the floor's corners sit at x0 and x1 with the lower skin
    running on past them -- on the trainer's root section the lower
    surface just aft of the bay's trailing edge is 0.4 mm higher than
    anything inside the band, so a floor placed against the in-band
    maximum left its corner 0.255 mm from a segment it was never measured
    against. Measuring the neighbourhood is the fix; a bigger gap would
    only have hidden it.
    """
    n = (len(loop_unit) + 1) // 2
    upper = loop_unit[:n][::-1]                  # LE -> TE
    lower = loop_unit[n - 1:]
    pad = _NEIGHBOURHOOD
    sel_u = (upper[:, 0] >= x0) & (upper[:, 0] <= x1)
    sel_l = (lower[:, 0] >= x0 - pad) & (lower[:, 0] <= x1 + pad)
    if sel_u.sum() < 2 or sel_l.sum() < 2:
        return None
    y_top = float(upper[sel_u, 1].min()) - min_groove_mm / chord_mm
    y_limit = float(lower[sel_l, 1].max()) + gap_mm / chord_mm
    if y_top <= y_limit:
        return None          # not even a one-bead groove fits: no cut
    # CLAMPED, not declined. The floor is deep enough for the box where
    # the section allows it and as deep as the section allows where it
    # does not, so the geometry can never self-intersect -- and whether
    # the box actually fits is `Volume.fits`'s question, reported as a
    # gate with the shortfall in millimetres.
    #
    # Declining instead broke the point-count invariant outright: the
    # budget counts the cuts REQUESTED, so a bay that silently refused to
    # cut left the layer two vertices short of every other layer and the
    # contour array would not broadcast. The first version did exactly
    # that, on the one panel that carries two bays.
    y_floor = max(y_top - depth_mm / chord_mm, y_limit)
    return y_top, y_floor


def insert_bay(loop_unit: np.ndarray, chord_mm: float, z_mm: float,
               spec: BaySpec, min_groove_mm: float) -> np.ndarray:
    """Rebuild the upper skin around one bay detour -> longer contour.

    The upper surface is REPARAMETRISED, not spliced, for the same reason
    `ribs.insert_ribs` does it: snapping the bay's corners onto whichever
    existing vertices happen to be nearest quantises them to the contour
    grid, and a corner that jumps a grid interval as the depth ramps is a
    step in the wall the printer has to build in mid-air.
    """
    n = (len(loop_unit) + 1) // 2
    upper = loop_unit[:n]                        # TE -> LE
    lower = loop_unit[n - 1:]
    up_x, up_y = upper[::-1, 0], upper[::-1, 1]  # ascending for interp

    def y_up(x):
        return np.interp(x, up_x, up_y)

    # two segments of upper skin, with FIXED vertex counts: the count per
    # segment is what keeps the total per layer constant, and only the
    # positions inside a segment move as the bay's corners do.
    n_aft = max(int(round(n * (1.0 - spec.x1))), 2)
    n_fwd = max(n - n_aft, 2)
    if n_aft + n_fwd != n:                        # keep the budget exact
        n_aft = n - n_fwd

    t_aft = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n_aft)))
    xs_aft = 1.0 + (spec.x1 - 1.0) * t_aft        # TE -> x1
    t_fwd = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n_fwd)))
    xs_fwd = spec.x0 + (0.0 - spec.x0) * t_fwd    # x0 -> LE

    # The detour ALWAYS contributes its two vertices, and that is what
    # makes the point-count invariant structural rather than conditional.
    # Returning the loop unchanged when the section cannot hold the bay --
    # the first version -- left that layer two vertices short of every
    # other layer in the panel, and the contour array would not broadcast.
    # It only showed up on the panels carrying two bays.
    #
    # With no room the two vertices sit ON the upper skin at the bay's own
    # corners: distinct in x, collinear with the surface, geometrically a
    # no-op. The depth gate is what reports that the box does not fit.
    lim = floor_limits(loop_unit, spec.x0, spec.x1, chord_mm,
                       spec.depth_mm, spec.floor_gap_mm, min_groove_mm)
    if lim is None:
        detour = np.array([[spec.x1, float(y_up(spec.x1))],
                           [spec.x0, float(y_up(spec.x0))]])
    else:
        y_closed, y_open = lim
        floor = y_closed - spec.depth_frac(z_mm) * (y_closed - y_open)
        detour = np.array([[spec.x1, floor], [spec.x0, floor]])

    out = [np.stack([xs_aft, y_up(xs_aft)], 1),
           detour,
           np.stack([xs_fwd, y_up(xs_fwd)], 1),
           lower[1:]]
    return np.concatenate(out, axis=0)


def bay_point_budget(n_bays: int) -> int:
    return POINTS_PER_BAY * max(int(n_bays), 0)


def ramp_for(depth_mm: float, rate_mm_per_mm: float) -> float:
    """Span needed to close a bay `depth_mm` deep at the allowed rate.

    inf when the wing has already spent its whole overhang budget, which
    means the bay cannot be closed in this panel at all -- micro's centre
    body, where 33 mm is needed and 10 mm remain."""
    if rate_mm_per_mm <= 0.0:
        return float("inf")
    return float(depth_mm / rate_mm_per_mm)
