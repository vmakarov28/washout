"""Internal ribs that keep the layer a SINGLE closed loop.

Spiralize mode will print exactly one contour per layer, so at first
glance the shell can have no internal structure at all: a rib joined to
the skin makes a T-junction, the curve stops being simple, and the
slicer either refuses it or quietly prints something else.

The way round it is the one commercial vase-wing tools use, and it is
worth stating precisely because the geometry is the whole trick. A rib is
not a separate loop joined to the skin -- it is a DETOUR of the skin
loop. The contour runs along the upper surface, turns down into the
interior, crosses to within one bead of the lower surface, comes back up
a slit-width away, and carries on along the upper surface. Topologically
it is still one simple closed curve. Physically the two legs are one
extrusion width apart, so they weld into a solid web, and the tip lands
one bead off the far skin and welds to that too.

    upper  ------.        .------------
                 |        |
                 |  rib   |            <- two legs, one bead apart
                 |        |
    lower  ------'--------'------------   (gap = one bead: welds, but
                                            never geometrically touches)

Two invariants make the rest of the pipeline keep working:

  * the point count per layer is CONSTANT, because the STL skinner joins
    layer k index i to layer k+1 index i. Ribs therefore exist at every
    layer -- they only move -- and the diagonal truss is what you get by
    moving them as Z rises.
  * every inserted vertex stays at least one extrusion width from every
    non-adjacent part of the contour, which is the same rule the plain
    skin already had to satisfy.

Sweeping the rib stations with Z is what turns a stack of webs into a
diamond truss: adjacent ribs travel in opposite directions, so in the
span-chord plane their paths cross, and the shell gets shear stiffness
instead of just a row of parallel walls.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

POINTS_PER_RIB = 2


@dataclass(frozen=True)
class RibSpec:
    """How the truss is laid out. `pitch_mm` is the Z period of one full
    zigzag; `n_ribs` is how many webs cross the chord."""

    n_ribs: int = 3
    pitch_mm: float = 30.0
    max_overhang_deg: float = 50.0
    """The sweep amplitude is NOT a free parameter -- it is set by this.

    A rib that slides A mm chordwise over half a pitch of Z leans at
    atan(2A / (pitch/2)) from vertical, and that is an overhang like any
    other. Specifying the sweep as a fraction of rib spacing (the obvious
    parametrization, and the first one tried) gave 20 mm of travel over
    12.5 mm of Z: a 73 degree lean, on a wall the printer has to build in
    mid-air. The amplitude is therefore DERIVED:

        A <= tan(theta_max) * pitch / 4

    which is the largest diagonal the truss can have and still print."""
    x_first: float = 0.20
    x_last: float = 0.72
    enabled: bool = True

    rate_mm_per_mm: float | None = None
    """Chordwise travel per mm of Z that the truss is allowed to use.

    This is a BUDGET, not a property of the rib, and that is the whole
    point. A tapered swept panel already moves its own section sideways
    as Z rises -- 37.9 deg of the 50 deg allowance on the trainer's root
    panel, before any rib exists. The rib's sweep ADDS to that. Sizing
    the rib against the full allowance in isolation produced 58 deg of
    real overhang and, because the excess came from the wing rather than
    the rib, it was stubbornly independent of rib pitch -- which is what
    finally gave it away after two wrong diagnoses.

    build_stack measures the bare panel first and passes the remainder."""

    def amplitude_mm(self) -> float:
        rate = (self.rate_mm_per_mm if self.rate_mm_per_mm is not None
                else np.tan(np.radians(self.max_overhang_deg)))
        return float(max(rate, 0.0) * self.pitch_mm / 4.0)

    def stations(self, z_mm: float, chord_mm: float = 250.0) -> np.ndarray:
        """Chordwise positions of every rib at this height, as x/c."""
        if self.n_ribs <= 0:
            return np.zeros(0)
        base = (np.linspace(self.x_first, self.x_last, self.n_ribs)
                if self.n_ribs > 1 else np.array([0.5 * (self.x_first + self.x_last)]))
        spacing_mm = ((self.x_last - self.x_first) / max(self.n_ribs - 1, 1)
                      * chord_mm)
        # never let adjacent ribs reach each other, whatever the pitch says
        amp_mm = min(self.amplitude_mm(), 0.35 * spacing_mm)
        phase = (z_mm / max(self.pitch_mm, 1e-6)) % 1.0
        tri = 4.0 * np.abs(phase - 0.5) - 1.0          # -1 .. +1
        sign = np.where(np.arange(self.n_ribs) % 2 == 0, 1.0, -1.0)
        return base + sign * tri * amp_mm / max(chord_mm, 1e-6)


def segment_counts(n_up: int, n_seg: int) -> list[int]:
    """How many skin vertices each inter-rib segment gets. FIXED.

    Fixed, because the count per segment is what keeps the total per
    layer constant; only the positions inside a segment are allowed to
    move as the ribs sweep."""
    base = n_up // n_seg
    counts = [base] * n_seg
    for i in range(n_up - base * n_seg):
        counts[i] += 1
    return [max(c, 2) for c in counts]


def insert_ribs(
    loop_unit: np.ndarray,
    chord_mm: float,
    z_mm: float,
    spec: RibSpec,
    slit_mm: float,
    gap_mm: float,
) -> np.ndarray:
    """Rebuild the upper skin around rib detours -> longer contour.

    The upper surface is REPARAMETRIZED, not spliced. An earlier version
    snapped each rib into whichever gap between existing vertices was
    wide enough, which quantised rib position to the contour grid: as a
    rib swept, it jumped a whole grid interval at a time -- 1.7 mm in one
    step, a 60 degree overhang on a wall the printer builds in mid-air,
    and pitch-independent, which is what gave the game away.

    Instead the skin between ribs is divided into segments with FIXED
    vertex counts, and the vertices inside each segment slide smoothly as
    the segment's ends move. Point count per layer stays constant; rib
    position becomes continuous in Z.
    """
    if not spec.enabled or spec.n_ribs <= 0:
        return loop_unit

    n = (len(loop_unit) + 1) // 2
    upper = loop_unit[:n]                    # TE -> LE, x decreasing
    lower = loop_unit[n - 1:]                # LE -> TE, x increasing
    up_x, up_y = upper[::-1, 0], upper[::-1, 1]      # ascending for interp

    slit = slit_mm / chord_mm
    gap = gap_mm / chord_mm
    xr = np.sort(np.clip(spec.stations(z_mm, chord_mm), 0.08, 0.88))[::-1]

    def y_up(x):
        return np.interp(x, up_x, up_y)

    def y_lo(x):
        return np.interp(x, lower[:, 0], lower[:, 1])

    # segment boundaries, walking TE -> LE (x descending)
    edges = [1.0]
    for x in xr:
        edges += [x + 0.5 * slit, x - 0.5 * slit]
    edges += [0.0]
    segs = [(edges[2 * i], edges[2 * i + 1]) for i in range(len(xr) + 1)]
    counts = segment_counts(n, len(segs))

    out: list[np.ndarray] = []
    for k, ((x_hi, x_lo), c) in enumerate(zip(segs, counts)):
        # cosine spacing inside the segment keeps the LE segment dense
        t = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, c)))
        xs = x_hi + (x_lo - x_hi) * t
        out.append(np.stack([xs, y_up(xs)], 1))
        if k < len(xr):                       # the rib detour itself
            xa, xb = x_lo, x_lo - slit
            floor = max(float(y_lo(xa)), float(y_lo(xb)),
                        float(y_lo(0.5 * (xa + xb)))) + gap
            out.append(np.array([[xa, floor], [xb, floor]]))
    out.append(lower[1:])
    return np.concatenate(out, axis=0)


def rib_point_budget(spec: RibSpec) -> int:
    return POINTS_PER_RIB * max(spec.n_ribs, 0) if spec.enabled else 0


def min_clearance_mm(contour_mm: np.ndarray, skip: int = 6) -> float:
    """Closest approach between non-adjacent parts of one contour.

    With ribs the old upper/lower index pairing no longer describes the
    geometry, so clearance is measured the general way: every vertex
    against every segment more than `skip` indices away. That is the rule
    a single-bead spiral actually has to obey -- the nozzle must never
    come back within one extrusion width of a pass it already laid."""
    p = np.asarray(contour_mm, dtype=float)
    m = len(p)
    a = p
    b = np.roll(p, -1, axis=0)
    ab = b - a
    denom = np.einsum("ij,ij->i", ab, ab)
    denom[denom < 1e-12] = 1e-12

    best = np.inf
    for i in range(m):
        d_idx = np.abs((np.arange(m) - i + m // 2) % m - m // 2)
        mask = d_idx > skip
        if not mask.any():
            continue
        ap = p[i][None, :] - a[mask]
        t = np.clip(np.einsum("ij,ij->i", ap, ab[mask]) / denom[mask], 0.0, 1.0)
        closest = a[mask] + t[:, None] * ab[mask]
        d = np.linalg.norm(p[i][None, :] - closest, axis=1).min()
        best = min(best, float(d))
    return best
