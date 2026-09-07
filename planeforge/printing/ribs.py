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

POINTS_PER_RIB = 4


@dataclass(frozen=True)
class RibSpec:
    """How the truss is laid out. `pitch_mm` is the Z period of one full
    zigzag; `n_ribs` is how many webs cross the chord."""

    n_ribs: int = 3
    pitch_mm: float = 30.0
    sweep_frac: float = 0.55
    """How far each rib slides chordwise over one period, as a fraction
    of the spacing between rib stations. 0 gives parallel walls; ~0.5
    makes adjacent ribs cross into diamonds without letting any two ribs
    collide."""
    x_first: float = 0.20
    x_last: float = 0.72
    enabled: bool = True

    def stations(self, z_mm: float) -> np.ndarray:
        """Chordwise positions of every rib at this height."""
        if self.n_ribs <= 0:
            return np.zeros(0)
        base = (np.linspace(self.x_first, self.x_last, self.n_ribs)
                if self.n_ribs > 1 else np.array([0.5 * (self.x_first + self.x_last)]))
        spacing = (self.x_last - self.x_first) / max(self.n_ribs - 1, 1)
        # triangle wave in Z, alternating sign per rib -> crossing diagonals
        phase = (z_mm / max(self.pitch_mm, 1e-6)) % 1.0
        tri = 4.0 * np.abs(phase - 0.5) - 1.0          # -1 .. +1
        sign = np.where(np.arange(self.n_ribs) % 2 == 0, 1.0, -1.0)
        return base + sign * tri * 0.5 * self.sweep_frac * spacing


def insert_ribs(
    loop_unit: np.ndarray,
    chord_mm: float,
    z_mm: float,
    spec: RibSpec,
    slit_mm: float,
    gap_mm: float,
) -> np.ndarray:
    """Add rib detours to a unit-chord contour -> longer contour.

    The loop arrives in Selig order: upper surface traversed TE -> LE
    (x decreasing), then lower surface LE -> TE. Ribs hang from the
    upper surface, so they are spliced into the first half, and because
    that half runs backwards in x the detour is entered at the HIGH-x leg
    and left at the low-x one.
    """
    if not spec.enabled or spec.n_ribs <= 0:
        return loop_unit

    n = (len(loop_unit) + 1) // 2
    upper = loop_unit[:n]                    # TE -> LE, x decreasing
    lower = loop_unit[n - 1:]                # LE -> TE, x increasing

    slit = slit_mm / chord_mm
    gap = gap_mm / chord_mm
    want = np.clip(spec.stations(z_mm), spec.x_first, spec.x_last)

    def y_lower_at(x: float) -> float:
        return float(np.interp(x, lower[:, 0], lower[:, 1]))

    def y_upper_at(x: float) -> float:
        return float(np.interp(x, upper[::-1, 0], upper[::-1, 1]))

    # Snap each rib into a GAP BETWEEN existing vertices wide enough to
    # hold the slit. Inserting without deleting is what keeps the point
    # count per layer constant, and the STL skinner joins layer k index i
    # to layer k+1 index i -- so a layer that dropped a vertex because a
    # rib happened to land on one would shear the whole mesh.
    widths = upper[:-1, 0] - upper[1:, 0]              # x decreases
    need = slit + 2.0 * (slit * 0.25)
    ok_idx = np.flatnonzero(widths > need)
    if len(ok_idx) == 0:
        return loop_unit                                # too coarse: no ribs

    chosen: list[int] = []
    for xr in want:
        mids = 0.5 * (upper[ok_idx, 0] + upper[ok_idx + 1, 0])
        order = np.argsort(np.abs(mids - xr))
        for k in order:                                 # never reuse a slot
            j = int(ok_idx[k])
            if j not in chosen:
                chosen.append(j)
                break
    chosen.sort()                                       # ascending index = x descending

    out: list[np.ndarray] = []
    prev = 0
    for j in chosen:
        out.extend(upper[prev:j + 1])
        mid = 0.5 * (upper[j, 0] + upper[j + 1, 0])
        xa, xb = mid + 0.5 * slit, mid - 0.5 * slit
        # Clear the HIGHEST point of the lower surface across the rib's
        # whole footprint, not the surface at its midpoint. The skin
        # slopes, and over a 0.45 mm footprint that slope ate 0.03 mm of
        # a 0.45 mm gap -- enough to fail the clearance rule by exactly
        # the amount the measurement said, at every chord.
        y_floor = max(y_lower_at(xa), y_lower_at(mid), y_lower_at(xb)) + gap
        out.extend([np.array([xa, y_upper_at(xa)]),
                    np.array([xa, y_floor]),
                    np.array([xb, y_floor]),
                    np.array([xb, y_upper_at(xb)])])
        prev = j + 1
    out.extend(upper[prev:n])
    out.extend(lower[1:])
    return np.asarray(out, dtype=float)


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
