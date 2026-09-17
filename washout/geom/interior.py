"""What occupies the inside of the wing, and whether two things want the
same space.

Three separate checks used to ask three separate questions and none of
them knew the others existed:

    bay_fits          is the section deep enough here?
    spars.fit_all     can a tube reach that far out?
    vase.panel_etas   does this panel fit the print envelope?

So nothing ever asked whether the tube and the battery wanted the same
place. They did, on every aircraft in the fleet: the trainer's TE spar at
0.540c runs through a pack occupying 0.291c-0.586c, demon1's LE spar at
0.210c through a pack occupying 0.197c-0.570c, and micro's single spar at
0.363c through both its pack and its receiver.

It was not detectable before this module, and not because nobody looked.
`spars.place` solves a chordwise station and nothing else, so **the spar
had no vertical coordinate at all** -- the tube was "somewhere in the
cavity", and a question about whether two objects overlap cannot be asked
of an object with two coordinates and a shrug.

## The frame

A point inside the shell is (eta, x, z):

    eta   span fraction, 0 at the centreline
    x     chord fraction at that eta
    z     MILLIMETRES above the lower inner skin surface, at that (eta, x)

The third coordinate is the one that makes intersection tests mean
something. It is measured from the lower inner skin rather than from the
chord line because that is the surface a part actually rests on, and
because the usable depth

    depth(eta, x) = thickness(x) * chord(eta) - 2 * wall

is then exactly the interval [0, depth] that everything must live inside.
Two objects at the same (eta, x) are measured from the same datum, so
their z intervals are directly comparable. Objects at different chord
stations are never compared, because they cannot collide.

## Why anchored rather than absolute

A `Volume` does not carry an absolute z. It carries an anchor -- which
skin it rests against -- an offset from that skin, and a height. The
section thins outboard and it thins toward both edges of the chord, so a
tube 6 mm above the lower skin at the root is *through* the upper skin at
eta 0.8. Anchoring follows the surface the part is really resting on and
turns "does it fit" into one subtraction at every station.

`mid` exists for the case where a part is centred in the depth, which is
what an unseated spar was implicitly assuming.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LOWER = "lower"
UPPER = "upper"
MID = "mid"
ANCHORS = (LOWER, UPPER, MID)


def depth_mm(plan, eta: float, x_frac: float, wall_mm: float) -> float:
    """Usable internal depth at one station, both skins removed.

    The same quantity `spars.depth_at` computes, kept here so this module
    does not depend on the spar fitter it is meant to serve."""
    st = plan.at(float(np.clip(eta, 0.0, 1.0)))
    t = float(st.airfoil.thickness(np.array([float(np.clip(x_frac, 0.0, 1.0))]))[0])
    return t * st.chord_m * 1000.0 - 2.0 * wall_mm


@dataclass(frozen=True)
class Volume:
    """A region of the interior that something already occupies.

    Chordwise and spanwise extents are fractions; the vertical extent is
    a height in mm placed against an anchor skin. `eta0` is normally 0 --
    payload sits at the centreline -- but a servo bay does not, which is
    why it is a field rather than an assumption.
    """

    name: str
    x0: float
    x1: float
    eta0: float
    eta1: float
    height_mm: float
    anchor: str = LOWER
    offset_mm: float = 0.0
    """Clearance between the anchor skin and the part. A printed bore
    carries first-layer squish and a little sag, and a bay floor is a
    bead thick, so nothing sits at exactly zero."""

    def __post_init__(self) -> None:
        if self.anchor not in ANCHORS:
            raise ValueError(f"{self.name}: anchor {self.anchor!r} "
                             f"not one of {ANCHORS}")

    @property
    def x_mid(self) -> float:
        return 0.5 * (self.x0 + self.x1)

    def z_interval(self, plan, eta: float, x_frac: float,
                   wall_mm: float) -> tuple[float, float]:
        """(lo, hi) in mm above the lower inner skin at this station.

        Returns an EMPTY interval (lo > hi) where the section is too
        shallow to hold the part at all, so a caller that asks about a
        station outboard of where the part fits gets "nothing here"
        rather than a silently clipped answer."""
        d = depth_mm(plan, eta, x_frac, wall_mm)
        if d < self.height_mm + self.offset_mm:
            return (1.0, 0.0)
        if self.anchor == LOWER:
            lo = self.offset_mm
        elif self.anchor == UPPER:
            lo = d - self.height_mm - self.offset_mm
        else:
            lo = 0.5 * (d - self.height_mm)
        return (lo, lo + self.height_mm)

    def fits(self, plan, wall_mm: float, n_eta: int = 7,
             n_x: int = 5) -> tuple[bool, float]:
        """Does the part fit inside the shell over its whole extent?

        -> (ok, worst_spare_mm). Negative spare is how much too deep it
        is at its tightest station, which is what a penalty needs: a box
        1 mm too deep must rank above one 10 mm too deep."""
        worst = np.inf
        for eta in np.linspace(self.eta0, self.eta1, n_eta):
            for x in np.linspace(self.x0, self.x1, n_x):
                d = depth_mm(plan, float(eta), float(x), wall_mm)
                worst = min(worst, d - self.height_mm - self.offset_mm)
        return bool(worst >= 0.0), float(worst)


def overlap_mm(a: Volume, b: Volume, plan, wall_mm: float,
               n_eta: int = 9, n_x: int = 5) -> float:
    """How far two volumes interpenetrate, in mm. 0.0 means clear.

    The returned number is the smallest of the three axis overlaps at the
    worst sampled station -- the minimum translation that would separate
    them along one axis. It is a severity, not a volume: the optimizer
    needs to know that a 1 mm clash is nearly solved and a 12 mm clash is
    not, and a shared volume in mm^3 does not say that.

    Sampled rather than solved. The section changes smoothly along both
    eta and x, so a clash deep enough to matter is never invisible
    between samples, and an exact intersection of two regions bounded by
    an interpolated aerofoil is not worth its own solver.
    """
    e0, e1 = max(a.eta0, b.eta0), min(a.eta1, b.eta1)
    x0, x1 = max(a.x0, b.x0), min(a.x1, b.x1)
    if e1 < e0 or x1 < x0:
        return 0.0                       # no shared chord or span at all

    worst = 0.0
    for eta in np.linspace(e0, e1, n_eta):
        for x in np.linspace(x0, x1, n_x):
            alo, ahi = a.z_interval(plan, float(eta), float(x), wall_mm)
            blo, bhi = b.z_interval(plan, float(eta), float(x), wall_mm)
            if ahi < alo or bhi < blo:
                continue                 # one of them does not fit here
            dz = min(ahi, bhi) - max(alo, blo)
            if dz <= 0.0:
                continue                 # stacked clear of each other
            # the separation is the cheapest axis to move along
            dx = (min(a.x1, b.x1) - max(a.x0, b.x0))
            st = plan.at(float(eta))
            dx_mm = dx * st.chord_m * 1000.0
            de_mm = (min(a.eta1, b.eta1) - max(a.eta0, b.eta0)) \
                * plan.half_span_m * 1000.0
            worst = max(worst, min(dz, dx_mm, de_mm))
    return float(worst)


def clashes(volumes, plan, wall_mm: float) -> list:
    """Every pair that wants the same space. -> [(name_a, name_b, mm), ...]

    Ordered worst first, because a report that leads with a 0.3 mm graze
    buries the 12 mm tube through the battery."""
    out = []
    vols = list(volumes)
    for i, a in enumerate(vols):
        for b in vols[i + 1:]:
            mm = overlap_mm(a, b, plan, wall_mm)
            if mm > 0.0:
                out.append((a.name, b.name, mm))
    return sorted(out, key=lambda t: -t[2])


def straddles(vol: Volume, panel_etas) -> tuple[int, int]:
    """Which printed panels a volume spans. -> (first, last) panel index.

    A part reaching across a panel joint is a part in two separately
    printed shells, and there is no geometry to make it work. Micro's
    2S 450 does exactly this: the pack reaches eta 0.170 and panel p0
    ends at 0.140, and until this function nothing noticed.
    """
    first = last = -1
    for i, (a, b) in enumerate(panel_etas):
        if vol.eta1 >= a and vol.eta0 <= b:
            if first < 0:
                first = i
            last = i
    return first, last


def bay_volume(name: str, x_frac: float, box_mm, plan,
               half_span_mm: float | None = None,
               anchor: str = LOWER, offset_mm: float = 0.0) -> Volume:
    """A payload box, as the interior region it actually occupies.

    `box_mm` is (length, width, height) as `search.design.Bay` declares
    it: length is chordwise, width is SPANWISE, height is through the
    section. The box is centred on the centreline, so it reaches
    eta = width / (2 * half_span) on each side and we model the right
    half -- the aircraft is symmetric and so is the clash.
    """
    length, width, height = box_mm
    root_c_mm = plan.stations[0].chord_m * 1000.0
    half_mm = half_span_mm or plan.half_span_m * 1000.0
    dx = 0.5 * length / max(root_c_mm, 1e-9)
    return Volume(name=name, x0=x_frac - dx, x1=x_frac + dx,
                  eta0=0.0, eta1=min(0.5 * width / max(half_mm, 1e-9), 1.0),
                  height_mm=height, anchor=anchor, offset_mm=offset_mm)


def spar_volume(fit, wall_mm: float, plan) -> Volume:
    """A fitted spar tube, as the interior region it occupies.

    Chordwise extent is the tube's own diameter about the station the fit
    solved for -- a tube is round, so it occupies chord as well as depth,
    and treating it as a line at `x_frac` is how it stayed invisible to
    the bay check for five generations.
    """
    root_c_mm = plan.stations[0].chord_m * 1000.0
    d = fit.spec.d_mm
    dx = 0.5 * d / max(root_c_mm, 1e-9)
    return Volume(name=fit.spec.name, x0=fit.x_frac - dx, x1=fit.x_frac + dx,
                  eta0=0.0, eta1=float(fit.reach_eta),
                  height_mm=d, anchor=getattr(fit, "anchor", MID),
                  offset_mm=0.5 * fit.spec.clearance_mm)
