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


def depth_mm(plan, eta: float, x_frac, wall_mm: float):
    """Usable internal depth at one station, both skins removed.

    The same quantity `spars.depth_at` computes, kept here so this module
    does not depend on the spar fitter it is meant to serve.

    `x_frac` may be an array of chord stations at the one span station,
    and then an array comes back. The checks below ask for five or seven
    chord stations at every span station they visit, and asking one at a
    time made this function 30% of a design evaluation."""
    st = plan.at(float(np.clip(eta, 0.0, 1.0)))
    x = np.clip(np.asarray(x_frac, dtype=float), 0.0, 1.0)
    t = st.airfoil.thickness(np.atleast_1d(x)) * st.chord_m * 1000.0 - 2.0 * wall_mm
    return float(t[0]) if x.ndim == 0 else t


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
    length_mm: float | None = None
    """The part's chordwise extent in MILLIMETRES, when it has one.

    `x0`/`x1` are root-chord fractions -- the datum every chord fraction
    in the program uses -- and they are the right coordinates for a
    centre. They are the wrong coordinates for a WIDTH: applied at a
    station whose chord is half the root's, a 23 mm servo's band came out
    11.5 mm wide, and the pocket that was cut from it narrowed outboard
    across its own span (17.8 mm at its inboard end, 15.5 at its
    outboard). A box does not get shorter because the chord did. With a
    length, `band()` gives the part's true extent at any station."""

    x_abs_mm: float | None = None
    """The part's centre as an ABSOLUTE station, mm aft of the root
    leading edge, when it has one. A rigid box does not follow the chord
    fraction as the wing sweeps and tapers under it: with a physical
    width on a fractional centre, the trainer's two root bays -- 5.6 mm
    apart at the root -- overlapped by 0.9 mm where their grooves ran
    out at z = 81 mm. A tube DOES follow a chord fraction (that is how
    `spars.place` fits it), so this stays None for a spar."""

    def band(self, plan, eta: float) -> tuple[float, float]:
        """(x0, x1) as fractions of the LOCAL chord at this station."""
        if self.length_mm is None:
            return (self.x0, self.x1)
        st = plan.at(float(np.clip(eta, 0.0, 1.0)))
        c_mm = st.chord_m * 1000.0
        h = 0.5 * self.length_mm / max(c_mm, 1e-9)
        if self.x_abs_mm is not None:
            mid = (self.x_abs_mm - st.x_le_m * 1000.0) / max(c_mm, 1e-9)
            return (mid - h, mid + h)
        return (self.x_mid - h, self.x_mid + h)

    def __post_init__(self) -> None:
        if self.anchor not in ANCHORS:
            raise ValueError(f"{self.name}: anchor {self.anchor!r} "
                             f"not one of {ANCHORS}")

    @property
    def x_mid(self) -> float:
        return 0.5 * (self.x0 + self.x1)

    def z_interval(self, plan, eta: float, x_frac,
                   wall_mm: float):
        """(lo, hi) in mm above the lower inner skin at this station.

        Returns an EMPTY interval (lo > hi) where the section is too
        shallow to hold the part at all, so a caller that asks about a
        station outboard of where the part fits gets "nothing here"
        rather than a silently clipped answer.

        With an array of chord stations, (lo, hi) are arrays, empty
        element by element."""
        d = depth_mm(plan, eta, x_frac, wall_mm)
        if np.ndim(d) == 0:
            if d < self.height_mm + self.offset_mm:
                return (1.0, 0.0)
            lo = self._lo(d)
            return (lo, lo + self.height_mm)
        lo = self._lo(d) + np.zeros_like(d)
        hi = lo + self.height_mm
        short = d < self.height_mm + self.offset_mm
        return np.where(short, 1.0, lo), np.where(short, 0.0, hi)

    def _lo(self, d):
        if self.anchor == LOWER:
            return self.offset_mm
        if self.anchor == UPPER:
            return d - self.height_mm - self.offset_mm
        return 0.5 * (d - self.height_mm)

    def fits(self, plan, wall_mm: float, n_eta: int = 7,
             n_x: int = 5) -> tuple[bool, float]:
        """Does the part fit inside the shell over its whole extent?

        -> (ok, worst_spare_mm). Negative spare is how much too deep it
        is at its tightest station, which is what a penalty needs: a box
        1 mm too deep must rank above one 10 mm too deep."""
        worst = np.inf
        for eta in np.linspace(self.eta0, self.eta1, n_eta):
            x0, x1 = self.band(plan, float(eta))
            d = depth_mm(plan, float(eta), np.linspace(x0, x1, n_x), wall_mm)
            worst = min(worst, float((d - self.height_mm - self.offset_mm).min()))
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
    if e1 < e0:
        return 0.0                       # no shared span at all

    worst = 0.0
    de_mm = (min(a.eta1, b.eta1) - max(a.eta0, b.eta0)) \
        * plan.half_span_m * 1000.0
    for eta in np.linspace(e0, e1, n_eta):
        # the bands are LOCAL to this station: a part with a physical
        # length is wider, as a fraction, where the chord is shorter
        ax, bx = a.band(plan, float(eta)), b.band(plan, float(eta))
        x0, x1 = max(ax[0], bx[0]), min(ax[1], bx[1])
        if x1 < x0:
            continue                     # no shared chord here
        xs = np.linspace(x0, x1, n_x)
        alo, ahi = a.z_interval(plan, float(eta), xs, wall_mm)
        blo, bhi = b.z_interval(plan, float(eta), xs, wall_mm)
        dz = np.minimum(ahi, bhi) - np.maximum(alo, blo)
        # one of them does not fit here, or they are stacked clear
        hit = (ahi >= alo) & (bhi >= blo) & (dz > 0.0)
        if not hit.any():
            continue
        # the separation is the cheapest axis to move along
        dx_mm = (x1 - x0) * plan.at(float(eta)).chord_m * 1000.0
        worst = max(worst, float(np.minimum(dz[hit], min(dx_mm, de_mm)).max()))
    return float(worst)


def gap_mm(a: Volume, b: Volume, plan, n_eta: int = 5) -> float:
    """Chordwise clearance between two volumes' bands, in mm, at the
    tightest of their shared stations. Negative means they overlap in
    chord. Two openings side by side must leave a WALL between them,
    not a fin, and the wall is a physical width wherever it is."""
    e0, e1 = max(a.eta0, b.eta0), min(a.eta1, b.eta1)
    if e1 < e0:
        return float("inf")
    worst = float("inf")
    for eta in np.linspace(e0, e1, n_eta):
        ax, bx = a.band(plan, float(eta)), b.band(plan, float(eta))
        c_mm = plan.at(float(eta)).chord_m * 1000.0
        worst = min(worst, max(ax[0] - bx[1], bx[0] - ax[1]) * c_mm)
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
               anchor: str = LOWER, offset_mm: float = 0.0,
               eta_frac: float | None = None) -> Volume:
    """A payload box, as the interior region it actually occupies.

    `box_mm` is (length, width, height) as `search.design.Bay` declares
    it: length is chordwise, width is SPANWISE, height is through the
    section.

    `eta_frac` is where the box is CENTRED along the span. None means the
    centreline, which is where payload in a blended body sits: the box
    then reaches eta = width / (2 * half_span) and we model the right half,
    because the aircraft is symmetric and so is the clash. A servo is not
    centreline payload -- it has to sit next to the surface it drives --
    so it is centred out at its own station and occupies eta_frac +/- half
    its width.

    Chordwise extent is expressed against the ROOT chord because that is
    the datum every other chord fraction in the program uses; the box does
    not get shorter because the local chord did.
    """
    length, width, height = box_mm
    root_c_mm = plan.stations[0].chord_m * 1000.0
    half_mm = half_span_mm or plan.half_span_m * 1000.0
    de = 0.5 * width / max(half_mm, 1e-9)
    if eta_frac is None:
        e0, e1 = 0.0, min(de, 1.0)
    else:
        e0, e1 = max(float(eta_frac) - de, 0.0), min(float(eta_frac) + de, 1.0)
    # `x_frac` is a fraction of the chord AT THE BOX'S OWN STATION -- the
    # root for centreline payload, its solved eta for a servo -- and the
    # box then sits at that absolute x across its whole width. `x0`/`x1`
    # keep the root-chord form for the callers that want a centre.
    st = plan.at(0.0 if eta_frac is None else float(eta_frac))
    x_abs = st.x_le_m * 1000.0 + x_frac * st.chord_m * 1000.0
    dx = 0.5 * length / max(root_c_mm, 1e-9)
    mid = x_abs / max(root_c_mm, 1e-9)
    return Volume(name=name, x0=mid - dx, x1=mid + dx,
                  eta0=e0, eta1=e1,
                  height_mm=height, anchor=anchor, offset_mm=offset_mm,
                  length_mm=float(length), x_abs_mm=float(x_abs))


def skin_z_mm(plan, eta: float, x_mm, which: str = LOWER, n: int = 81):
    """Flight z of one skin of the placed section at eta, at flight x.

    The section as it flies: twisted about its quarter chord, scaled,
    swept and lifted by its dihedral. Linear between cosine-spaced chord
    stations, which is within a few microns on these sections."""
    st = plan.at(float(np.clip(eta, 0.0, 1.0)))
    c = st.chord_m * 1000.0
    u = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n)))
    v = st.airfoil.y_upper(u) if which == UPPER else st.airfoil.y_lower(u)
    a = np.radians(-st.twist_deg)
    xs = st.x_le_m * 1000.0 + 0.25 * c + c * ((u - 0.25) * np.cos(a) - v * np.sin(a))
    zs = st.z_le_m * 1000.0 + c * ((u - 0.25) * np.sin(a) + v * np.cos(a))
    o = np.argsort(xs)
    return np.interp(np.asarray(x_mm, dtype=float), xs[o], zs[o])


@dataclass(frozen=True)
class TubeVolume:
    """A straight tube, as the interior region it occupies.

    A `Volume` keeps a fixed chord fraction and a fixed seat against one
    skin, which is exactly the bendable spar ROADMAP-CAD.md section 0.1
    found: a carbon tube is straight, so on a swept, dihedralled wing its
    chord fraction drifts and its height above either skin changes along
    the span. This carries the line itself -- flight x and z at the
    centreline and their rates per mm of span -- and answers `band` and
    `z_interval` from it, so every clash check that takes a `Volume`
    takes a tube.

    In a `y = const` plane a cylinder whose axis runs along (sx, 1, sz) is
    an ELLIPSE, not a circle: r sqrt(1 + sx^2) wide along the chord and
    r sqrt(1 + sz^2) tall. Its extent at any chord station is the root of

        (1 + sz^2) dx^2 - 2 sx sz dx dz + (1 + sx^2) dz^2 = r^2 (1 + sx^2 + sz^2)
    """

    name: str
    root_xz_mm: tuple
    slope: tuple
    radius_mm: float
    """The RESERVED radius: tube plus its fit clearance."""
    eta0: float = 0.0
    eta1: float = 1.0
    anchor: str = "tube"

    @property
    def height_mm(self) -> float:
        return 2.0 * self.radius_mm

    def centre_mm(self, plan, eta: float) -> tuple[float, float]:
        y = float(eta) * plan.half_span_m * 1000.0
        return (self.root_xz_mm[0] + self.slope[0] * y,
                self.root_xz_mm[1] + self.slope[1] * y)

    def band(self, plan, eta: float) -> tuple[float, float]:
        st = plan.at(float(np.clip(eta, 0.0, 1.0)))
        c = st.chord_m * 1000.0
        xc, _ = self.centre_mm(plan, eta)
        hx = self.radius_mm * np.sqrt(1.0 + self.slope[0] ** 2)
        x_le = st.x_le_m * 1000.0
        return ((xc - hx - x_le) / c, (xc + hx - x_le) / c)

    def z_interval(self, plan, eta: float, x_frac, wall_mm: float):
        st = plan.at(float(np.clip(eta, 0.0, 1.0)))
        c = st.chord_m * 1000.0
        x = st.x_le_m * 1000.0 + np.asarray(x_frac, dtype=float) * c
        xc, zc = self.centre_mm(plan, eta)
        sx, sz = self.slope
        dx = x - xc
        A = 1.0 + sx * sx
        B = -2.0 * sx * sz * dx
        C = (1.0 + sz * sz) * dx * dx - self.radius_mm ** 2 * (1.0 + sx * sx + sz * sz)
        disc = B * B - 4.0 * A * C
        root = np.sqrt(np.maximum(disc, 0.0))
        floor = skin_z_mm(plan, eta, x, LOWER) + wall_mm
        lo = zc + (-B - root) / (2.0 * A) - floor
        hi = zc + (-B + root) / (2.0 * A) - floor
        empty = disc < 0.0
        if np.ndim(lo) == 0:
            return (1.0, 0.0) if empty else (float(lo), float(hi))
        return np.where(empty, 1.0, lo), np.where(empty, 0.0, hi)


def spar_volume(fit, wall_mm: float, plan):
    """A fitted spar, as the interior region it occupies: the straight
    tube along its own line when the fit has one, which every fit from
    `spars.place` does."""
    if getattr(fit, "slope", None) is not None and fit.root_xz_mm is not None:
        return TubeVolume(name=fit.spec.name, root_xz_mm=tuple(fit.root_xz_mm),
                          slope=tuple(fit.slope),
                          radius_mm=0.5 * fit.spec.d_mm + fit.spec.clearance_mm,
                          eta0=0.0, eta1=float(fit.reach_eta))
    return _chord_fraction_volume(fit, wall_mm, plan)


def _chord_fraction_volume(fit, wall_mm: float, plan) -> Volume:
    """A fitted spar tube, as the interior region it occupies.

    Chordwise extent is the tube's own diameter about the station the fit
    solved for -- a tube is round, so it occupies chord as well as depth,
    and treating it as a line at `x_frac` is how it stayed invisible to
    the bay check for five generations.
    """
    root_c_mm = plan.stations[0].chord_m * 1000.0
    d = fit.spec.d_mm
    # Chordwise extent is the tube's diameter PLUS its fit clearance on
    # each side, for the same reason the bore gate measures an inscribed
    # circle: a tube is round, and it needs its clearance in every
    # direction, not only through the depth. Without it a servo pocket
    # whose wall stood 4 mm from an 8 mm tube's centre passed the volume
    # check -- no shared x -- while the bore gate, which asks whether a
    # circle fits, rightly reported 4 mm of corridor. The reservation has
    # to be at least as strict as the gate, or the solver seats tubes the
    # gate then rejects.
    half_mm = 0.5 * d + fit.spec.clearance_mm
    dx = half_mm / max(root_c_mm, 1e-9)
    # And in z the same: the tube PLUS its clearance, flush against the
    # skin it seats on. Reserving the bare diameter with half a clearance
    # of offset left the reservation a rectangle that cleared a servo
    # pocket's ceiling by 0.2 mm while the bore gate -- a circle -- caught
    # the pocket's top corner and reported 7.66 mm of corridor for an
    # 8 mm tube. The reservation has to be at least as strict as the gate
    # in every direction, or the solver seats tubes the gate then
    # rejects. d + 2c here against d + wall + c at the gate: strictly
    # stricter, by a bead less a clearance.
    return Volume(name=fit.spec.name, x0=fit.x_frac - dx, x1=fit.x_frac + dx,
                  eta0=0.0, eta1=float(fit.reach_eta),
                  height_mm=d + 2.0 * fit.spec.clearance_mm,
                  anchor=getattr(fit, "anchor", MID), offset_mm=0.0,
                  length_mm=2.0 * half_mm)
