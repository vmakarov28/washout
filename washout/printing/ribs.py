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
    slit_allowance_mm: float = 0.5
    """The slit a rib carries, for spacing arithmetic. `insert_ribs` is
    handed the real value; this is what `with_layout` reserves when it
    bounds one rib's sweep against its neighbour's."""
    x_clip: tuple = (0.08, 0.88)
    """Hard limits on where a swept rib may end up, in the LOOP's own
    chord units. It was a literal (0.08, 0.88) inside `insert_ribs`, which
    is correct for a full section and wrong for a panel truncated at the
    hinge line: the loop then spans [0, x_hinge], and a rib clipped to
    0.88 lands past the cut face, where `np.interp` clamps it onto the
    closing segment and the clearance gate fails. `build_stack` narrows it
    for a truncated panel."""
    enabled: bool = True
    avoid: tuple = ()
    """Chordwise bands the ribs must not enter, as
    (centre_chord_fraction, half_width_MM) -- the spar corridors.

    The two components are deliberately different kinds, and conflating
    them was a real bug. A spar's CENTRE is a chord fraction: that is how
    `spars.place` fits it, and the tube follows the same fraction out the
    span. Its half-width is a PHYSICAL millimetre -- the tube's radius
    plus a bead plus the fit clearance -- and it does not shrink just
    because the chord does.

    Both used to be root-chord fractions, applied as local-chord
    fractions. At the root they agree; outboard the local chord is a
    third of the root's, so the corridor came out three times too narrow
    and the truss ran straight through the tube. Measured on the gen5
    trainer, the largest circle that fitted at the LE corridor was
    7.65 mm at the root and 2.73 mm at the tip, for an 8 mm spar. Every
    ribbed panel this project has exported has ribs through its spars.

    Ribs are chordwise webs running skin to skin, and they sweep with Z.
    A spanwise tube at a fixed chord fraction is therefore GUARANTEED to
    meet one somewhere unless the rib pattern is told where it is. This
    is the whole of 'cut a spar hole': there is no hole, there is a
    corridor the truss is not allowed to cross."""

    layout: tuple = ()
    """((base_chord_frac, amplitude_mm), ...), solved ONCE per panel.

    The gaps between exclusion bands are computed from the LOCAL chord, so
    they move as the chord runs out -- and the number of ribs allotted to
    each gap is an integer, so it changes in steps. Recomputing the layout
    per layer therefore made ribs hop between gaps from one layer to the
    next, which is the same quantisation failure `insert_ribs` documents
    for a third time.

    The truss is a property of the PANEL: its bases and amplitudes are
    solved at the panel's mid station and only the sweep phase varies with
    Z. `with_layout` fills this in; an empty tuple means "solve it now",
    which keeps the spec usable on its own."""
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

    def with_layout(self, chord_mm: float) -> "RibSpec":
        """Solve the truss layout once, for a whole panel."""
        from dataclasses import replace as _replace

        n = max(int(self.n_ribs), 0)
        if n <= 0 or not self.enabled:
            return _replace(self, layout=())
        gaps = free_bands(self, chord_mm)
        if not gaps:
            return _replace(self, layout=())

        widths = np.array([g1 - g0 for g0, g1 in gaps])
        share = np.maximum(np.round(n * widths / widths.sum()), 0).astype(int)
        while share.sum() > n:
            share[int(np.argmax(share))] -= 1
        while share.sum() < n:
            share[int(np.argmax(widths / np.maximum(share, 1)))] += 1

        out = []
        for (g0, g1), cnt in zip(gaps, share):
            if cnt <= 0:
                continue
            inset = 0.5 * (g1 - g0) / (cnt + 1)
            base = np.linspace(g0 + inset, g1 - inset, cnt)
            spacing = ((g1 - g0 - 2 * inset) / max(cnt - 1, 1)) * chord_mm
            # A rib must not leave its own gap, and the INSET is what
            # bounds that -- not a fraction of the gap's width. Clamping
            # the amplitude to 0.30 of the width while the inset is only
            # 0.25 of it let a rib swing 0.05 of a gap PAST the edge, into
            # the exclusion band and toward the rib in the next gap. On
            # demon1's tip panel two legs came within 0.34 mm against a
            # 0.45 mm limit, and because they were in different gaps
            # neither one's own spacing clamp could see it.
            amp = min(self.amplitude_mm(),
                      0.35 * spacing if cnt > 1 else 1e9,
                      inset * chord_mm)
            out.extend([float(b), float(amp), float(g0), float(g1)]
                       for b in base)

        # Now bound every amplitude by the room it ACTUALLY has, across
        # gaps as well as within them. Two ribs in different gaps converge
        # toward each other through the band between them, and neither
        # one's own gap can see the other: on demon1's tip panel that put
        # two legs 0.34 mm apart against a 0.45 mm limit. Adjacent ribs may
        # each travel at most half the distance between their bases, less
        # the slit they carry and a bead of clearance.
        out.sort()
        clear = 2.0 * self.slit_allowance_mm
        for a, b in zip(out, out[1:]):
            room = 0.5 * ((b[0] - a[0]) * chord_mm - clear)
            a[1] = min(a[1], max(room, 0.0))
            b[1] = min(b[1], max(room, 0.0))
        for r in out:                       # and never leave its own gap
            r[1] = min(r[1], max((r[0] - r[2]) * chord_mm - clear, 0.0),
                       max((r[3] - r[0]) * chord_mm - clear, 0.0))
        return _replace(self, layout=tuple((r[0], r[1]) for r in out))

    def stations(self, z_mm: float, chord_mm: float = 250.0) -> np.ndarray:
        """Chordwise positions of every rib at this height, as x/c.

        Ribs live INSIDE the gaps between exclusion bands. They are not
        placed evenly and then pushed clear, which is what the first two
        versions did and what failed twice for the same reason: a push has
        to choose a destination, and a destination chosen without looking
        at the other bands lands in one of them.

        Pushing to the nearer edge flips as a rib sweeps across a band's
        centre -- a 72 mm jump and 64 degrees of overhang once a payload
        bay became a band. Pushing to a side fixed by the unswept base is
        stable in Z but still blind: on the trainer's centre body the
        pack's band pushed the middle rib to 0.610c, inside the TE spar's
        corridor at 0.593c-0.637c, so the rib's legs straddled the spar
        station. The bore gate read 0.45 mm of corridor where the
        reservation said the tube was clear, and it looked like two
        measurements disagreeing rather than a third object on top of
        both.

        A gap is by construction clear of every band, so placing into one
        cannot do that. Only the sweep phase varies with Z -- the bases
        and amplitudes come from `with_layout`, solved once per panel.
        """
        spec = self if self.layout else self.with_layout(chord_mm)
        if not spec.layout:
            return np.zeros(0)
        phase = (z_mm / max(self.pitch_mm, 1e-6)) % 1.0
        tri = 4.0 * np.abs(phase - 0.5) - 1.0          # -1 .. +1
        base = np.array([b for b, _ in spec.layout])
        amp = np.array([a for _, a in spec.layout])
        sign = np.where(np.arange(len(base)) % 2 == 0, 1.0, -1.0)
        return np.sort(base + sign * tri * amp / max(chord_mm, 1e-6))


def ribs_that_fit(spec: "RibSpec", chord_mm: float,
                  slit_mm: float) -> int:
    """How many ribs the chord still has room for, given the exclusions.

    Pushing a rib out of an exclusion band works while the bands are
    narrow. It stops working when they are not: micro's centre body
    carries two payload bays that between them exclude 0.24c to 0.77c --
    nearly the whole rib band -- so all three ribs were shoved against
    the same few edges, and the result was 83 degrees of overhang against
    a 50 degree limit.

    A panel whose chord is mostly openings cannot carry a chordwise truss,
    and the honest thing is to say so and let the buckling gate price it,
    rather than to squeeze webs into a gap that is not there.

    The room a rib needs is DERIVED from the amplitude it is allowed to
    sweep, not declared: `amplitude_mm` is already `rate * pitch / 4`, the
    largest diagonal the overhang limit permits, and a rib that cannot
    travel that far chordwise is a parallel wall rather than part of a
    truss. Two amplitudes of clear chord is therefore the requirement,
    floored at three slit widths so a rib is at least wider than its own
    slit. Three slit widths ALONE was the first version's rule and it let
    three ribs into a 6.6 mm gap.
    """
    if not spec.enabled or spec.n_ribs <= 0:
        return 0
    lo, hi = spec.x_first, spec.x_last
    free = [(lo, hi)]
    for c, half_mm in spec.avoid:
        half = half_mm / max(chord_mm, 1e-6)
        a, b = c - half, c + half
        nxt = []
        for f0, f1 in free:
            if b <= f0 or a >= f1:
                nxt.append((f0, f1))
                continue
            if a > f0:
                nxt.append((f0, min(a, f1)))
            if b < f1:
                nxt.append((max(b, f0), f1))
        free = [(f0, f1) for f0, f1 in nxt if f1 > f0]
    need_mm = max(2.0 * spec.amplitude_mm(), 3.0 * slit_mm)
    need = need_mm / max(chord_mm, 1e-6)
    room = sum(int((f1 - f0) / need) for f0, f1 in free)
    return int(min(spec.n_ribs, max(room, 0)))


def free_bands(spec: "RibSpec", chord_mm: float) -> list:
    """The chord intervals a rib may occupy, exclusions removed.

    Subtracting the bands one at a time is correct even when they
    overlap, which is why placing into the result is safe where pushing
    out of the bands one at a time was not. See `stations`."""
    lo, hi = spec.x_first, spec.x_last
    free = [(lo, hi)]
    for c, half_mm in spec.avoid:
        half = half_mm / max(chord_mm, 1e-6)
        a, b = c - half, c + half
        nxt = []
        for f0, f1 in free:
            if b <= f0 or a >= f1:
                nxt.append((f0, f1))
                continue
            if a > f0:
                nxt.append((f0, min(a, f1)))
            if b < f1:
                nxt.append((max(b, f0), f1))
        free = [(f0, f1) for f0, f1 in nxt if f1 > f0]
    return free


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
    # One pass, so a skin is never re-interpolated by a second inserter:
    # doing it twice put a vertex 7.2 mm down a wall between two layers a
    # millimetre apart on the trainer's centre body.
    from .detours import insert_detours

    return insert_detours(loop_unit, chord_mm, z_mm, spec,
                          slit_mm=slit_mm, gap_mm=gap_mm)


def rib_point_budget(spec: RibSpec) -> int:
    return POINTS_PER_RIB * max(spec.n_ribs, 0) if spec.enabled else 0


def min_clearance_mm(contour_mm: np.ndarray, skip: int = 6) -> float:
    """Closest approach between non-adjacent parts of one contour.

    With ribs the old upper/lower index pairing no longer describes the
    geometry, so clearance is measured the general way: every vertex
    against every segment more than `skip` indices away. That is the rule
    a single-bead spiral actually has to obey -- the nozzle must never
    come back within one extrusion width of a pass it already laid.

    Fully vectorised. The Python loop this replaces was 38% of an entire
    design evaluation: 247 iterations of small numpy calls, 481 times per
    evaluation. One (m, m) distance matrix costs about a megabyte and is
    two orders of magnitude faster.
    """
    p = np.asarray(contour_mm, dtype=float)
    m = len(p)
    a = p
    ab = np.roll(p, -1, axis=0) - a
    denom = np.einsum("ij,ij->i", ab, ab)
    denom[denom < 1e-12] = 1e-12

    ap = p[:, None, :] - a[None, :, :]                    # (m, m, 2)
    t = np.clip(np.einsum("ijk,jk->ij", ap, ab) / denom[None, :], 0.0, 1.0)
    closest = a[None, :, :] + t[:, :, None] * ab[None, :, :]
    d = np.linalg.norm(p[:, None, :] - closest, axis=2)   # (m, m)

    idx = np.arange(m)
    sep = np.abs((idx[:, None] - idx[None, :] + m // 2) % m - m // 2)
    d[sep <= skip] = np.inf
    return float(d.min())
