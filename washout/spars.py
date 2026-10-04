"""Spanwise spars: where they go, how far they reach, what blocks them.

A spar runs along the span, which after the print-orientation trick is
the print Z axis -- so it needs no hole through the single wall, and the
shell's own cavity is its bore. That gift comes with three problems the
rest of the program did not have to face:

  1. The wing THINS outboard. Past some station the section is shallower
     than the tube and the spar simply stops. Where it stops is not a
     detail: it decides whether the outer panel is joined to the aircraft
     or merely resting against it.

  2. Every panel joint the spar crosses must be deep enough to pass it.
     The root of each section therefore has to clear the tube -- if it
     does not, the spar cannot reach that panel at all, however deep the
     panel gets further out.

  3. RIBS BLOCK IT. The internal truss runs chordwise from skin to skin
     at stations that sweep with Z, so a spanwise tube at a fixed chord
     fraction is guaranteed to collide with a rib somewhere. The corridor
     has to be kept clear, which means the rib pattern must be told where
     the spars are.

The last panel is the exception, deliberately: it is allowed to run out
of depth part-way, because insisting the spar reach the tip would force
the whole wing thick to satisfy its thinnest 10%. What matters is that
it penetrates far enough to carry the joint.

Placement is SOLVED, not assumed: within an allowed chordwise band, the
station that reaches furthest outboard wins, because reach is the thing
that decides whether the joint works.

## A tube is straight

Until 2026-09-22 the fit asked whether the section was deep enough at one
chord fraction, span station by span station -- which let the tube follow
its corridor aft with the sweep and up with the dihedral, as if it bent.
A carbon tube is straight. Measured on the loft, the best straight tube
per half reached eta 0.66-0.69 where that fit claimed 0.76-1.00, and the
"one tube tip to tip" the build sheet asked for left the skin at eta
0.25-0.43 on every aircraft (ROADMAP-CAD.md section 0.1).

So a spar is now a LINE per half wing: a seat at the centreline and a
direction, sweep and dihedral both. The two halves meet in a V joiner at
the centreline whose angles are twice the tube's; a line with neither is
one tube tip to tip. Reach is where the tube's section -- an ellipse in
any `y = const` plane, because the tube crosses it at an angle -- first
leaves the section with a bead of wall around it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SparSpec:
    """What is being fitted, and where it is allowed to sit."""

    name: str
    d_mm: float = 8.0
    x_lo: float = 0.15          # chordwise band the solver may use
    x_hi: float = 0.38
    clearance_mm: float = 0.8
    """Radial slop between tube and cavity. The bore is printed, so it
    carries the usual first-layer squish and a little sag."""

    def needed_mm(self, wall_mm: float) -> float:
        return self.d_mm + 2.0 * wall_mm + self.clearance_mm


@dataclass
class SparFit:
    spec: SparSpec
    x_frac: float               # chosen chordwise station
    reach_eta: float            # outermost span fraction it can occupy
    reach_mm: float             # measured along the span from the root
    depth_at_root_mm: float
    joints_cleared: tuple[float, ...]
    joints_blocked: tuple[float, ...]
    anchor: str = "mid"
    """Which inner skin the tube rests against: "lower", "upper" or "mid".

    The spar had NO vertical coordinate at all until this field. `place`
    solved a chordwise station, `depth_at` reported the thickness there,
    and the tube was "somewhere in the cavity" -- so the question "is the
    spar inside the battery" could not be asked, and the answer on every
    aircraft in the fleet turned out to be yes.

    It is chosen on CLEARANCE grounds, not structural ones, and that is
    deliberate rather than a shortcut: `structure.select` computes bending
    from the tube's own second moment with no offset-from-neutral-axis
    term, so the vertical seat does not enter the strength calculation at
    all. Choosing it to keep the payload's volume free is therefore honest;
    claiming it was chosen for stiffness would not be."""
    clash_mm: float = 0.0
    """Worst interpenetration with a reserved volume at the chosen seat,
    0.0 if clear. Carried out rather than raised, so the search can
    penalise it by HOW FAR it misses."""
    root_xz_mm: tuple | None = None
    """Flight x and z of the tube's axis at the centreline, mm; x from the
    root leading edge. With `slope`, the whole tube."""
    slope: tuple | None = None
    """(dx/dy, dz/dy) of the axis: the tube's sweep and dihedral as
    rates. Both zero is one tube tip to tip."""

    @property
    def ok(self) -> bool:
        return not self.joints_blocked and self.clash_mm <= 0.0

    def centre_mm(self, y_mm):
        """Flight (x, z) of the axis at span position y, mm."""
        x0, z0 = self.root_xz_mm
        sx, sz = self.slope
        y = np.asarray(y_mm, dtype=float)
        return x0 + sx * y, z0 + sz * y

    @property
    def reach_y_mm(self) -> float:
        """How far out the tube goes, measured along the SPAN."""
        sx, sz = self.slope or (0.0, 0.0)
        return float(self.reach_mm) / float(np.sqrt(1.0 + sx * sx + sz * sz))

    @property
    def sweep_deg(self) -> float:
        return float(np.degrees(np.arctan((self.slope or (0.0, 0.0))[0])))

    @property
    def dihedral_deg(self) -> float:
        return float(np.degrees(np.arctan((self.slope or (0.0, 0.0))[1])))

    @property
    def one_piece(self) -> bool:
        """Parallel to the span: one tube tip to tip, no joiner."""
        return abs(self.sweep_deg) < 0.05 and abs(self.dihedral_deg) < 0.05

    def x_frac_at(self, plan, eta: float) -> float:
        """The axis' chord fraction at a span station. It drifts along a
        swept, tapered wing -- that is the whole point of the line."""
        st = plan.at(float(np.clip(eta, 0.0, 1.0)))
        x, _ = self.centre_mm(eta * plan.half_span_m * 1000.0)
        return float((x - st.x_le_m * 1000.0) / (st.chord_m * 1000.0))

    def centroid_x_mm(self) -> float:
        """Flight x of the tube's mass: the middle of its length."""
        x, _ = self.centre_mm(0.5 * self.reach_y_mm)
        return float(x)

    def line(self, half_span_mm: float) -> str:
        mark = "OK " if self.ok else "BLOCKED"
        shape = ("one tube tip to tip" if self.one_piece else
                 f"swept {self.sweep_deg:.1f}, dihedral {self.dihedral_deg:.1f} deg"
                 f" -> V joiner {2*self.sweep_deg:.0f}/{2*self.dihedral_deg:.0f} deg")
        return (f"  [{mark}] {self.spec.name}: {self.spec.d_mm:.0f} mm from "
                f"{self.x_frac:.2f}c {self.anchor} at the root, {shape} | "
                f"reaches eta {self.reach_eta:.2f} "
                f"({self.reach_mm:.0f} mm of tube a side) | "
                f"root depth {self.depth_at_root_mm:.1f} mm"
                + (f" | CLASHES {self.clash_mm:.1f} mm"
                   if self.clash_mm > 0.0 else "")
                + ("" if not self.joints_blocked else
                   " | joints too shallow: "
                   + ", ".join(f"{e:.2f}" for e in self.joints_blocked)))


def depth_at(plan, eta: float, x_frac: float, wall_mm: float) -> float:
    """Usable internal depth at one station, minus both skins."""
    st = plan.at(float(np.clip(eta, 0.0, 1.0)))
    t = float(st.airfoil.thickness(np.array([x_frac]))[0]) * st.chord_m * 1000.0
    return t - 2.0 * wall_mm


def reach_of(plan, x_frac: float, spec: SparSpec, wall_mm: float,
             n: int = 160) -> float:
    """Outermost eta at which the tube still fits, scanning outward.

    Scanned rather than solved: thickness along the span is not monotone
    on a blended wing body (the body is fat, the blend pinches, the outer
    panel can thicken again relative to its chord), and a root-finder
    would happily return the far side of a pinch the tube cannot pass
    through. The FIRST station that blocks is the real limit."""
    return float(reach_many(plan, np.array([x_frac]), spec, wall_mm, n)[0])


def reach_many(plan, xs, spec: SparSpec, wall_mm: float,
               n: int = 160) -> np.ndarray:
    """`reach_of` for every chord station in `xs` at once.

    The same scan, span station by span station, with the whole row of
    candidate stations evaluated together: the fitter asks about 25 of
    them, and asking one at a time lofted every span station 25 times."""
    need = spec.needed_mm(wall_mm)
    xs = np.asarray(xs, dtype=float)
    out = np.ones(len(xs))
    open_ = np.ones(len(xs), dtype=bool)
    for e in np.linspace(0.0, 1.0, n):
        st = plan.at(float(e))
        d = (st.airfoil.thickness(xs) * st.chord_m * 1000.0 - 2.0 * wall_mm)
        stop = open_ & (d < need)
        out[stop] = max(e - 1.0 / (n - 1), 0.0)
        open_ &= ~stop
        if not open_.any():
            break
    return out


def fit_radius_mm(spec: SparSpec, wall_mm: float) -> float:
    """How far the tube's axis must stay from the contour line.

    The contour is the bead's centreline, so half a bead of wall, the
    tube's radius and half the diametral fit clearance. The bore gate
    asks the same question with `PrintSettings.spar_clearance_mm`, 0.5 mm
    against this 0.8, so a tube the fit seats the gate passes.

    It replaces `needed_mm` less two walls, which counted each wall twice
    (ROADMAP-BUILD.md section 1, "Minor, flagged"): once subtracted from
    the depth and once added to the requirement."""
    return 0.5 * spec.d_mm + 0.5 * wall_mm + 0.5 * spec.clearance_mm


class _Sections:
    """The placed upper and lower skins at a grid of span stations, flight
    mm, sorted in x -- built once per fit rather than once per line."""

    def __init__(self, plan, n_eta: int = 161, n_x: int = 81):
        self.etas = np.linspace(0.0, 1.0, n_eta)
        self.y = self.etas * plan.half_span_m * 1000.0
        u = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n_x)))
        self.up, self.lo, self.xr = [], [], []
        for e in self.etas:
            st = plan.at(float(e))
            c = st.chord_m * 1000.0
            a = np.radians(-st.twist_deg)
            out = []
            for v in (st.airfoil.y_upper(u), st.airfoil.y_lower(u)):
                xs = (st.x_le_m * 1000.0 + 0.25 * c
                      + c * ((u - 0.25) * np.cos(a) - v * np.sin(a)))
                zs = st.z_le_m * 1000.0 + c * ((u - 0.25) * np.sin(a) + v * np.cos(a))
                o = np.argsort(xs)
                out.append((xs[o], zs[o]))
            self.up.append(out[0])
            self.lo.append(out[1])
            self.xr.append((max(out[0][0][0], out[1][0][0]),
                            min(out[0][0][-1], out[1][0][-1])))

    def inside(self, j: int, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Are these points (flight x, z) inside the section at station j?"""
        (xu, zu), (xl, zl) = self.up[j], self.lo[j]
        lo_x, hi_x = self.xr[j]
        return ((x > lo_x) & (x < hi_x)
                & (z < np.interp(x, xu, zu)) & (z > np.interp(x, xl, zl)))


def _ellipse(sx, sz, r, n: int = 12):
    """Offsets (dx, dz) around the boundary of a tube's section by a
    y = const plane, for each line -> (lines, n, 2). Major semi-axis
    r sqrt(1 + sx^2 + sz^2) along (sx, sz); minor r across it."""
    sx, sz = np.asarray(sx, dtype=float), np.asarray(sz, dtype=float)
    m = np.hypot(sx, sz)
    e1 = np.where(m[:, None] > 1e-12,
                  np.stack([sx, sz], 1) / np.maximum(m, 1e-12)[:, None],
                  np.array([[1.0, 0.0]]))
    e2 = np.stack([-e1[:, 1], e1[:, 0]], 1)
    # CIRCUMSCRIBED, not inscribed: points on the ellipse leave the chords
    # between them up to r (1 - cos(pi/n)) inside it -- 0.2 mm at twelve
    # points on micro's tube, which an independent check at twenty-four
    # found poking through the skin. Pushed out by 1/cos(pi/n), the polygon
    # contains the circle, and an affine map carries that to the ellipse.
    grow = 1.0 / np.cos(np.pi / n)
    a = grow * r * np.sqrt(1.0 + sx * sx + sz * sz)
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return (np.cos(th)[None, :, None] * (a[:, None, None] * e1[:, None, :])
            + np.sin(th)[None, :, None] * (grow * r * e2[:, None, :]))


def _reach(sec: _Sections, x0, z0, sx, sz, r) -> np.ndarray:
    """Span fraction each line reaches: the last station of an unbroken
    run from the root at which the whole inflated ellipse is inside the
    section. -1 if it does not fit even at the root. Scanned outward,
    never solved, for the reason `reach_of` gives: thickness along a
    blended body is not monotone."""
    off = _ellipse(sx, sz, r)
    reach = np.full(len(x0), -1.0)
    alive = np.ones(len(x0), dtype=bool)
    for j, y in enumerate(sec.y):
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            break
        px = (x0[idx] + sx[idx] * y)[:, None] + off[idx, :, 0]
        pz = (z0[idx] + sz[idx] * y)[:, None] + off[idx, :, 1]
        ok = sec.inside(j, px.ravel(), pz.ravel()).reshape(px.shape).all(1)
        reach[idx[ok]] = sec.etas[j]
        alive[idx[~ok]] = False
    return reach


def place(plan, spec: SparSpec, wall_mm: float, joints=(),
          n_x: int = 13, reserved=(), min_reach: float = 0.0,
          targets=(0.35, 0.5, 0.65, 0.8, 1.0), sections=None) -> SparFit:
    """Choose the straight tube: its seat at the root and its direction.

    Candidates are lines from a root seat -- across the declared chord
    band, against the lower skin, the upper, or centred -- aimed at the
    SAME seat at a target station further out, plus the line parallel to
    the span, which is one tube tip to tip. A line through two seats is
    the chord of the curve the old fit let the tube follow: the straight
    tube that stays nearest its corridor.

    Reach is a CONSTRAINT, not the objective, and that ordering was wrong
    for one release. It used to pick the station that reached furthest and
    break ties on clearance -- which is fine while nothing else is in the
    wing, and wrong as soon as something is. Among lines that meet
    `min_reach` -- the mission's `min_spar_reach_frac` -- the tube goes
    where it is CLEAREST, then furthest, then nearest the middle of the
    band, then seated rather than floating, then with the smallest joiner.
    If nothing meets `min_reach`, reach leads again, because then the
    binding problem is reach and the gate should say so.

    A clash that survives is REPORTED, not fixed by moving the tube
    outside its declared band. If the only line that reaches far enough
    runs through the battery, that is a real conflict the optimizer has to
    be charged for -- papering over it is how the fleet ended up with a
    tube through every pack.
    """
    from .geom import interior as it

    sec = sections or _Sections(plan)
    H = plan.half_span_m * 1000.0
    r = fit_radius_mm(spec, wall_mm)
    seat_margin = 0.3
    rows = []                                # (x_frac, anchor, x0, z0, sx, sz)

    def seat(j, f, anchor):
        st = plan.at(float(sec.etas[j]))
        x = st.x_le_m * 1000.0 + f * st.chord_m * 1000.0
        (xu, zu), (xl, zl) = sec.up[j], sec.lo[j]
        lo = float(np.interp(x, xl, zl)) + r + seat_margin
        hi = float(np.interp(x, xu, zu)) - r - seat_margin
        return x, {it.LOWER: lo, it.UPPER: hi}.get(anchor, 0.5 * (lo + hi))

    # Targets need not repeat the root seat: a tube can leave the root on
    # the lower skin and arrive centred, or drift a little forward along a
    # wing whose sweep carries its band aft. Restricting the family to
    # same-seat lines rejected clear tubes that exist -- on the trainer
    # every such TE line went through the pack. The target's chord
    # fraction stays inside the declared band, like the root's.
    t_idx = sorted({int(np.argmin(np.abs(sec.etas - t))) for t in targets} - {0})
    fs = np.linspace(spec.x_lo, spec.x_hi, n_x)
    step = (spec.x_hi - spec.x_lo) / max(n_x - 1, 1)
    anchors = (it.LOWER, it.UPPER, it.MID)
    for f in fs:
        for anchor in anchors:
            x0, z0 = seat(0, float(f), anchor)
            rows.append((float(f), anchor, x0, z0, 0.0, 0.0))
            for j in t_idx:
                for ft in {float(np.clip(f + k * step, spec.x_lo, spec.x_hi))
                           for k in (-2, 0, 2)}:
                    for at in anchors:
                        xt, zt = seat(j, ft, at)
                        rows.append((float(f), anchor, x0, z0,
                                     (xt - x0) / sec.y[j], (zt - z0) / sec.y[j]))
    arr = np.array([row[2:] for row in rows])
    reach = np.maximum(_reach(sec, arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], r), 0.0)
    # A cut tube's end face is square to its AXIS, and a swept axis crosses
    # the span at an angle, so the far edge of that face lies further out
    # than the point the axis ends at -- by r sin(angle), 3.3 mm on
    # micro's 46 degree tube -- in section no station checked. The bore
    # gate found the difference as 7.91 mm for an 8 mm tube at micro_p1's
    # last tube layer. So the tube ends that far short of its last clear
    # station: all of it, end face included, is in checked section.
    lean = np.hypot(arr[:, 2], arr[:, 3])
    protrude = r * lean / np.sqrt(1.0 + lean * lean)
    reach = np.maximum(reach - protrude / H, 0.0)

    viable = reach >= min_reach - 1e-9
    if not viable.any():                     # reach is the binding problem
        viable = reach >= reach.max() - 1e-9
    mid_band = 0.5 * (spec.x_lo + spec.x_hi)
    order = sorted(
        np.flatnonzero(viable),
        key=lambda i: (-float(reach[i]), abs(rows[i][0] - mid_band),
                       1 if rows[i][1] == it.MID else 0,
                       abs(rows[i][4]) + abs(rows[i][5]), i))

    def fit_for(i, clash=0.0):
        f, anchor, x0, z0, sx, sz = rows[i]
        ry = float(reach[i]) * H
        return SparFit(spec, f, float(reach[i]),
                       ry * float(np.sqrt(1.0 + sx * sx + sz * sz)),
                       depth_at(plan, 0.0, f, wall_mm),
                       tuple(float(e) for e in joints if e <= reach[i] + 1e-9),
                       tuple(float(e) for e in joints if e > reach[i] + 1e-9),
                       anchor=anchor, clash_mm=float(clash),
                       root_xz_mm=(float(x0), float(z0)),
                       slope=(float(sx), float(sz)))

    # Clearest first -- but no clash beats zero, so the first line in the
    # secondary order that is clear wins without the rest being asked, and
    # the whole ordering is only paid for when nothing is clear.
    reserved = tuple(reserved)
    scored = []
    for rank, i in enumerate(order):
        fit = fit_for(i)
        vol = it.spar_volume(fit, wall_mm, plan)
        clash = max((it.overlap_mm(vol, v, plan, wall_mm) for v in reserved),
                    default=0.0)
        if clash <= 0.0:
            return fit
        scored.append((round(clash, 6), rank, i))
    clash, _, i = min(scored)
    return fit_for(i, clash)


def fit_all(plan, specs, wall_mm: float, panel_etas=(),
            reserved=(), min_reach: float = 0.0) -> list[SparFit]:
    """Fit every spar. `panel_etas` are the joint stations the spar must
    cross -- the interior ones only, since the outermost end is the tip
    and nothing joins there.

    `reserved` is what is already inside the wing. Spars are fitted in
    declaration order and each is reserved against the ones before it, so
    two 8 mm tubes cannot both claim the same corner: the LE spar takes
    its seat, and the TE spar sees it."""
    from .geom import interior as it

    joints = tuple(e for _, e in panel_etas[:-1]) if panel_etas else ()
    taken = list(reserved)
    fits = []
    sec = _Sections(plan) if specs else None
    for s in specs:
        f = place(plan, s, wall_mm, joints, reserved=tuple(taken),
                  min_reach=min_reach, sections=sec)
        fits.append(f)
        taken.append(it.spar_volume(f, wall_mm, plan))
    return fits


def exclusion_bands(fits, wall_mm: float = 0.45) -> tuple:
    """Chordwise bands the ribs must not enter, as (x_frac, half_width_mm).

    Ribs are chordwise webs running skin to skin, and they sweep with Z,
    so a spanwise tube at a fixed chord fraction WILL meet one unless the
    rib pattern is told to avoid it.

    The half-width is MILLIMETRES and that is the whole point. It is the
    tube's radius, plus half a bead because the rib's extrusion is
    centred on the contour, plus the fit clearance -- all physical
    lengths that do not shrink as the chord does. Expressed as a
    root-chord fraction and applied as a local-chord fraction, as it was,
    the corridor came out three times too narrow at the tip and the truss
    ran through the spar.
    """
    return tuple((f.x_frac,
                  0.5 * f.spec.d_mm + 0.5 * wall_mm + f.spec.clearance_mm)
                 for f in fits)


def report(fits, plan, min_tip_penetration_mm: float = 35.0) -> str:
    half_mm = plan.half_span_m * 1000.0
    lines = ["spars:"]
    for f in fits:
        lines.append(f.line(half_mm))
    worst = min((f.reach_mm for f in fits), default=0.0)
    if worst < min_tip_penetration_mm:
        lines.append(f"  NOTE reach {worst:.0f} mm is under the "
                     f"{min_tip_penetration_mm:.0f} mm the outer joint wants")
    lines.append("  the bore is the shell cavity: no hole through the wall, "
                 "but the")
    lines.append("  ROOT FACE needs opening (0 bottom layers on that face, "
                 "or drill it)")
    return "\n".join(lines)
