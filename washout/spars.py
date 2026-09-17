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

    @property
    def ok(self) -> bool:
        return not self.joints_blocked and self.clash_mm <= 0.0

    def line(self, half_span_mm: float) -> str:
        mark = "OK " if self.ok else "BLOCKED"
        return (f"  [{mark}] {self.spec.name}: {self.spec.d_mm:.0f} mm at "
                f"{self.x_frac:.2f}c {self.anchor} | reaches eta "
                f"{self.reach_eta:.2f} "
                f"({self.reach_mm:.0f} mm of {half_span_mm:.0f}) | "
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
    need = spec.needed_mm(wall_mm)
    etas = np.linspace(0.0, 1.0, n)
    for e in etas:
        if depth_at(plan, float(e), x_frac, wall_mm) < need:
            return float(max(e - 1.0 / (n - 1), 0.0))
    return 1.0


def place(plan, spec: SparSpec, wall_mm: float, joints=(),
          n_x: int = 25, reserved=()) -> SparFit:
    """Choose the chordwise station and the vertical seat.

    REACH stays the primary criterion -- it is what decides whether the
    outer joint is carried -- and among the stations that reach equally
    far, the tie now breaks on CLEARANCE from `reserved`: the payload
    volumes the mission has already declared. Failing that it breaks
    toward the middle of the allowed band, which keeps the tube away from
    the leading-edge curvature and the trailing-edge thickening, both
    places where a printed bore is least round.

    The seat is solved the same way: "lower", "upper" and "mid" are tried
    and the one with the most clearance wins. A tube seated hard against
    one skin leaves a single contiguous cavity for the payload instead of
    two useless slots, which is the whole reason to seat it at all.

    A clash that survives is REPORTED, not fixed by moving the tube
    outside its declared band. If the only station that reaches far enough
    is inside the battery, that is a real conflict the optimizer has to be
    charged for -- papering over it is how the fleet ended up with a tube
    through every pack.
    """
    from .geom import interior as it

    xs = np.linspace(spec.x_lo, spec.x_hi, n_x)
    reaches = np.array([reach_of(plan, float(x), spec, wall_mm) for x in xs])
    best = reaches.max()
    cand = xs[reaches >= best - 1e-9]
    half_mm = plan.half_span_m * 1000.0
    reserved = tuple(reserved)
    mid_band = 0.5 * (spec.x_lo + spec.x_hi)

    def trial(x: float, anchor: str) -> SparFit:
        return SparFit(spec, float(x), float(best), float(best) * half_mm,
                       depth_at(plan, 0.0, float(x), wall_mm), (), (),
                       anchor=anchor)

    scored = []
    for x in cand:
        for anchor in (it.LOWER, it.UPPER, it.MID):
            vol = it.spar_volume(trial(x, anchor), wall_mm, plan)
            clash = max((it.overlap_mm(vol, r, plan, wall_mm)
                         for r in reserved), default=0.0)
            # clearest first, then nearest the middle of the band, then a
            # seated tube ahead of a floating one
            scored.append((clash, abs(float(x) - mid_band),
                           1 if anchor == it.MID else 0, float(x), anchor))
    scored.sort()
    clash, _, _, x_best, anchor = scored[0]

    need = spec.needed_mm(wall_mm)
    cleared, blocked = [], []
    for e in joints:
        (cleared if depth_at(plan, e, x_best, wall_mm) >= need
         else blocked).append(float(e))

    return SparFit(spec, x_best, float(best), float(best) * half_mm,
                   depth_at(plan, 0.0, x_best, wall_mm),
                   tuple(cleared), tuple(blocked), anchor=anchor,
                   clash_mm=float(clash))


def fit_all(plan, specs, wall_mm: float, panel_etas=(),
            reserved=()) -> list[SparFit]:
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
    for s in specs:
        f = place(plan, s, wall_mm, joints, reserved=tuple(taken))
        fits.append(f)
        taken.append(it.spar_volume(f, wall_mm, plan))
    return fits


def exclusion_bands(fits, spec_pad: float = 0.5) -> tuple[tuple[float, float], ...]:
    """Chordwise bands the ribs must not enter.

    Ribs are chordwise webs running skin to skin, and they sweep with Z,
    so a spanwise tube at a fixed chord fraction WILL meet one unless the
    rib pattern is told to avoid it. Half-width is the tube radius plus a
    pad, expressed in chord fractions at the root -- the root is where the
    chord is longest, so the band is widest in absolute terms exactly
    where the spar is largest."""
    out = []
    for f in fits:
        half = spec_pad * f.spec.d_mm / 1000.0     # metres, scaled below
        out.append((f.x_frac, half))
    return tuple(out)


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
