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

    @property
    def ok(self) -> bool:
        return not self.joints_blocked

    def line(self, half_span_mm: float) -> str:
        mark = "OK " if self.ok else "BLOCKED"
        return (f"  [{mark}] {self.spec.name}: {self.spec.d_mm:.0f} mm at "
                f"{self.x_frac:.2f}c | reaches eta {self.reach_eta:.2f} "
                f"({self.reach_mm:.0f} mm of {half_span_mm:.0f}) | "
                f"root depth {self.depth_at_root_mm:.1f} mm"
                + ("" if self.ok else
                   f" | joints too shallow: "
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
          n_x: int = 25) -> SparFit:
    """Choose the chordwise station that reaches furthest outboard.

    Ties are broken toward the middle of the allowed band, which keeps
    the tube away from the leading-edge curvature and the trailing-edge
    thickening -- both places where a printed bore is least round."""
    xs = np.linspace(spec.x_lo, spec.x_hi, n_x)
    reaches = np.array([reach_of(plan, float(x), spec, wall_mm) for x in xs])
    best = reaches.max()
    cand = xs[reaches >= best - 1e-9]
    x_best = float(cand[int(np.argmin(np.abs(cand - 0.5 * (spec.x_lo + spec.x_hi))))])

    need = spec.needed_mm(wall_mm)
    cleared, blocked = [], []
    for e in joints:
        (cleared if depth_at(plan, e, x_best, wall_mm) >= need
         else blocked).append(float(e))

    half_mm = plan.half_span_m * 1000.0
    return SparFit(spec, x_best, float(best), float(best) * half_mm,
                   depth_at(plan, 0.0, x_best, wall_mm),
                   tuple(cleared), tuple(blocked))


def fit_all(plan, specs, wall_mm: float, panel_etas=()) -> list[SparFit]:
    """Fit every spar. `panel_etas` are the joint stations the spar must
    cross -- the interior ones only, since the outermost end is the tip
    and nothing joins there."""
    joints = tuple(e for _, e in panel_etas[:-1]) if panel_etas else ()
    return [place(plan, s, wall_mm, joints) for s in specs]


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
