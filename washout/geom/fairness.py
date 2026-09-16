"""Is it a fair surface? Continuity measured on the loft, not the inputs.

Every earlier gate asks whether the aircraft flies or prints. None asks
whether it is one SHAPE. The generation-2 fleet passed every gate and
looked assembled from parts: a leading edge that bent back and forth as
adjacent segments disagreed about sweep by 24 degrees, trailing edges
that swept aft and then hooked forward again at the tip, twist that went
+0.4, -4.7, -0.9 across three stations, and a centre body 36% thick
whose surface fell away at 12 degrees into a wing 16% thick. PCHIP made
each of those C1 -- no creases -- and C1 is not the same thing as fair.

A designer's word for the property is FAIRNESS: curvature that changes
slowly and does not reverse without a reason. This module measures it
on the interpolated surface itself, so it judges what will be printed
rather than the station numbers that produced it, and it counts things a
person would point at:

  * leading-edge inflections  -- the sweep distribution should rise from
    the rounded nose to a maximum and then relax outboard. One reversal
    of curvature is that shape; two is a wave.
  * trailing-edge hooks       -- a trailing edge may sweep forward over
    the body and aft over the wing, but once it is going aft it must not
    turn forward again. That is the notch and the tip hook.
  * sweep jumps               -- how far the leading-edge sweep changes
    across any 10% of the span, outboard of the nose.
  * twist and dihedral reversals, reverse taper -- monotone by intent.
  * thickness slope           -- how steeply the thickness envelope falls
    spanwise. This is the difference between a blended body and a pod.
  * root t/c, tip rise        -- the two absolute proportions that made
    the potato roots and the 200 mm winglets.

Reversals are counted with an AMPLITUDE in degrees -- a sweep curve has
to come back by 2 degrees before it counts as having turned -- so a
straight wing does not count numerical ripple as waves, and the limits
mean the same thing on a 350 mm micro and a 900 mm trainer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Limits:
    max_le_inflections: int = 1
    max_te_inflections: int = 2
    allow_te_hook: bool = False
    max_sweep_jump_deg: float = 20.0
    max_twist_reversals: int = 0
    max_dihedral_reversals: int = 1
    allow_reverse_taper: bool = False
    max_thickness_slope_deg: float = 7.0
    max_root_t_over_c: float = 0.26
    max_tip_rise_frac: float = 0.25


@dataclass(frozen=True)
class Fairness:
    le_inflections: int
    te_inflections: int
    te_hook: bool
    sweep_jump_deg: float
    twist_reversals: int
    dihedral_reversals: int
    reverse_taper: bool
    thickness_slope_deg: float
    root_t_over_c: float
    tip_rise_frac: float

    def violations(self, lim: Limits) -> list[tuple[str, float]]:
        """(reason, severity) for every limit exceeded. Severity is the
        fractional overshoot, so the optimizer gets a slope back towards
        fair geometry instead of a cliff."""
        out = []
        if self.le_inflections > lim.max_le_inflections:
            out.append((f"leading edge waves ({self.le_inflections} "
                        f"inflections, max {lim.max_le_inflections})",
                        float(self.le_inflections - lim.max_le_inflections)))
        if self.te_inflections > lim.max_te_inflections:
            out.append((f"trailing edge waves ({self.te_inflections} "
                        f"inflections, max {lim.max_te_inflections})",
                        float(self.te_inflections - lim.max_te_inflections)))
        if self.te_hook and not lim.allow_te_hook:
            out.append(("trailing edge hooks forward after sweeping aft", 1.0))
        if self.sweep_jump_deg > lim.max_sweep_jump_deg:
            out.append((f"sweep jumps {self.sweep_jump_deg:.0f} deg in 10% "
                        f"span (max {lim.max_sweep_jump_deg:.0f})",
                        self.sweep_jump_deg / lim.max_sweep_jump_deg - 1.0))
        if self.twist_reversals > lim.max_twist_reversals:
            out.append((f"twist reverses {self.twist_reversals}x",
                        float(self.twist_reversals - lim.max_twist_reversals)))
        if self.dihedral_reversals > lim.max_dihedral_reversals:
            out.append((f"dihedral reverses {self.dihedral_reversals}x",
                        float(self.dihedral_reversals - lim.max_dihedral_reversals)))
        if self.reverse_taper and not lim.allow_reverse_taper:
            out.append(("chord grows outboard (reverse taper)", 1.0))
        if self.thickness_slope_deg > lim.max_thickness_slope_deg:
            out.append((f"body falls away at {self.thickness_slope_deg:.1f} deg "
                        f"spanwise (max {lim.max_thickness_slope_deg:.0f}) -- a pod",
                        self.thickness_slope_deg / lim.max_thickness_slope_deg - 1.0))
        if self.root_t_over_c > lim.max_root_t_over_c:
            out.append((f"root t/c {self.root_t_over_c:.3f} over "
                        f"{lim.max_root_t_over_c:.2f}",
                        self.root_t_over_c / lim.max_root_t_over_c - 1.0))
        if self.tip_rise_frac > lim.max_tip_rise_frac:
            out.append((f"tip rises {100*self.tip_rise_frac:.0f}% of semi-span "
                        f"(max {100*lim.max_tip_rise_frac:.0f}%)",
                        self.tip_rise_frac / lim.max_tip_rise_frac - 1.0))
        return out

    def report(self, lim: Limits | None = None) -> str:
        bad = self.violations(lim) if lim else []
        head = "fairness: " + ("ok" if lim and not bad else
                               "; ".join(r for r, _ in bad) if bad else "")
        return "\n".join([
            head,
            f"          LE/TE inflections {self.le_inflections}/"
            f"{self.te_inflections} | TE hook {'yes' if self.te_hook else 'no'}"
            f" | max sweep jump {self.sweep_jump_deg:.1f} deg/10% span",
            f"          twist/dihedral reversals {self.twist_reversals}/"
            f"{self.dihedral_reversals} | thickness slope "
            f"{self.thickness_slope_deg:.1f} deg | root t/c "
            f"{self.root_t_over_c:.3f} | tip rise {100*self.tip_rise_frac:.0f}%",
        ])


def _extrema(v: np.ndarray, amp: float) -> tuple[int, int]:
    """(maxima, minima) of v, each with at least `amp` of prominence.

    Counted on the curve itself, not on its derivative. The first version
    counted sign changes of d(sweep)/d(eta) inside a deadband, and on the
    faired loft that measured noise: the Planform re-interpolates its
    dense stations with PCHIP, which is C1 but not C2, so the second
    derivative of every edge is a sawtooth. A leading edge whose sweep
    changed by 1.5 degrees across any tenth of the span scored THREE
    inflections. Here a turn registers only once the curve has come back
    by `amp` from its running extreme, so ripple smaller than that never
    counts and a real wave always does."""
    lo = hi = float(v[0])
    direction = 0
    n_max = n_min = 0
    for x in map(float, v[1:]):
        if direction == 0:
            if x >= lo + amp:
                direction, hi = 1, x
            elif x <= hi - amp:
                direction, lo = -1, x
            else:
                lo, hi = min(lo, x), max(hi, x)
        elif direction == 1:
            if x > hi:
                hi = x
            elif x <= hi - amp:
                n_max, direction, lo = n_max + 1, -1, x
        else:
            if x < lo:
                lo = x
            elif x >= lo + amp:
                n_min, direction, hi = n_min + 1, 1, x
    return n_max, n_min


def _turns(v: np.ndarray, amp: float) -> int:
    return sum(_extrema(v, amp))


def _sample(plan, n: int) -> np.ndarray:
    """(n, 4) chord, x_le, z_le, twist -- through the loft itself."""
    eta = np.linspace(0.0, 1.0, n)
    return np.array([[s.chord_m, s.x_le_m, s.z_le_m, s.twist_deg]
                     for s in (plan.at(float(e)) for e in eta)]), eta


def measure(plan, n: int = 241, n_thickness: int = 61) -> Fairness:
    v, eta = _sample(plan, n)
    chord, xle, z, twist = v.T
    half = plan.half_span_m
    y = eta * half
    xte = xle + chord

    lam_le = np.degrees(np.arctan(np.gradient(xle, y)))
    lam_te = np.degrees(np.arctan(np.gradient(xte, y)))
    gam = np.degrees(np.arctan(np.gradient(z, y)))

    le_infl = _turns(lam_le, amp=2.0)
    te_infl = _turns(lam_te, amp=3.0)

    # A HOOK is an aft extreme of the trailing edge ITSELF: the edge goes
    # back and then comes forward again by a visible distance. The first
    # rule looked at the sign of the TE sweep instead, and flagged a 3 mm
    # tail behind a 240 mm root chord as a hook -- ripple, not a shape --
    # which would have rejected a large share of otherwise fair designs.
    hook = _extrema(xte, amp=0.03 * chord[0])[0] > 0

    # sweep change across 10% of span, outboard of the nose: the nose is
    # where the mirrored halves are meant to round off from zero sweep
    k = max(int(round(0.1 * (n - 1))), 1)
    start = int(np.searchsorted(eta, plan.controls[1]))
    seg = lam_le[start:]
    jump = float(np.max(np.abs(seg[k:] - seg[:-k]))) if len(seg) > k else 0.0

    twist_rev = _turns(twist, amp=0.3)
    dih_rev = _turns(gam, amp=2.0)
    rev_taper = bool(np.any(np.gradient(chord, eta) > 0.02 * chord[0]))

    et = np.linspace(0.0, 1.0, n_thickness)
    h = np.array([0.5 * s_.airfoil.t_max * s_.chord_m
                  for s_ in (plan.at(float(e)) for e in et)])
    t_slope = float(np.degrees(np.arctan(np.max(np.abs(np.gradient(h, et * half))))))

    return Fairness(
        le_inflections=le_infl, te_inflections=te_infl, te_hook=hook,
        sweep_jump_deg=jump, twist_reversals=twist_rev,
        dihedral_reversals=dih_rev, reverse_taper=rev_taper,
        thickness_slope_deg=t_slope,
        root_t_over_c=float(plan.stations[0].airfoil.t_max),
        tip_rise_frac=float((z[-1] - z[0]) / half),
    )
