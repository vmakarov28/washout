"""Can the mechanism actually deliver the deflection the score assumed?

`search.design` computes `dcm_ddeg` and demon1's entire objective is top
speed, which is scored by holding `max_elevon_deflect_deg` of down
elevon:

    dCL = dCm / SM  ->  a lower trim CL  ->  a higher speed

Nothing has ever checked that the servo, its arm, the pushrod and the
horn can reach that angle. The number at the top of the fleet table is a
deflection the aircraft has never been shown able to make.

## Why a four-bar solve and not a ratio

The textbook shortcut is delta = (r_servo / r_horn) * theta, which is the
small-angle limit of a parallel linkage and is wrong in the two ways that
matter here. It is linear, so it cannot show that a linkage runs out of
travel; and it is symmetric, so it says up and down deflection are equal
when a real four-bar is DIFFERENTIAL -- and differential elevon travel is
a handling property a trainer cares about.

So the linkage is solved as what it is: two circles and a rigid rod.

    H   hinge axis, at the elevon's upper surface
    S   servo output shaft, inside the wing forward of the hinge
    r_h horn arm, hanging BELOW the hinge
    r_s servo arm, hanging below the shaft
    L   pushrod, rigid, its length fixed by the neutral position

The horn tip rides a circle of radius r_h about H; the servo arm tip
rides a circle of radius r_s about S. Given a servo angle the rod places
the horn tip on a circle of radius L about the arm tip, and the horn tip
is therefore at an intersection of two circles -- zero, one or two
solutions. Zero means the linkage has locked: the servo has travel left
and the surface cannot follow it, which is exactly the failure a ratio
cannot express.

Signs, once, so they are not re-derived: x is aft, z is up, and the
elevon deflecting TRAILING EDGE DOWN swings the horn tip FORWARD, because
the horn hangs below a hinge that the trailing edge is aft of. The servo
arm also hangs down, so rotating it moves its tip fore and aft, which is
the direction the rod needs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Linkage:
    """Everything the mechanism is, in millimetres in the section plane.

    All four lengths are DECLARED hardware, in the same category as the
    motor and the pack: a 9 g servo's outermost arm hole is about 11 mm
    from the shaft, and a moulded control horn puts its hole 10 to 15 mm
    off the surface. None of it is derived and none of it is guessed --
    they are dimensions of parts someone buys, and the gate exists so the
    geometry has to accommodate them rather than the other way round.
    """

    hinge_x_mm: float               # aft of the local leading edge
    hinge_z_mm: float               # above the local chord line
    servo_x_mm: float
    servo_z_mm: float
    servo_arm_mm: float = 11.0
    horn_arm_mm: float = 12.0
    travel_deg: float = 60.0
    """Usable one-sided servo rotation. 60 degrees of a nominal 90 is
    what a standard servo gives before the arm starts to bind and the
    transmitter's endpoints are the limit rather than the mechanism."""

    @property
    def H(self) -> np.ndarray:
        return np.array([self.hinge_x_mm, self.hinge_z_mm])

    @property
    def S(self) -> np.ndarray:
        return np.array([self.servo_x_mm, self.servo_z_mm])

    def horn_tip(self, delta_deg: float) -> np.ndarray:
        """Horn tip at an elevon deflection. Positive delta is TE DOWN."""
        d = np.radians(delta_deg)
        return self.H + self.horn_arm_mm * np.array([-np.sin(d), -np.cos(d)])

    def arm_tip(self, theta_deg: float) -> np.ndarray:
        th = np.radians(theta_deg)
        return self.S + self.servo_arm_mm * np.array([np.sin(th), -np.cos(th)])

    @property
    def rod_mm(self) -> float:
        """Pushrod length, FIXED by the neutral position rather than
        declared. A rod of any other length would trim the surface away
        from neutral, which a linkage is adjusted at the field to undo."""
        return float(np.linalg.norm(self.arm_tip(0.0) - self.horn_tip(0.0)))


def _circle_intersections(c0, r0, c1, r1):
    """Points common to two circles. Empty when they cannot reach."""
    d = np.linalg.norm(np.asarray(c1) - np.asarray(c0))
    if d < 1e-12 or d > r0 + r1 or d < abs(r0 - r1):
        return []
    a = (r0 * r0 - r1 * r1 + d * d) / (2.0 * d)
    h2 = r0 * r0 - a * a
    if h2 < 0.0:
        return []
    h = np.sqrt(h2)
    e = (np.asarray(c1) - np.asarray(c0)) / d
    n = np.array([-e[1], e[0]])
    mid = np.asarray(c0) + a * e
    return [mid + h * n, mid - h * n]


def deflection_at(link: Linkage, theta_deg: float,
                  delta_hint: float = 0.0) -> float | None:
    """Elevon deflection for a servo angle. None if the linkage locks.

    Continued from `delta_hint` rather than chosen by a rule: a four-bar
    has two branches and they meet at the locking position, so the only
    way to stay on the branch the mechanism is actually assembled in is
    to follow it from the previous solution.
    """
    L = link.rod_mm
    tip = link.arm_tip(theta_deg)
    pts = _circle_intersections(link.H, link.horn_arm_mm, tip, L)
    if not pts:
        return None
    best, best_err = None, np.inf
    for p in pts:
        v = p - link.H
        # invert horn_tip: v = r * (-sin d, -cos d)
        d = float(np.degrees(np.arctan2(-v[0], -v[1])))
        err = abs(d - delta_hint)
        if err < best_err:
            best, best_err = d, err
    return best


def sweep(link: Linkage, use_frac: float = 0.9,
          n: int = 41) -> tuple[float, float, bool]:
    """Deflection reachable each way. -> (down_deg, up_deg, locked)

    `use_frac` of the servo's travel, not all of it: the last tenth is
    where the arm binds and where a transmitter's endpoints live, so
    designing to it is designing to a number nobody flies.

    `locked` is True when the four-bar ran out of solutions inside the
    travel that was swept -- the servo still turning and the surface no
    longer following. Reported rather than clamped, because a locked
    linkage is a different fault from a weak one and wants a different
    fix.
    """
    lim = use_frac * link.travel_deg
    locked = False
    down = up = 0.0
    for sign in (+1.0, -1.0):
        delta = 0.0
        for th in np.linspace(0.0, sign * lim, n)[1:]:
            d = deflection_at(link, float(th), delta)
            if d is None:
                locked = True
                break
            delta = d
        down = max(down, delta)
        up = min(up, delta)
    return float(down), float(up), bool(locked)


def for_station(plan, eta: float, hinge_x_frac: float,
                servo_x_frac: float, servo_arm_mm: float,
                horn_below_mm: float, travel_deg: float,
                wall_mm: float = 0.45) -> Linkage:
    """Build the linkage from the geometry at one spanwise station.

    Only two lengths are declared here and the rest is measured off the
    aerofoil, because the arms are not free: both ends of the pushrod have
    to be OUTSIDE the shell, and how far outside depends on how thick the
    wing is at that station.

    The hinge sits on the UPPER surface at the hinge line -- where
    `printing.elevons` puts it and where a tape hinge goes. A control horn
    is screwed to the elevon's LOWER surface, so its hole is the section's
    own thickness below the axis, plus however far the horn protrudes:

        horn_arm = thickness(x_hinge) + horn_below_mm

    which is why `horn_below_mm` is the declared number rather than the
    arm itself. On the trainer's inboard elevon the section is 10.8 mm, so
    an 8 mm protrusion gives a 19 mm arm -- and the same 8 mm horn on the
    tip elevon, 5.6 mm thick, gives 13.6 mm. The mechanism's leverage
    changes along the span whether or not anyone models it.

    The servo shaft sits against the LOWER inner skin at its solved chord
    station, and its arm hangs down through a slot, so the arm tip is
    `servo_arm - wall` outside the aeroplane and can reach the rod.
    """
    st = plan.at(float(np.clip(eta, 0.0, 1.0)))
    c_mm = st.chord_m * 1000.0
    xh = float(np.clip(hinge_x_frac, 0.0, 1.0))
    xs = float(np.clip(servo_x_frac, 0.0, 1.0))
    z_upper = float(st.airfoil.y_upper(np.array([xh]))[0]) * c_mm
    thick = float(st.airfoil.thickness(np.array([xh]))[0]) * c_mm
    z_lower_s = float(st.airfoil.y_lower(np.array([xs]))[0]) * c_mm
    return Linkage(hinge_x_mm=xh * c_mm, hinge_z_mm=z_upper,
                   servo_x_mm=xs * c_mm,
                   servo_z_mm=z_lower_s + wall_mm,
                   servo_arm_mm=servo_arm_mm,
                   horn_arm_mm=thick + horn_below_mm,
                   travel_deg=travel_deg)


def report(link: Linkage, wanted_deg: float) -> str:
    down, up, locked = sweep(link)
    mark = "OK" if (min(down, -up) >= wanted_deg and not locked) else "SHORT"
    lines = [
        f"linkage: {mark}",
        f"  servo arm {link.servo_arm_mm:.0f} mm at "
        f"{link.servo_x_mm:.0f} mm | horn {link.horn_arm_mm:.0f} mm at "
        f"{link.hinge_x_mm:.0f} mm | pushrod {link.rod_mm:.1f} mm",
        f"  reaches {down:+.1f} / {up:+.1f} deg at "
        f"{0.9 * link.travel_deg:.0f} deg of servo travel "
        f"(wanted +/-{wanted_deg:.0f})",
        f"  differential {abs(down) - abs(up):+.1f} deg "
        f"(down minus up; a four-bar is never symmetric)",
    ]
    if locked:
        lines.append("  LOCKS inside the swept travel: the servo turns and "
                     "the surface stops following")
    return "\n".join(lines)
