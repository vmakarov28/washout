"""Motor, propeller, battery: can it actually reach that speed?

Nothing in this program modelled thrust, and for the efficiency missions
that never mattered -- L/D constrains itself. Give it a SPEED objective
and the gap opens immediately: cruise speed is

    v = sqrt(2W / (rho S CL))

so the optimizer maximises it by driving CL toward zero. The demon1
search duly returned a wing that trims at CL 0.024 and "cruises" at
48 m/s with an L/D of 0.45 -- perfectly self-consistent, and it would
need 766 gf of thrust and 361 W to hold, from a motor that makes about
800 gf standing still and far less at speed. The objective was unbounded
because nothing was pushing.

The model here is deliberately modest. Blade-element theory would be
more accurate and needs propeller geometry nobody has; what is wanted is
a defensible thrust-versus-airspeed curve that goes to zero at a
sensible place, respects the electrical limit, and can be calibrated
against numbers a hobbyist actually knows -- static thrust off a bench
test, and the pack voltage.

    n      = RPM / 60                        revolutions per second
    J      = v / (n D)                       advance ratio
    Ct(J)  = Ct0 (1 - J/J0)                  thrust falls off linearly
    T      = Ct rho n^2 D^4                  and vanishes at J0 ~ pitch/D

The linear Ct(J) is the honest simplification: real propellers curve,
but they all cross zero within a few percent of the geometric advance
ratio, and that crossing is what bounds top speed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RHO_AIR = 1.225
G = 9.80665
IN_TO_M = 0.0254
FIGURE_OF_MERIT = 0.55
"""Fraction of ideal momentum-theory thrust a real propeller achieves."""
MAX_TIP_SPEED_MS = 200.0


@dataclass(frozen=True)
class Motor:
    name: str
    kv: float                 # rpm per volt, unloaded
    mass_kg: float = 0.029
    max_power_w: float = 380.0
    """Burst electrical power the motor and ESC will take. A 2205 on 4S
    through a 30 A ESC is ~440 W on paper; 380 is what it will hold for
    more than a few seconds without cooking."""
    load_factor: float = 0.82
    """Loaded RPM as a fraction of no-load Kv*V. Propellers pull motors
    down; 0.80-0.85 is the usual range for a well-matched prop."""


@dataclass(frozen=True)
class Propeller:
    diameter_in: float
    pitch_in: float
    ct0: float = 0.105
    """Static thrust coefficient. 0.10-0.12 covers most sport props; this
    is the single number to calibrate if you bench-test yours."""

    @property
    def d_m(self) -> float:
        return self.diameter_in * IN_TO_M

    @property
    def j_zero(self) -> float:
        """Advance ratio at which thrust vanishes -- essentially the
        geometric pitch ratio, slightly reduced for slip."""
        return 0.95 * (self.pitch_in / self.diameter_in)


@dataclass(frozen=True)
class Battery:
    name: str
    cells: int
    capacity_ah: float
    mass_kg: float

    @property
    def volts(self) -> float:
        return 3.7 * self.cells        # nominal under load


@dataclass(frozen=True)
class Powertrain:
    motor: Motor
    prop: Propeller
    battery: Battery
    eta_esc: float = 0.90

    def rpm(self) -> float:
        return self.motor.kv * self.battery.volts * self.motor.load_factor

    def thrust_n(self, v_ms: float) -> float:
        """Thrust available at airspeed v. Clamped at zero -- a propeller
        past its zero-thrust advance ratio is a brake, and modelling that
        as negative thrust would let the optimizer exploit it."""
        n = self.rpm() / 60.0
        d = self.prop.d_m
        j = v_ms / max(n * d, 1e-9)
        ct = self.prop.ct0 * max(0.0, 1.0 - j / self.prop.j_zero)
        return float(ct * RHO_AIR * n**2 * d**4)

    def static_thrust_gf(self) -> float:
        return self.power_limited_thrust_n(0.0) / G * 1000.0

    def shaft_power_w(self, v_ms: float) -> float:
        """Useful power at this speed. The electrical draw is higher by
        the ESC and motor efficiencies, which is where the limit bites."""
        return self.thrust_n(v_ms) * max(v_ms, 0.0)

    def power_limited_thrust_n(self, v_ms: float) -> float:
        """Thrust after the electrical ceiling is applied.

        Momentum theory gives the best thrust any disc can make from a
        given shaft power: T = (2 rho A P^2)^(1/3). Above about 20 m/s
        that ceiling, not the Ct curve, is what actually stops a small
        motor -- which is precisely the regime demon1 was claiming."""
        t_prop = self.thrust_n(v_ms)
        p_shaft = self.motor.max_power_w * self.eta_esc * 0.80   # motor eta
        area = np.pi * 0.25 * self.prop.d_m**2
        # Ideal momentum-theory thrust for this shaft power, derated by a
        # figure of merit. Real propellers reach 50-60% of the ideal, and
        # the limit applies AT ZERO SPEED TOO -- exempting static thrust
        # (the first version did) let a 2205 claim 1596 gf on a 7 inch
        # prop, which is roughly twice what that motor can put into the
        # air on any prop at all.
        t_power = FIGURE_OF_MERIT * (2.0 * RHO_AIR * area
                                     * max(p_shaft, 1e-6) ** 2) ** (1.0 / 3.0)
        return float(min(t_prop, t_power))

    def tip_speed_ms(self) -> float:
        """Propeller tip speed. Past ~200 m/s a hobby prop is into
        compressibility and, more practically, past its mechanical
        rating -- which is the real reason a 2300 kv motor cannot swing
        a 7 inch prop, and why every aircraft here uses a 5."""
        return float(np.pi * self.prop.d_m * self.rpm() / 60.0)

    def top_speed_ms(self, drag_at: callable, v_max: float = 80.0) -> float:
        """Where available thrust and drag cross. Bisection on
        (thrust - drag), which is monotone decreasing over the range that
        matters."""
        def excess(v):
            return self.power_limited_thrust_n(v) - drag_at(v)
        lo, hi = 1.0, v_max
        if excess(lo) <= 0:
            return 0.0
        if excess(hi) > 0:
            return v_max
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if excess(mid) > 0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def endurance_min(self, v_ms: float, drag_n: float,
                      reserve: float = 0.20) -> float:
        """Level-flight endurance at this speed, 20% pack held back."""
        p_shaft = drag_n * v_ms
        p_elec = p_shaft / max(self.eta_esc * 0.80 * 0.75, 1e-6)  # prop eta
        wh = self.battery.volts * self.battery.capacity_ah * (1.0 - reserve)
        return float(60.0 * wh / max(p_elec, 1e-6))

    def report(self, v_ms: float = 0.0) -> str:
        return (f"powertrain: {self.motor.name} on {self.battery.name}, "
                f"{self.prop.diameter_in:.0f}x{self.prop.pitch_in:.1f} prop\n"
                f"  {self.rpm():.0f} rpm loaded | static thrust "
                f"{self.static_thrust_gf():.0f} gf | "
                f"J0 {self.prop.j_zero:.2f}")


# The fleet all fly this motor and this receiver; only the pack and the
# propeller change.
M2205 = Motor("2205 2300kv", kv=2300.0, mass_kg=0.029, max_power_w=380.0)

PACKS = {
    "2S 450": Battery("2S 450", 2, 0.45, 0.028),
    "3S 1300": Battery("3S 1300", 3, 1.30, 0.110),
    "4S 1300": Battery("4S 1300", 4, 1.30, 0.150),
    "4S 850": Battery("4S 850", 4, 0.85, 0.105),
}


def trainer_power() -> Powertrain:
    # 5 inch, low pitch: a 2300 kv motor is a 5 inch motor, and the
    # trainer wants thrust at 9 m/s rather than top end
    return Powertrain(M2205, Propeller(5.0, 3.0), PACKS["3S 1300"])


def demon_power() -> Powertrain:
    # high pitch: thrust has to survive to 30 m/s and beyond
    return Powertrain(M2205, Propeller(5.0, 5.0), PACKS["4S 850"])


def micro_power() -> Powertrain:
    return Powertrain(M2205, Propeller(5.0, 3.0), PACKS["2S 450"])
