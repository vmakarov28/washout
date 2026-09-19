"""Where the wires go: open channels cut into the upper skin.

## Why channels, and why they are all on one skin

In vase mode an opening is a recess: exterior space, sealed from the
wing's interior and from every other recess by one bead. There is no
"through". A servo lead cannot pass from the receiver's pocket to the
servo's pocket inside the wing, and the battery's lead cannot reach the
ESC through the wall between their bays. Every wire therefore runs on
the OUTSIDE of the skin, and the cheapest honest thing the geometry can
do about it is to give each one a channel: a shallow detour of the loop,
constant along its run, ramped at its ends like a bay, taped over.

A channel lives on one skin, so every pocket it joins has to be on that
skin too. That is why the whole fleet opens from the top: pack, receiver
and servos, with the horns on the elevons' upper surfaces and the belly
clean for the landing and the CG mark.

## The three leads

    battery -> ESC     a cross channel over the wall between the two
                       root bays, wide enough for the pack's connector
    ESC -> motor       a chordwise channel from the electronics bay to
                       the trailing edge, BESIDE the motor mount's
                       flange rather than under it, so the flange still
                       bonds to skin
    receiver -> servo  a spanwise channel at the servo pocket's station,
                       from the receiver bay's ramp to the pocket's

Each is a `vase.BayCut` with the same ramp machinery as a bay and the
same reservation against the spars. What does not fit is reported, not
squeezed: a wall too narrow for the pack's connector is a design fact
the optimizer has to pay for.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geom import interior as it
from .printing import vase

SERVO_LEAD_MM = (4.5, 2.2)
"""(width, depth) for a servo lead: three 26 AWG conductors side by
side, laid into the channel from above with the connector already at
the receiver end. Declared from the cable, not measured on a print."""
MOTOR_WIRES_MM = (8.0, 2.6)
"""(width across the span, depth) for three 18 AWG silicone motor
wires side by side."""
PACK_LEAD_MM = (8.0, 6.5)
"""(minimum wall width, depth) for the pack's lead with an XT30 on it:
the connector is 8 x 6 mm and has to cross the wall between the two
root bays."""
WALL_INSET_MM = 1.5
"""Skin left between a channel's wall and a bay's wall: two beads
1.5 mm apart centre to centre is a bead of air between them."""
CENTRE_HALF_SPAN_MM = 6.0
"""A root channel runs this far each side of the centreline, so the
pack's connector has twelve millimetres of channel to lie in."""
MOTOR_CHANNEL_OFFSET_MM = 19.0
"""The motor-wire channel's centre, mm from the centreline: just outside
the mount's 15 mm half-width, so the wires go round the flange."""


@dataclass(frozen=True)
class Channel:
    """One channel, ready to become a cut once its ramps are measured."""

    name: str
    x0_abs_mm: float                # forward edge, mm aft of the root LE
    x1_abs_mm: float                # aft edge
    eta0: float
    eta1: float
    depth_mm: float
    lead: str                       # what runs in it, for the build sheet
    length_mm_wire: float           # how much wire to cut, for the BOM

    @property
    def x_abs_mm(self) -> float:
        return 0.5 * (self.x0_abs_mm + self.x1_abs_mm)

    @property
    def length_mm(self) -> float:
        return self.x1_abs_mm - self.x0_abs_mm

    def volume(self, plan, wall_mm: float) -> it.Volume:
        root_c = plan.stations[0].chord_m * 1000.0
        return it.Volume(f"{self.name} channel",
                         (self.x_abs_mm - 0.5 * self.length_mm) / root_c,
                         (self.x_abs_mm + 0.5 * self.length_mm) / root_c,
                         self.eta0, self.eta1,
                         height_mm=self.depth_mm + 2.0 * wall_mm,
                         anchor=it.UPPER, offset_mm=0.0,
                         length_mm=self.length_mm, x_abs_mm=self.x_abs_mm)


def _root_bays(mission, bay_vols, cut_bands):
    """Root bays on the upper skin, forward to aft, with their CUT bands
    (absolute mm)."""
    out = []
    for bay in mission.bays:
        if bay.open_from != "upper":
            continue
        v = next(x for x in bay_vols if x.name == bay.name)
        if v.eta0 > 1e-9:
            continue
        x0, x1 = cut_bands[bay.name]
        out.append((bay, v, x0, x1))
    return sorted(out, key=lambda t: t[2])


def plan_channels(plan, mission, bay_vols, bay_etas, cut_bands,
                  x_te_at, wall_mm: float) -> tuple[list[Channel], list[str]]:
    """The channels, from the seats alone. -> (channels, reasons)

    `cut_bands` maps bay name -> (x0, x1) of its CUT in absolute mm;
    `x_te_at(eta)` gives the trailing edge's absolute station. Ramps are
    not decided here: the caller measures them on bare panels with the
    same walk a bay gets (`search.design.ramp_walk`), and gates each run
    against every opening it is not meant to reach."""
    half_mm = plan.half_span_m * 1000.0
    channels: list[Channel] = []
    reasons: list[str] = []
    roots = _root_bays(mission, bay_vols, cut_bands)

    # battery -> ESC: over the wall between consecutive root bays
    for (ba, va, a0, a1), (bb, vb, b0, b1) in zip(roots, roots[1:]):
        lo, hi = a1 + WALL_INSET_MM, b0 - WALL_INSET_MM
        if hi - lo < PACK_LEAD_MM[0]:
            reasons.append(f"no room for the pack lead between {ba.name} and "
                           f"{bb.name}: {b0 - a1:.1f} mm of wall, the connector "
                           f"needs {PACK_LEAD_MM[0] + 2 * WALL_INSET_MM:.0f}")
            continue
        channels.append(Channel(f"{ba.name} to {bb.name} lead", lo, hi,
                                0.0, CENTRE_HALF_SPAN_MM / half_mm,
                                PACK_LEAD_MM[1], "battery lead",
                                length_mm_wire=(hi - lo) + 60.0))

    # ESC -> motor: from the aft-most root bay to the trailing edge,
    # beside the mount's flange
    if roots:
        _, _, _, aft = roots[-1]
        e_c = MOTOR_CHANNEL_OFFSET_MM / half_mm
        e0 = (MOTOR_CHANNEL_OFFSET_MM - 0.5 * MOTOR_WIRES_MM[0]) / half_mm
        e1 = (MOTOR_CHANNEL_OFFSET_MM + 0.5 * MOTOR_WIRES_MM[0]) / half_mm
        x1 = float(x_te_at(e0)) - 2.5
        x0 = aft + WALL_INSET_MM
        if x1 - x0 < 10.0:
            reasons.append("no chord left between the electronics bay and the "
                           "trailing edge for the motor wires")
        else:
            channels.append(Channel("motor wires", x0, x1, e0, e1,
                                    MOTOR_WIRES_MM[1], "motor wires",
                                    length_mm_wire=(x1 - x0) + 80.0))

    # receiver -> servo: spanwise at the servo pocket's station
    rx = next((b for b in mission.bays if any("rx" in h for h in b.holds)), None)
    servo = next((b for b in mission.bays if b.name == "servos"), None)
    if rx is not None and servo is not None and bay_etas.get("servos"):
        v_rx = next(x for x in bay_vols if x.name == rx.name)
        v_s = next(x for x in bay_vols if x.name == servo.name)
        xc = v_s.x_abs_mm
        channels.append(Channel("servo lead",
                                xc - 0.5 * SERVO_LEAD_MM[0],
                                xc + 0.5 * SERVO_LEAD_MM[0],
                                float(v_rx.eta1), float(v_s.eta0),
                                SERVO_LEAD_MM[1], "servo lead",
                                length_mm_wire=(v_s.eta0 - v_rx.eta1) * half_mm + 100.0))
    return channels, reasons

def report(channels, reasons) -> str:
    lines = ["wiring:"]
    for c in channels:
        lines.append(f"  {c.name:<24} {c.length_mm:5.1f} x {c.depth_mm:.1f} mm channel, "
                     f"x {c.x0_abs_mm:.0f}-{c.x1_abs_mm:.0f} mm, eta "
                     f"{c.eta0:.3f}-{c.eta1:.3f} | cut {c.length_mm_wire:.0f} mm of {c.lead}")
    for r in reasons:
        lines.append(f"  NOTE {r}")
    if not channels and not reasons:
        lines.append("  none")
    return "\n".join(lines)
