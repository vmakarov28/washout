"""Blended-wing-body planform: a stack of spanwise stations, lofted.

Axes (aircraft frame, metres):
    x  aft   (+x downstream)
    y  right (+y starboard; the model is the RIGHT half, mirrored)
    z  up

A station fixes chord, leading-edge x (sweep), leading-edge z (dihedral),
twist and a CST section at one span fraction. Everything between two
stations is interpolated -- linearly in the planform numbers, and in CST
COEFFICIENT space for the section, which is the whole reason the airfoil
module uses that basis: a blended wing body is a continuous morph from a
thick, long-chord body section to a thin, reflexed outer-wing section,
and that morph has to stay a valid airfoil at every station in between,
including the ones the slicer asks for at 0.1 mm intervals.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.interpolate import PchipInterpolator

from .cst import Airfoil, blend


@dataclass(frozen=True)
class Station:
    eta: float          # span fraction, 0 at centreline .. 1 at tip
    chord_m: float
    x_le_m: float       # leading-edge x -- this is the sweep
    z_le_m: float       # leading-edge z -- this is the dihedral
    twist_deg: float    # positive = nose up (washout at the tip is negative)
    airfoil: Airfoil


@dataclass(frozen=True)
class Planform:
    """The right half. `half_span_m` is the y of the tip station."""

    half_span_m: float
    stations: tuple[Station, ...]
    name: str = "bwb"
    smooth: bool = True
    """Interpolate between stations with a monotone cubic (PCHIP) rather
    than straight lines.

    This is what turns a polygon into an aircraft. Linear interpolation
    puts a CREASE at every station: the leading edge kinks, the chord
    distribution has corners, and the lofted skin carries a visible facet
    down the span at each one. A blended wing body is defined by NOT
    doing that -- the body has to flow into the wing.

    PCHIP specifically, not a natural cubic spline, because it is
    shape-preserving: it will not overshoot between stations. A spline
    fitted through a fast chord taper happily returns a NEGATIVE chord
    just outboard of the body, and the optimizer would find that hole
    within a few hundred evaluations."""

    def __post_init__(self) -> None:
        etas = [s.eta for s in self.stations]
        if etas[0] != 0.0 or etas[-1] != 1.0:
            raise ValueError("stations must span eta = 0 .. 1")
        if any(b <= a for a, b in zip(etas, etas[1:])):
            raise ValueError("station etas must strictly increase")
        object.__setattr__(self, "_interp", None)

    def _lofter(self):
        """Build (once) the interpolators over every station quantity.

        Everything is interpolated in one array -- planform numbers AND
        the CST coefficients -- so the section morphs with exactly the
        same smoothness as the chord does. Built lazily and cached
        because `at()` is called tens of thousands of times per search
        and rebuilding a PCHIP each time dominated the profile."""
        cache = object.__getattribute__(self, "_interp")
        if cache is not None:
            return cache
        st = self.stations
        etas = np.array([s.eta for s in st])
        n_c = len(st[0].airfoil.au)
        rows = []
        for s_ in st:
            rows.append(np.concatenate([
                [s_.chord_m, s_.x_le_m, s_.z_le_m, s_.twist_deg],
                s_.airfoil.au, s_.airfoil.al,
                [s_.airfoil.te_gap, s_.airfoil.te_camber]]))
        data = np.array(rows)
        if self.smooth and len(st) >= 3:
            f = PchipInterpolator(etas, data, axis=0, extrapolate=True)
        else:
            def f(e, _e=etas, _d=data):
                return np.array([np.interp(e, _e, _d[:, k])
                                 for k in range(_d.shape[1])])
        cache = (f, n_c)
        object.__setattr__(self, "_interp", cache)
        return cache

    # ---------------------------------------------------------------- loft

    def at(self, eta: float) -> Station:
        """The interpolated station at any span fraction."""
        eta = float(np.clip(eta, 0.0, 1.0))
        f, n_c = self._lofter()
        v = np.asarray(f(eta)).ravel()
        au = v[4:4 + n_c]
        al = v[4 + n_c:4 + 2 * n_c]
        return Station(
            eta=eta,
            chord_m=float(max(v[0], 1e-4)),
            x_le_m=float(v[1]),
            z_le_m=float(v[2]),
            twist_deg=float(v[3]),
            airfoil=Airfoil(au=au, al=al, te_gap=float(v[-2]),
                            te_camber=float(v[-1]), name="lofted"),
        )

    def section_3d(self, eta: float, n: int = 121) -> np.ndarray:
        """The section loop placed in aircraft coordinates -> (N, 3).

        Twist rotates about the QUARTER CHORD, the convention every
        stability derivative below assumes; rotating about the leading
        edge instead would silently move the aerodynamic centre."""
        s = self.at(eta)
        loop = s.airfoil.coords(n)                      # unit chord, (N,2)
        p = loop - np.array([0.25, 0.0])
        a = np.radians(-s.twist_deg)                    # +twist = nose up
        ca, sa = np.cos(a), np.sin(a)
        rot = np.stack([p[:, 0] * ca - p[:, 1] * sa,
                        p[:, 0] * sa + p[:, 1] * ca], 1)
        xy = rot * s.chord_m + np.array([s.x_le_m + 0.25 * s.chord_m,
                                         s.z_le_m])
        y = np.full((len(loop), 1), s.eta * self.half_span_m)
        return np.concatenate([xy[:, :1], y, xy[:, 1:]], 1)   # (N,3) x,y,z

    # ------------------------------------------------------- planform maths

    @property
    def span_m(self) -> float:
        return 2.0 * self.half_span_m

    def _grid(self, n: int = 400) -> tuple[np.ndarray, np.ndarray]:
        eta = np.linspace(0.0, 1.0, n)
        chord = np.array([self.at(e).chord_m for e in eta])
        return eta, chord

    @property
    def area_m2(self) -> float:
        """Reference area, BOTH halves -- the projected planform."""
        eta, chord = self._grid()
        return float(2.0 * np.trapezoid(chord, eta * self.half_span_m))

    @property
    def aspect_ratio(self) -> float:
        return self.span_m**2 / self.area_m2

    @property
    def mac_m(self) -> float:
        """Mean aerodynamic chord: integral(c^2) / integral(c)."""
        eta, chord = self._grid()
        y = eta * self.half_span_m
        return float(np.trapezoid(chord**2, y) / np.trapezoid(chord, y))

    @property
    def x_mac_le_m(self) -> float:
        """Leading-edge x of the MAC -- the datum static margin is
        measured from."""
        eta, chord = self._grid()
        y = eta * self.half_span_m
        xle = np.array([self.at(e).x_le_m for e in eta])
        return float(np.trapezoid(chord * xle, y) / np.trapezoid(chord, y))

    @property
    def taper_ratio(self) -> float:
        return self.stations[-1].chord_m / self.stations[0].chord_m

    def sweep_quarter_chord_deg(self) -> float:
        root, tip = self.stations[0], self.stations[-1]
        dx = (tip.x_le_m + 0.25 * tip.chord_m) - (root.x_le_m + 0.25 * root.chord_m)
        return float(np.degrees(np.arctan2(dx, self.half_span_m)))

    def wetted_area_m2(self) -> float:
        """Both halves, both surfaces -- what the vase-mode shell mass and
        the skin-friction estimate are both proportional to."""
        eta, _ = self._grid(200)
        per = np.array([self.at(e).chord_m * self.at(e).airfoil.perimeter
                        for e in eta])
        return float(2.0 * np.trapezoid(per, eta * self.half_span_m))

    def volume_m3(self) -> float:
        """Enclosed volume of the right half x2 -- the payload bay budget
        (battery, electronics) lives inside this number."""
        eta, _ = self._grid(200)
        a = np.array([self.at(e).airfoil.area * self.at(e).chord_m**2
                      for e in eta])
        return float(2.0 * np.trapezoid(a, eta * self.half_span_m))

    def local_reynolds(self, v_ms: float, eta: float, nu: float = 1.5e-5) -> float:
        return self.at(eta).chord_m * v_ms / nu

    def is_valid(self) -> tuple[bool, str]:
        """Cheap geometric sanity, run before anything expensive."""
        for e in np.linspace(0.0, 1.0, 41):
            s = self.at(e)
            if s.chord_m <= 1e-3:
                return False, f"chord collapses at eta={e:.2f}"
            if not s.airfoil.is_valid():
                return False, f"self-intersecting section at eta={e:.2f}"
        if self.aspect_ratio < 1.0 or self.aspect_ratio > 30.0:
            return False, f"aspect ratio {self.aspect_ratio:.1f} out of range"
        return True, "ok"

    def report(self) -> str:
        v, why = self.is_valid()
        return "\n".join([
            f"{self.name}: span {self.span_m*1000:.0f} mm, "
            f"S {self.area_m2*1e4:.0f} cm^2, AR {self.aspect_ratio:.2f}",
            f"  MAC {self.mac_m*1000:.1f} mm at x_le {self.x_mac_le_m*1000:.1f} mm"
            f" | taper {self.taper_ratio:.2f}"
            f" | sweep_c/4 {self.sweep_quarter_chord_deg():.1f} deg",
            f"  wetted {self.wetted_area_m2()*1e4:.0f} cm^2,"
            f" volume {self.volume_m3()*1e6:.0f} cm^3 | valid: {why}",
        ])


def bwb(
    half_span_m: float,
    root_chord_m: float,
    body_eta: float,
    body_chord_frac: float,
    kink_eta: float,
    kink_chord_frac: float,
    tip_chord_frac: float,
    sweep_body_deg: float,
    sweep_mid_deg: float,
    sweep_outer_deg: float,
    dihedral_deg: float,
    twist_body_deg: float,
    twist_kink_deg: float,
    twist_tip_deg: float,
    root_airfoil: Airfoil,
    tip_airfoil: Airfoil,
    body_thickness_scale: float = 1.6,
    blend_thickness_scale: float | None = None,
    name: str = "bwb",
) -> Planform:
    """Four-station blended wing body: centre body, body edge, kink, tip.

    Four rather than three, and it matters. Three stations can express a
    fat middle and a thin tip but not the SHAPE of the transition between
    them, so the body met the wing at a corner and the optimizer had to
    choose between a thick body and a clean blend. A separate body-edge
    station lets the centre section hold its depth out to `body_eta` and
    then taper, which is what a blended wing body actually looks like --
    and with the PCHIP loft the leading edge comes out as a continuous
    curve rather than three straight segments.

    Sweep is given per segment as an angle, which keeps the parameters
    interpretable; the loft turns them into a smooth curve anyway.
    """
    if not (0.0 < body_eta < kink_eta < 1.0):
        raise ValueError(f"need 0 < body_eta {body_eta} < kink_eta {kink_eta} < 1")

    y_body = body_eta * half_span_m
    y_kink = kink_eta * half_span_m
    x_body = y_body * np.tan(np.radians(sweep_body_deg))
    x_kink = x_body + (y_kink - y_body) * np.tan(np.radians(sweep_mid_deg))
    x_tip = x_kink + (half_span_m - y_kink) * np.tan(np.radians(sweep_outer_deg))
    tan_dih = np.tan(np.radians(dihedral_deg))

    blend_scale = (blend_thickness_scale if blend_thickness_scale is not None
                   else 1.0 + 0.75 * (body_thickness_scale - 1.0))
    body = root_airfoil.scaled_thickness(body_thickness_scale)
    edge = root_airfoil.scaled_thickness(blend_scale)
    mid = blend(root_airfoil, tip_airfoil, 0.6).scaled_thickness(
        1.0 + 0.30 * (body_thickness_scale - 1.0))

    return Planform(
        half_span_m=half_span_m,
        stations=(
            Station(0.0, root_chord_m, 0.0, 0.0, twist_body_deg, body),
            Station(body_eta, root_chord_m * body_chord_frac, x_body,
                    y_body * tan_dih, twist_body_deg, edge),
            Station(kink_eta, root_chord_m * kink_chord_frac, x_kink,
                    y_kink * tan_dih, twist_kink_deg, mid),
            Station(1.0, root_chord_m * tip_chord_frac, x_tip,
                    half_span_m * tan_dih, twist_tip_deg, tip_airfoil),
        ),
        name=name,
    )
