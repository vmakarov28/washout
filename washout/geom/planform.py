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
    control_etas: tuple = ()
    """The stations a DESIGNER placed, as opposed to the dense ones the
    faired loft emits to carry its curves. Printed panel breaks, the
    station table and the fairness nose all key off these. Empty means
    every station is a control station, which is true of anything built
    by lofted() or bwb()."""

    @property
    def controls(self) -> tuple:
        return self.control_etas or tuple(s.eta for s in self.stations)

    def __post_init__(self) -> None:
        etas = [s.eta for s in self.stations]
        if etas[0] != 0.0 or etas[-1] != 1.0:
            raise ValueError("stations must span eta = 0 .. 1")
        if any(b <= a for a, b in zip(etas, etas[1:])):
            raise ValueError("station etas must strictly increase")
        object.__setattr__(self, "_interp", None)
        object.__setattr__(self, "_at_cache", {})

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
            # Fit across BOTH halves, mirrored about the centreline. PCHIP
            # estimates an endpoint slope one-sidedly, so fitted on the
            # right half alone it gave the leading edge a finite slope at
            # eta = 0 -- and the mirrored left half met it at an angle.
            # That is the spike on demon1's nose and the notch in micro's
            # trailing edge: a crease exactly on the symmetry plane, the
            # one place a blended body must be smoothest. Mirrored, every
            # quantity has equal and opposite secants either side of the
            # root, PCHIP sets the derivative there to zero, and the two
            # halves join tangent -- a rounded nose, no notch, and
            # nothing outboard of the first station changes at all.
            e_full = np.concatenate([-etas[:0:-1], etas])
            d_full = np.concatenate([data[:0:-1], data])
            f = PchipInterpolator(e_full, d_full, axis=0, extrapolate=True)
        else:
            def f(e, _e=etas, _d=data):
                return np.array([np.interp(e, _e, _d[:, k])
                                 for k in range(_d.shape[1])])
        cache = (f, n_c)
        object.__setattr__(self, "_interp", cache)
        return cache

    # ---------------------------------------------------------------- loft

    def at(self, eta: float) -> Station:
        """The interpolated station at any span fraction.

        Memoised per planform, because one evaluation asks for about
        160 000 stations and only about 6 000 of them are different: the
        interior checks walk the same span grid once per chordwise sample
        and once per candidate seat. The cache returns the SAME Station, so
        its coefficient arrays are made read-only -- a caller that edited
        one in place would otherwise be editing every later answer."""
        # min/max rather than np.clip: the same value for every finite
        # float and for NaN, at a tenth of the cost on a scalar
        eta = min(max(float(eta), 0.0), 1.0)
        cache = object.__getattribute__(self, "_at_cache")
        hit = cache.get(eta)
        if hit is not None:
            return hit
        f, n_c = self._lofter()
        v = np.asarray(f(eta)).ravel()
        au = v[4:4 + n_c]
        al = v[4 + n_c:4 + 2 * n_c]
        au.flags.writeable = False
        al.flags.writeable = False
        st = Station(
            eta=eta,
            chord_m=float(max(v[0], 1e-4)),
            x_le_m=float(v[1]),
            z_le_m=float(v[2]),
            twist_deg=float(v[3]),
            airfoil=Airfoil(au=au, al=al, te_gap=float(v[-2]),
                            te_camber=float(v[-1]), name="lofted"),
        )
        cache[eta] = st
        return st

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
        ] + self.station_table())

    def station_table(self) -> list[str]:
        """One line per station. Worth printing because the two things
        most recently added -- polyhedral and a real spanwise section
        family -- are invisible in any summary number. A wing with a
        winglet and one without have the same span and nearly the same
        area; they differ in z_le at the tip and in where the camber
        went."""
        out = ["  station    eta   chord   x_le   z_le    t/c   camber  twist"]
        prev_y = prev_z = 0.0
        for i, s_ in enumerate(self.at(e) for e in self.controls):
            y = s_.eta * self.half_span_m
            dy = y - prev_y
            dih = np.degrees(np.arctan2(s_.z_le_m - prev_z, dy)) if dy > 1e-9 else 0.0
            prev_y, prev_z = y, s_.z_le_m
            out.append(
                f"  [{i}]       {s_.eta:.3f} {s_.chord_m*1000:7.1f} "
                f"{s_.x_le_m*1000:6.1f} {s_.z_le_m*1000:+6.1f} "
                f"{s_.airfoil.t_max:6.4f} {s_.airfoil.camber_max:+7.5f} "
                f"{s_.twist_deg:+6.2f}"
                + (f"  dih {dih:5.1f}" if i else "           ")
                + ("  <- WINGLET" if dih >= 40.0 else ""))
        return out


@dataclass(frozen=True)
class Segment:
    """One spanwise segment, described by where it ENDS.

    The generator used to take four stations and a single dihedral, with
    every angle as its own positional argument -- seventeen of them, and
    no way to add a fifth station without an eighteenth. A segment list
    says the same thing and does not care how many there are.

    `blend` is the one that matters aerodynamically: 0 takes the root
    section, 1 takes the tip section, and anything between is a lerp in
    CST coefficient space. Before this existed the pipeline passed the
    SAME airfoil as both root and tip, which made the blend a no-op and
    gave the whole aircraft one camber line -- measured identical to five
    decimals at every station. Thickness varied; aerodynamic shape did
    not. A blended wing body that cannot vary its section spanwise is
    not really blended.
    """

    eta: float                  # outboard end of this segment
    chord_frac: float           # chord there, as a fraction of root chord
    sweep_deg: float            # LE sweep ACROSS this segment
    dihedral_deg: float         # dihedral ACROSS this segment
    twist_deg: float            # twist AT the outboard end
    blend: float = 0.0          # 0 = root section, 1 = tip section
    thickness_scale: float = 1.0


def lofted(
    half_span_m: float,
    root_chord_m: float,
    root_twist_deg: float,
    root_airfoil: Airfoil,
    tip_airfoil: Airfoil,
    segments: "tuple[Segment, ...]",
    root_thickness_scale: float = 1.0,
    name: str = "lofted",
) -> Planform:
    """Walk a segment list outboard, accumulating sweep and dihedral.

    Dihedral is PER SEGMENT, which is the whole point: a single constant
    dihedral from root to tip is one straight line, and a flying wing
    wants its dihedral outboard, where it buys roll stability without
    the roll-yaw coupling that inboard dihedral brings. Taken far enough
    on the last segment it stops being dihedral and becomes a WINGLET --
    and because the section simply moves up in z as eta advances, that
    winglet is still a spanwise loft, so it still prints in vase mode as
    one contour per layer. The overhang gate polices how fast it may
    turn up, with no special case anywhere.
    """
    if not segments:
        raise ValueError("need at least one segment")
    etas = [s_.eta for s_ in segments]
    if any(b <= a for a, b in zip(etas, etas[1:])):
        raise ValueError(f"segment etas must strictly increase, got {etas}")
    if abs(etas[-1] - 1.0) > 1e-9:
        raise ValueError(f"last segment must end at eta 1.0, got {etas[-1]}")

    stations = [Station(0.0, root_chord_m, 0.0, 0.0, root_twist_deg,
                        root_airfoil.scaled_thickness(root_thickness_scale))]
    x = z = 0.0
    y_prev = 0.0
    for sg in segments:
        y = sg.eta * half_span_m
        dy = y - y_prev
        x += dy * np.tan(np.radians(sg.sweep_deg))
        z += dy * np.tan(np.radians(sg.dihedral_deg))
        y_prev = y
        sec = blend(root_airfoil, tip_airfoil, float(np.clip(sg.blend, 0.0, 1.0)))
        stations.append(Station(sg.eta, root_chord_m * sg.chord_frac, x, z,
                                sg.twist_deg,
                                sec.scaled_thickness(sg.thickness_scale)))
    return Planform(half_span_m=half_span_m, stations=tuple(stations),
                    name=name)


def _cumtrapz(f: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(x))])


def faired(
    half_span_m: float,
    root_chord_m: float,
    root_twist_deg: float,
    root_airfoil: Airfoil,
    tip_airfoil: Airfoil,
    segments: "tuple[Segment, ...]",
    root_thickness_scale: float = 1.0,
    max_tip_rise_frac: float | None = None,
    n_stations: int = 33,
    name: str = "faired",
) -> Planform:
    """Loft through the ANGLES, then integrate. The fair version of lofted().

    lofted() places stations and lets PCHIP interpolate their x_le, z_le
    and chord. PCHIP never overshoots a VALUE, and that is exactly the
    trouble: sweep, dihedral and trailing-edge sweep are not values, they
    are SLOPES of those values, and PCHIP overshoots slopes freely. Traced
    on a random gen3 draw whose inputs were perfectly monotone -- body
    sweep 21, then 19, 16, 12.5 degrees -- the loft produced a leading
    edge sweeping 27 degrees just outboard of the nose, a trailing edge
    that swung aft and then forward at -44 degrees (the hump in every
    contact sheet), and dihedral that ran 0, 8.3, 5.7 and back up for an
    input of 6.3. Nothing in the inputs asked for any of it.

    So this interpolates the SLOPES themselves: each segment's sweep,
    dihedral and taper rate is placed at the segment's MIDPOINT, pinned
    to zero on the centreline (the mirror image must join tangent), held
    flat to the tip, and joined LINEARLY in slope space. x_le, z_le and
    chord are the integrals, so each edge is piecewise quadratic: C1,
    with curvature changing only at the knots.

    Linear rather than PCHIP, and the reason was measured, not assumed.
    PCHIP on the angles fixed the leading-edge overshoot (a 21 degree
    body peaked at 21.7 instead of 27) but left a bump behind the root:
    sweep and taper rate each got their own PCHIP derivative at the
    first knot, so near the centreline the leading edge swept back
    before the chord began to shrink, and the trailing edge went aft 8
    degrees and then forward 35. Trailing-edge slope is tan(sweep) plus
    taper rate; with both linear between shared knots that sum is linear
    too, so the trailing edge can only change direction AT a knot --
    which is to say, only where the segment values tell it to. The
    rounded nose falls out for free: a slope rising linearly from zero
    is a parabola.

    The tip chord still lands where the segments put it: the rate curve
    is rescaled so its integral matches.

    Twist, section blend and thickness scale are values in their own
    right and stay on mirrored PCHIP through the control stations.

    A winglet taller than `max_tip_rise_frac` of the semi-span cannot be
    expressed: the dihedral in excess of the inboard value is scaled back
    until it fits. That proportion is a design limit, and a limit the
    generator can express is one the optimizer will spend evaluations
    violating.

    Emits ~n_stations dense stations so the Planform's own interpolation
    only ever sees an already-smooth curve, plus the control etas, which
    are recorded separately.
    """
    if not segments:
        raise ValueError("need at least one segment")
    e_ctrl = np.array([0.0] + [float(s_.eta) for s_ in segments])
    if np.any(np.diff(e_ctrl) <= 0.0):
        raise ValueError(f"segment etas must strictly increase, got {e_ctrl[1:]}")
    if abs(e_ctrl[-1] - 1.0) > 1e-9:
        raise ValueError(f"last segment must end at eta 1.0, got {e_ctrl[-1]}")

    grid = np.linspace(0.0, 1.0, 801)
    knots = np.concatenate([[0.0], 0.5 * (e_ctrl[:-1] + e_ctrl[1:]), [1.0]])

    def slope_curve(per_segment) -> np.ndarray:
        # linear between knots, in SLOPE space: see the docstring
        v = np.asarray(per_segment, dtype=float)
        return np.interp(grid, knots, np.concatenate([[0.0], v, [v[-1]]]))

    def integrate(per_segment_deg) -> np.ndarray:
        t = np.tan(np.radians(np.asarray(per_segment_deg, dtype=float)))
        return half_span_m * _cumtrapz(slope_curve(t), grid)

    x_le = integrate([s_.sweep_deg for s_ in segments])

    dih = np.array([s_.dihedral_deg for s_ in segments], dtype=float)
    z_le = integrate(dih)
    if max_tip_rise_frac is not None and len(dih) > 1:
        limit = max_tip_rise_frac * half_span_m
        if z_le[-1] > limit:
            base = dih[0]
            excess = np.maximum(dih - base, 0.0)

            def scaled(lam: float) -> np.ndarray:
                return np.where(excess > 0.0, base + lam * excess, dih)

            if integrate(scaled(0.0))[-1] <= limit:
                lo, hi = 0.0, 1.0
                for _ in range(30):
                    mid = 0.5 * (lo + hi)
                    if integrate(scaled(mid))[-1] > limit:
                        hi = mid
                    else:
                        lo = mid
                z_le = integrate(scaled(lo))

    c_ctrl = root_chord_m * np.array([1.0] + [float(s_.chord_frac) for s_ in segments])
    rates = np.diff(c_ctrl) / np.diff(e_ctrl)
    rate = slope_curve(rates)
    got = float(np.trapezoid(rate, grid))
    want = float(c_ctrl[-1] - c_ctrl[0])
    if got * want > 0.0:
        rate = rate * (want / got)
    chord = np.maximum(c_ctrl[0] + _cumtrapz(rate, grid), 0.04 * root_chord_m)

    def mirrored(vals):
        v = np.asarray(vals, dtype=float)
        return PchipInterpolator(np.concatenate([-e_ctrl[:0:-1], e_ctrl]),
                                 np.concatenate([v[:0:-1], v]))

    twist = mirrored([root_twist_deg] + [s_.twist_deg for s_ in segments])
    blend_f = mirrored([0.0] + [s_.blend for s_ in segments])
    thick = mirrored([root_thickness_scale] + [s_.thickness_scale for s_ in segments])

    dense = np.linspace(0.0, 1.0, n_stations)
    gap = np.min(np.abs(dense[:, None] - e_ctrl[None, :]), axis=1)
    st_etas = np.sort(np.concatenate([dense[gap > 0.012], e_ctrl]))
    stations = []
    for e in st_etas:
        sec = blend(root_airfoil, tip_airfoil, float(np.clip(blend_f(e), 0.0, 1.0)))
        stations.append(Station(
            float(e), float(np.interp(e, grid, chord)),
            float(np.interp(e, grid, x_le)), float(np.interp(e, grid, z_le)),
            float(twist(e)), sec.scaled_thickness(float(thick(e)))))
    return Planform(half_span_m=half_span_m, stations=tuple(stations), name=name,
                    control_etas=tuple(float(e) for e in e_ctrl))


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
