"""Vase-mode (spiralize) layer stacks, and the gates that make them real.

The print orientation is the whole trick. A wing laid flat on the bed
cannot print in vase mode: a horizontal slice through a flat wing is two
or more disjoint contours, and the spiralize setting needs exactly one
closed loop per layer. Stand the panel on its ROOT instead, span along
printer Z, and every layer becomes the airfoil section at that span
station -- one loop, by construction, for the whole part. Sweep, taper
and twist all turn into XY motion of that loop as Z rises.

Three consequences follow, and they are gates, not warnings:

  1. Span becomes print height, so a panel taller than the Z envelope
     must be split spanwise (see segments.py) and rejoined on a spar.
  2. The bed footprint is chord x thickness, so root chord is capped by
     the bed diagonal.
  3. Sweep and taper become OVERHANG. A layer's material must land on
     the layer below it; measured perpendicular to the previous loop,
     not by point index, because points slide tangentially as the chord
     shrinks and that is not an overhang.

And one gift: the spar runs spanwise, which is now the print Z axis, so
the spar channel needs no hole through the single wall at all -- the
shell's own cavity IS the channel and a carbon tube slides in after the
print. `spar_fit` checks the tube clears every layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.spatial import ConvexHull

from ..geom.planform import Planform
from .bays import BaySpec, bay_point_budget, insert_bay
from .ribs import (POINTS_PER_RIB, RibSpec, insert_ribs,
                   min_clearance_mm, ribs_that_fit)


def _hull_indices(points: np.ndarray) -> np.ndarray:
    """Convex hull vertex indices; falls back to everything if degenerate."""
    try:
        return ConvexHull(points).vertices
    except Exception:
        return np.arange(len(points))


@dataclass(frozen=True)
class PrintSettings:
    """Printer and process envelope. Every gate below reads from here."""

    nozzle_mm: float = 0.4
    extrusion_width_mm: float = 0.45
    layer_h_mm: float = 0.25
    bed_x_mm: float = 256.0
    bed_y_mm: float = 256.0
    bed_z_mm: float = 250.0
    max_overhang_deg: float = 50.0      # from vertical; 50 is conservative
    min_te_mm: float = 1.0              # blunt TE: >= 2 extrusion widths
    te_blend: float = 0.35              # smooth-max width, mm
    spar_d_mm: float = 6.0
    spar_clearance_mm: float = 0.5
    spar_x_frac: float = 0.30           # chordwise seat of the spar
    filament_density_gcc: float = 0.60  # LW-PLA, foamed; 1.24 for solid PLA
    contour_points: int = 121           # per surface -> 2n-1 per loop
    ribs: bool = False
    rib_count: int = 3
    rib_pitch_mm: float = 30.0
    rib_clearance_factor: float = 1.10
    spar_avoid: tuple = ()
    """(centre, half_width) chord-fraction bands the ribs must keep out
    of -- the spar corridors, from washout.spars."""
    spar_corridors: tuple = ()
    """(x_frac, reach_eta) for every fitted spar, for the BORE gate.

    Separate from `spar_avoid`, which the ribs consume, because the two
    need different things: the truss must keep clear of a corridor
    wherever the tube goes, while the bore gate must only ask whether the
    tube fits in the panels it actually ENTERS.

    Without the reach, the gate failed the outer panels of every design
    -- correctly observing that the trainer's TE spar does not fit at
    0.54c beyond eta 0.76, which `spars.fit_all` already knows and
    reports as its reach. The spar is allowed to stop short: insisting one
    reach the tip would force the whole wing thick to satisfy its
    thinnest tenth. `min_spar_reach_frac` is the gate for that, and it is
    a mission constraint, not a print one."""
    elevon_chord: float = 0.0
    elevon_eta: float = 1.0
    """The control surface, as the EXPORTER sees it. Zero chord means the
    trailing edge stays attached and someone cuts it with a knife, which
    is what every generation before this one shipped.

    These duplicate two design variables on purpose. `elevon_chord` and
    `elevon_eta` are searched -- they set the control authority the score
    is computed from -- and they have to reach the geometry that gets
    printed, or the aircraft that flies is not the aircraft that was
    scored. `search.design` copies them across; nothing here invents
    them."""
    hinge_gap_mm: float = 0.8
    """Gap between the wing's cut face and the elevon's nose. One bead is
    not enough: the nose has to rotate."""
    min_first_layer_mm2: float = 300.0
    elevon_first_layer_mm2: float = 40.0
    """Bed-adhesion floors, per part class. Both are DECLARED process
    limits in the same category as `max_overhang_deg = 50` and
    `min_te_mm = 1.0`: chosen conservatively, not derived, and neither has
    been validated against a failed print. The elevon floor assumes a
    brim. See docs/ROADMAP-BUILD.md -- it is listed as a debt rather than
    dressed up as physics."""
    hinge_margin_deg: float = 4.0
    """Chamfer angle beyond the mission's own deflection limit, so the
    surface reaches its full travel before the geometry stops it."""
    """Rib slit and floor gap, as a multiple of extrusion width.

    One bead exactly is the physical requirement -- the beads must touch
    and weld -- but building to exactly one bead lands 3 microns short of
    the clearance rule after interpolation, which is meaningless
    physically and fails the gate. A 10% bias costs nothing and makes the
    geometry honestly satisfy the rule it is checked against."""

    @property
    def min_wall_mm(self) -> float:
        return self.extrusion_width_mm

    @property
    def bed_diagonal_mm(self) -> float:
        return float(np.hypot(self.bed_x_mm, self.bed_y_mm))


@dataclass
class LayerStack:
    """A single printable part: contours at rising Z, in printer mm."""

    z_mm: np.ndarray            # (L,)
    eta: np.ndarray             # (L,) span fraction of each layer
    contours: np.ndarray        # (L, N, 2) closed loops, printer XY
    settings: PrintSettings
    name: str = "panel"
    z_step_mm: float | None = None   # sampling pitch; None = the real layer
    has_ribs: bool = False
    n_upper: int | None = None
    """How many of the loop's points are the upper surface.

    None means the wing convention: 2n-1 points with ONE shared vertex at
    the leading edge, where the two surfaces genuinely meet, and a blunt
    trailing edge carrying `min_te_mm`.

    An elevon does not obey it. Its nose is a CUT, so the surfaces do not
    meet there and a shared vertex would be a feather edge thinner than
    the nozzle. It has 2n points and two blunt faces, and the pairing runs
    (i, 2n-1-i) instead. Making the split explicit is what lets one
    thickness gate serve both instead of silently misaligning by one."""
    spar_x_local: tuple = ()
    """Chord fractions of THIS LOOP at which the bore is measured.

    The gate used to measure at `settings.spar_x_frac`, a default of 0.30,
    while `spars.place` solved the real stations -- 0.17c and 0.54c on
    demon1. Two places disagreeing about where the spar is, which is the
    same shape of mistake as the bay and its contents. `build_stack` fills
    this from the solved corridors, converting into the loop's own chord
    when the panel is truncated at the hinge line."""
    role: str = "wing"
    """"wing" or "elevon". Not decoration: two of the printability gates
    are wing-panel rules and are meaningless on a control surface. An
    elevon has no spar, so asking whether an 8 mm tube fits its 6 mm
    section rejects a part that was never meant to hold one; and the
    300 mm2 bed-adhesion floor is sized for a 111 mm tall centre body,
    not for a 25 mm wide trailing-edge wedge."""

    @property
    def layers_per_sample(self) -> float:
        """How many printed layers each sampled contour stands for.

        The search evaluates thousands of designs and does not need the
        geometry at 0.25 mm to answer 'does it fit the bed, does it
        overhang, does the spar clear'. Those gates are scale-free, so it
        samples coarsely -- but MASS is not scale-free, so every sampled
        layer is weighted by how many real beads it represents. Coarse
        sampling then costs accuracy in the gates, never a wrong mass."""
        return (self.z_step_mm or self.settings.layer_h_mm) / self.settings.layer_h_mm

    @property
    def height_mm(self) -> float:
        return float(self.z_mm[-1] - self.z_mm[0])

    @property
    def footprint_mm(self) -> tuple[float, float]:
        p = self.contours.reshape(-1, 2)
        return float(np.ptp(p[:, 0])), float(np.ptp(p[:, 1]))

    def best_bed_rotation(self) -> tuple[float, float, float]:
        """(angle_deg, bbox_x, bbox_y) of the tightest placement.

        A wing panel is long and thin, so the bed's DIAGONAL is most of
        its usable length -- refusing a 300 mm chord on a 256 mm bed
        without trying 45 degrees would reject printable parts. Sweeping
        the rotation is exact enough here and costs nothing: the hull is
        a few hundred points."""
        p = self.contours.reshape(-1, 2)
        p = p[_hull_indices(p)]
        best = (0.0, np.inf, np.inf)
        for deg in np.arange(0.0, 90.0, 0.5):
            a = np.radians(deg)
            r = p @ np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            bx, by = float(np.ptp(r[:, 0])), float(np.ptp(r[:, 1]))
            if max(bx, by) < max(best[1], best[2]):
                best = (float(deg), bx, by)
        return best

    def wall_separation_mm(self) -> np.ndarray:
        """Skin-to-skin distance at every paired chord station.

        Paired points are the pre-rotation upper/lower pair at the same
        x/c, and twist is a RIGID rotation, so their distance is the true
        section thickness however the layer is turned.

        Two conventions, one gate. On the wing's 2n-1 loop the two
        surfaces SHARE the leading-edge vertex, so its 'thickness' is
        trivially zero and is dropped -- keeping it would sink every
        design. On an elevon's 2n loop nothing is shared: the nose is a
        cut face and its thickness is a real number that must be
        measured, because a bevel that closed it to a point is exactly
        the bug this pairing caught."""
        N = self.contours.shape[1]
        if self.n_upper is not None:
            n = int(self.n_upper)
            up = self.contours[:, :n]                     # TE -> nose
            lo = self.contours[:, n:][:, ::-1]            # TE -> nose
            return np.linalg.norm(up - lo, axis=2)
        n = (N + 1) // 2
        up = self.contours[:, :n][:, ::-1]      # LE -> TE
        lo = self.contours[:, n - 1:]           # LE -> TE
        return np.linalg.norm(up - lo, axis=2)[:, 1:]

    def extrusion_length_mm(self) -> float:
        seg = np.linalg.norm(np.diff(self.contours, axis=1), axis=2).sum(1)
        closing = np.linalg.norm(self.contours[:, 0] - self.contours[:, -1], axis=1)
        return float((seg + closing).sum())

    def mass_g(self) -> float:
        """Vase mode lays exactly one bead per layer, so shell mass is
        (loop length) x (bead cross-section) x density -- no infill term,
        no guessing. This is why the print is predictable to the gram."""
        s = self.settings
        bead_mm2 = s.extrusion_width_mm * s.layer_h_mm
        return (self.extrusion_length_mm() * self.layers_per_sample
                * bead_mm2 * 1e-3 * s.filament_density_gcc)

    def print_time_min(self, speed_mm_s: float = 120.0) -> float:
        return (self.extrusion_length_mm() * self.layers_per_sample
                / speed_mm_s / 60.0)


# ------------------------------------------------------------------ shaping


def thicken_for_nozzle(
    loop_unit: np.ndarray,
    chord_mm: float,
    settings: PrintSettings,
) -> np.ndarray:
    """Force minimum printable thickness on a unit-chord section.

    A sharp trailing edge is thinner than the nozzle for the last few
    percent of chord. In vase mode the contour is the bead's centreline,
    so where the surfaces converge closer than one extrusion width the
    nozzle crosses its own previous pass. We push the surfaces apart
    about the CAMBER LINE -- camber is what makes the lift and the
    pitching moment, so it must survive; thickness is what makes the
    print, so it gets bent.

    The floor is applied with a smooth max, not clip(): a hard corner in
    thickness puts a crease in the lofted skin and a visible facet in the
    slicer's spiral. eps sets how gently the two meet.
    """
    n = (len(loop_unit) + 1) // 2
    upper = loop_unit[:n][::-1]                 # LE -> TE
    lower = loop_unit[n - 1:]                   # LE -> TE
    x = upper[:, 0]
    cam = 0.5 * (upper[:, 1] + lower[:, 1])
    t = upper[:, 1] - lower[:, 1]

    t_min = settings.min_te_mm / chord_mm
    eps = max(settings.te_blend / chord_mm, 1e-6)
    t_new = 0.5 * (t + t_min + np.sqrt((t - t_min) ** 2 + eps**2))

    up = np.stack([x, cam + 0.5 * t_new], 1)
    lo = np.stack([x, cam - 0.5 * t_new], 1)
    return np.concatenate([up[::-1], lo[1:]], 0)


def build_stack(
    plan: Planform,
    settings: PrintSettings,
    eta0: float = 0.0,
    eta1: float = 1.0,
    name: str = "panel",
    z_step_mm: float | None = None,
    bay_cuts=(),
) -> LayerStack:
    """Slice a spanwise panel of the planform into printable layers.

    Print Z is arc length along the panel, so dihedral does not shorten
    the part -- dihedral is a joint angle applied at assembly, and the
    panel itself prints straight.
    """
    dih = plan.at(0.5 * (eta0 + eta1))
    tip, root = plan.at(eta1), plan.at(eta0)
    dy = (eta1 - eta0) * plan.half_span_m
    dz = tip.z_le_m - root.z_le_m
    panel_len_mm = float(np.hypot(dy, dz)) * 1000.0

    # Wholly outboard of the hinge station -> this panel is the wing
    # FORWARD of the elevon and its trailing edge is a cut face. Wholly,
    # not partly: a panel that changed shape half way up would need a
    # surface normal to the span, which is a roof.
    truncated = (settings.elevon_chord > 1e-6
                 and eta0 >= settings.elevon_eta - 1e-9)

    step = z_step_mm or settings.layer_h_mm
    n_layers = max(int(round(panel_len_mm / step)), 2)
    z = np.arange(n_layers) * step
    z = z[z <= panel_len_mm + 1e-9]
    eta = eta0 + (z / panel_len_mm) * (eta1 - eta0)

    rib_spec = RibSpec(n_ribs=settings.rib_count,
                       pitch_mm=settings.rib_pitch_mm,
                       max_overhang_deg=settings.max_overhang_deg,
                       enabled=settings.ribs,
                       avoid=tuple(settings.spar_avoid) + tuple(
                           # A bay is an exclusion band like a spar
                           # corridor, and for the same reason: the bay's
                           # floor IS where a rib's floor would be, so a
                           # rib inside the band would land on it. Given
                           # for the whole panel rather than only where
                           # the bay is open, because the rib count per
                           # layer has to stay constant.
                           (0.5 * (c.x0 + c.x1),
                            0.5 * (c.x1 - c.x0)
                            * plan.stations[0].chord_m * 1000.0)
                           for c in bay_cuts))
    if truncated:
        # The loop now spans [0, x_hinge], so the truss has to fit the box
        # that exists. Scaled rather than clipped: the ribs keep the same
        # fractions OF THE REMAINING CHORD, which keeps the truss evenly
        # spread instead of bunching it behind the spar corridors.
        xh = 1.0 - settings.elevon_chord
        rib_spec = replace(rib_spec,
                           x_first=rib_spec.x_first * xh,
                           x_last=rib_spec.x_last * xh,
                           x_clip=(rib_spec.x_clip[0] * xh,
                                   rib_spec.x_clip[1] * xh))
    if settings.ribs:
        # What the bare panel already spends of the overhang budget,
        # computed rather than measured. A contour point at chord
        # fraction s sits at x = x_le(eta) + s * chord(eta), so it
        # travels
        #     dx/dz = [dx_le/deta + s * dchord/deta] * deta/dz
        # and the worst case is at one of the two chord ends, s = 0 or 1.
        # Building a throwaway bare stack just to measure this made every
        # ribbed evaluation 4.9x more expensive than a plain one (3.80 s
        # against 0.78) and dominated a 3h 22m search.
        de = 1e-3
        e_mid = 0.5 * (eta0 + eta1)
        s0, s1 = plan.at(max(e_mid - de, 0.0)), plan.at(min(e_mid + de, 1.0))
        dxle = (s1.x_le_m - s0.x_le_m) / (2 * de)
        dc = (s1.chord_m - s0.chord_m) / (2 * de)
        deta_dz = (eta1 - eta0) / max(panel_len_mm / 1000.0, 1e-9)
        used = max(abs(dxle), abs(dxle + dc)) * deta_dz
        allowed = np.tan(np.radians(settings.max_overhang_deg))
        # The analytic bound is a LOWER bound: it tracks the leading and
        # trailing edges but not twist, thickness change or the trailing-
        # edge thickening, all of which also move contour points. Measured
        # against the real overhang across nine panels it runs 1.21x to
        # 1.82x low, so 2.0 is the factor that keeps it conservative. The
        # cost is a shallower truss, which is the right way to be wrong.
        rib_spec = replace(rib_spec,
                           rate_mm_per_mm=max(allowed - 2.0 * used, 0.02))
    cuts = tuple(bay_cuts)
    # How many ribs this panel can actually take, after the bay bands are
    # removed from the chord. Per panel, because the exclusions are per
    # panel -- and the point budget follows it, so the invariant holds
    # within a stack without pretending every panel has the same truss.
    n_rib_eff = ribs_that_fit(
        rib_spec, plan.at(0.5 * (eta0 + eta1)).chord_m * 1000.0,
        settings.extrusion_width_mm * settings.rib_clearance_factor)
    # Solved ONCE, at the panel's mid station: the gaps move with the
    # local chord and the share between them is an integer, so a layout
    # recomputed per layer made ribs hop between gaps from one layer to
    # the next.
    rib_spec = replace(rib_spec, n_ribs=n_rib_eff).with_layout(
        plan.at(0.5 * (eta0 + eta1)).chord_m * 1000.0)
    n_pts = 2 * settings.contour_points - 1 + (
        POINTS_PER_RIB * n_rib_eff) + bay_point_budget(len(cuts))
    contours = np.empty((len(z), n_pts, 2))
    for k, e in enumerate(eta):
        st = plan.at(float(e))
        chord_mm = st.chord_m * 1000.0
        loop = st.airfoil.coords(settings.contour_points)
        loop = thicken_for_nozzle(loop, chord_mm, settings)
        if truncated:
            from .elevons import hinge_x, truncate_loop
            loop = truncate_loop(loop, hinge_x(settings))
        for c in cuts:
            loop = insert_bay(loop, chord_mm, float(z[k]), c,
                              settings.extrusion_width_mm)
        if n_rib_eff > 0:
            clr = settings.extrusion_width_mm * settings.rib_clearance_factor
            loop = insert_ribs(loop, chord_mm, float(z[k]), rib_spec, clr, clr)
        # twist about the quarter chord, then scale to mm and sweep
        p = loop - np.array([0.25, 0.0])
        a = np.radians(-st.twist_deg)
        ca, sa = np.cos(a), np.sin(a)
        rot = np.stack([p[:, 0] * ca - p[:, 1] * sa,
                        p[:, 0] * sa + p[:, 1] * ca], 1)
        contours[k] = rot * chord_mm + np.array(
            [st.x_le_m * 1000.0 + 0.25 * chord_mm, 0.0])

    # centre the whole part on the bed
    # The solved corridors, in this loop's own chord. A truncated panel's
    # loop spans [0, x_hinge] of the original chord, so a spar at 0.54c
    # sits at 0.54/x_hinge of what remains.
    xh = (1.0 - settings.elevon_chord) if truncated else 1.0
    local = tuple(float(np.clip(c / max(xh, 1e-6), 0.0, 1.0))
                  for c, reach in settings.spar_corridors
                  if reach >= eta0 + 1e-9)

    flat = contours.reshape(-1, 2)
    contours -= 0.5 * (flat.min(0) + flat.max(0))
    return LayerStack(z_mm=z, eta=eta, contours=contours,
                      settings=settings, name=name, z_step_mm=step,
                      has_ribs=n_rib_eff > 0, spar_x_local=local)


# ------------------------------------------------------------------- gates


@dataclass
class Gate:
    name: str
    passed: bool
    value: float
    limit: float
    units: str
    detail: str = ""

    def line(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        return (f"  [{mark}] {self.name:<22} {self.value:8.2f} {self.units:<6}"
                f" (limit {self.limit:g}) {self.detail}")


@dataclass
class Printability:
    gates: list[Gate] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(g.passed for g in self.gates)

    def report(self) -> str:
        head = "printability: " + ("OK" if self.ok else "REJECTED")
        return "\n".join([head] + [g.line() for g in self.gates])

    def failures(self) -> list[str]:
        return [g.name for g in self.gates if not g.passed]


def overhang_deg(stack: LayerStack, stride: int = 4) -> tuple[float, float]:
    """Worst overhang angle from vertical, and the Z where it happens.

    Measured as the distance from each point of layer k+1 to the NEAREST
    point anywhere on layer k -- 'is there material under me' -- not the
    distance to the same point index. Index distance counts tangential
    sliding (which happens on every tapered section as the chord shrinks)
    as if it were an overhang, and would reject perfectly printable wings.
    """
    worst, worst_z = 0.0, 0.0
    c = stack.contours
    for k in range(0, len(c) - stride, stride):
        a, b = c[k], c[k + stride]
        d = np.linalg.norm(b[:, None, :] - a[None, :, :], axis=2).min(1)
        # rise is the ACTUAL Z between the two sampled contours, not the
        # nominal layer height: when the search samples coarsely those
        # differ by 8x, and using the nominal value reports 82 deg of
        # overhang on a wing that really has 42.
        rise = float(stack.z_mm[k + stride] - stack.z_mm[k])
        ang = np.degrees(np.arctan2(d.max(), max(rise, 1e-9)))
        if ang > worst:
            worst, worst_z = float(ang), float(stack.z_mm[k])
    return worst, worst_z


def ramp_budget(stack: LayerStack, x0: float, x1: float,
                stride: int = 4) -> float:
    """How fast an internal feature may deepen, in mm of depth per mm of Z.

    A bay floor, a servo pocket or a hatch rebate cannot appear abruptly:
    a wall normal to the span is a ROOF in this print orientation, and
    spiralize has no top layers and cannot bridge. Features therefore fade
    in and out, and the fade rate is an overhang like any other.

    The available rate is bounded in QUADRATURE, not by subtraction, and
    the distinction decides whether a battery bay is possible at all.
    `overhang_deg` measures, for each point of layer k+1, the distance to
    the nearest point anywhere on layer k. A feature ramping in the
    THICKNESS direction adds its own dy to whatever dx that point already
    carries from the wing's taper, sweep and twist, so

        hypot(dx, dy) / dz <= tan(theta_max)
        =>  dy/dz <= sqrt(tan(theta_max)^2 - (dx/dz)^2)

    Subtracting linearly -- which is how the rib truss budget in
    `build_stack` is written, correctly, because a rib's motion is
    CHORDWISE and adds to the wing's own chordwise motion -- gives the
    trainer's centre body 0.36 mm/mm, so a 20 mm feature would need 56 mm
    of span and would not fit the 111 mm panel. In quadrature the same
    panel has 0.796 mm/mm and needs 25 mm, which it has.

    Measured on the built contours over the chord band [x0, x1] of the
    upper surface, where a floor detour lives -- not from the analytic
    bound, because that bound tracks the leading and trailing edges only
    and runs 1.21x to 1.82x low.

    Returns 0.0 if the wing alone has already spent the whole budget
    somewhere in the band, which means no feature can ramp there at all.
    """
    s = stack.settings
    lim = np.tan(np.radians(s.max_overhang_deg))
    c = stack.contours
    n = (c.shape[1] + 1) // 2
    worst = lim
    for k in range(0, len(c) - stride, stride):
        a, b = c[k], c[k + stride]
        rise = float(stack.z_mm[k + stride] - stack.z_mm[k])
        if rise <= 0.0:
            continue
        d = np.linalg.norm(b[:, None, :] - a[None, :, :], axis=2)
        dv = b - a[d.argmin(1)]
        up = b[:n]
        lo_x, hi_x = up[:, 0].min(), up[:, 0].max()
        frac = (up[:, 0] - lo_x) / max(hi_x - lo_x, 1e-9)
        sel = (frac >= min(x0, x1)) & (frac <= max(x0, x1))
        if not sel.any():
            continue
        dx = np.abs(dv[:n][sel, 0]) / rise
        avail = np.sqrt(np.maximum(lim * lim - dx * dx, 0.0))
        worst = min(worst, float(avail.min()))
    return float(max(worst, 0.0))


def ramp_span_mm(stack: LayerStack, depth_mm: float,
                 x0: float, x1: float) -> float:
    """Span needed to fade a feature `depth_mm` deep in or out. inf if it
    cannot be done in this panel at all."""
    rate = ramp_budget(stack, x0, x1)
    return float(depth_mm / rate) if rate > 0.0 else float("inf")


def _inscribed_gap(contour: np.ndarray, x: float, n: int = 17) -> float:
    """Diameter of the largest circle that fits at chord station x.

    This is the question the bore gate is actually asking. A spar is a
    round tube entering along the print Z axis, so its cross-section in
    the bed plane is a circle, and what decides whether it goes in is the
    largest circle the contour will hold -- not the section's thickness,
    and not the contour's vertical extent.

    The two approximations that came before both disagree with it and with
    each other. `spars.depth_at` takes the aerofoil's thickness at x,
    which is measured normal-ish to the chord and ignores twist;
    `_vertical_extent` measures the rotated contour's height on a vertical
    line, which near the leading edge cuts across a steeply sloping
    surface. On the trainer's tip panel they differ by about half a
    millimetre -- enough to decide an 8 mm gate either way, which is
    exactly the kind of 0.5 mm two places can quietly disagree by.

    Centres are scanned along the vertical line at x and the clearance at
    each is its distance to the nearest contour segment; twice the best
    clearance is the usable diameter. Scanned rather than solved because
    the section is not convex -- a reflexed aerofoil with a rib detour has
    more than one local optimum, and a root-finder would return whichever
    it started next to.
    """
    p = np.asarray(contour, dtype=float)
    a = p
    ab = np.roll(p, -1, axis=0) - a
    denom = np.einsum("ij,ij->i", ab, ab)
    denom[denom < 1e-12] = 1e-12

    ys = _crossings(p, x)
    if ys.size < 2:
        return 0.0
    y0, y1 = ys.min(), ys.max()
    cy = np.linspace(y0, y1, n)[1:-1]
    if cy.size == 0:
        return 0.0
    c = np.stack([np.full(cy.shape, x), cy], 1)            # (m, 2)

    ap = c[:, None, :] - a[None, :, :]                      # (m, N, 2)
    tt = np.clip(np.einsum("mnk,nk->mn", ap, ab) / denom[None, :], 0.0, 1.0)
    closest = a[None, :, :] + tt[:, :, None] * ab[None, :, :]
    d = np.linalg.norm(c[:, None, :] - closest, axis=2).min(1)   # (m,)
    return float(2.0 * d.max())


def _crossings(contour: np.ndarray, x: float) -> np.ndarray:
    """y of every point where the closed contour meets the line at x."""
    x0, y0 = contour[:, 0], contour[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    hit = (((x0 - x) * (x1 - x)) <= 0.0) & (np.abs(x1 - x0) > 1e-12)
    if not hit.any():
        return np.zeros(0)
    tt = (x - x0[hit]) / (x1[hit] - x0[hit])
    return y0[hit] + tt * (y1[hit] - y0[hit])


def _vertical_extent(contour: np.ndarray, x: float) -> float:
    """Skin-to-skin height of a closed contour on the vertical line at x.

    A RAY CAST, not an index lookup, and that matters. The old version
    took `n = (N + 1) // 2` as the upper/lower split and paired index i
    against its mirror -- which is exactly right for a plain section and
    wrong by `n_ribs` points once `insert_ribs` has reparametrised the
    upper surface into `n + 2 * n_ribs` points. So the bore gate has been
    pairing an upper station against the wrong lower station on every
    ribbed panel ever exported. It passed anyway, because the
    misalignment usually still lands in thick material; truncating the
    panel at the hinge line moved it somewhere it did not.

    `wall_separation_mm` has the same assumption and `check` knows it --
    it switches to the general `min_clearance_mm` when ribs are present.
    This gate had no such guard.

    The OUTERMOST pair of crossings is taken, so rib detours inside the
    section are ignored: a rib in the corridor is the corridor's problem
    and `min_clearance_mm` already reports it.
    """
    y0 = contour[:, 1]
    x0 = contour[:, 0]
    x1 = np.roll(x0, -1)
    y1 = np.roll(y0, -1)
    # segments straddling the line, either direction
    hit = ((x0 - x) * (x1 - x)) <= 0.0
    hit &= np.abs(x1 - x0) > 1e-12
    if not hit.any():
        return 0.0
    tt = (x - x0[hit]) / (x1[hit] - x0[hit])
    ys = y0[hit] + tt * (y1[hit] - y0[hit])
    return float(ys.max() - ys.min())


def spar_fit(stack: LayerStack) -> tuple[float, float]:
    """Largest spar tube the cavity accepts, and the Z of the pinch point.

    At the spar station the usable diameter is the local skin-to-skin
    thickness minus two wall widths (the bead sits inside the contour)
    minus the fit clearance.
    """
    s = stack.settings
    fracs = stack.spar_x_local or (s.spar_x_frac,)
    best, pinch_z = np.inf, 0.0
    for k, layer in enumerate(stack.contours):
        lo_x, hi_x = layer[:, 0].min(), layer[:, 0].max()
        chord = hi_x - lo_x
        for f in fracs:
            # the inscribed circle already stops one bead inside the
            # contour's centreline, so only the fit clearance is deducted
            gap = (_inscribed_gap(layer, lo_x + float(f) * chord)
                   - s.extrusion_width_mm - s.spar_clearance_mm)
            if gap < best:
                best, pinch_z = float(gap), float(stack.z_mm[k])
    return best, pinch_z


def check(stack: LayerStack) -> Printability:
    """Every reason a vase-mode wing fails on the bed, as one report."""
    s = stack.settings
    gates: list[Gate] = []

    deg, bx, by = stack.best_bed_rotation()
    fits = bx <= s.bed_x_mm and by <= s.bed_y_mm
    gates.append(Gate("bed footprint", fits, max(bx, by),
                      max(s.bed_x_mm, s.bed_y_mm), "mm",
                      f"{bx:.0f}x{by:.0f} at {deg:.0f} deg on the bed"))
    gates.append(Gate("print height", stack.height_mm <= s.bed_z_mm,
                      stack.height_mm, s.bed_z_mm, "mm",
                      "split the panel spanwise" if stack.height_mm > s.bed_z_mm else ""))

    ang, z_at = overhang_deg(stack)
    gates.append(Gate("max overhang", ang <= s.max_overhang_deg, ang,
                      s.max_overhang_deg, "deg", f"at z={z_at:.0f} mm"))

    # One simple closed loop per layer is guaranteed, not sampled: each
    # contour is built as y_upper > y_lower over a monotone x and then
    # RIGIDLY rotated, so it cannot self-intersect unless the thickness
    # floor was violated. Checking that floor proves the premise, and
    # costs one array op instead of 40M segment-pair tests.
    if stack.has_ribs:
        # With ribs the upper/lower index pairing no longer describes the
        # section, so clearance is measured the general way: every vertex
        # against every non-adjacent segment. Sampled, because it is
        # O(n^2) per layer and the geometry varies smoothly with Z.
        step = max(len(stack.contours) // 40, 1)
        t_min = min(min_clearance_mm(c, skip=8)
                    for c in stack.contours[::step])
        detail = f"general contour clearance, {stack.settings.rib_count} ribs"
    else:
        t_min = float(stack.wall_separation_mm().min())
        detail = "=> single simple loop per layer"
    gates.append(Gate("min wall separation", t_min >= s.min_wall_mm, t_min,
                      s.min_wall_mm, "mm", detail))

    if stack.role == "wing":
        spar, spar_z = spar_fit(stack)
        gates.append(Gate("spar bore", spar >= s.spar_d_mm, spar, s.spar_d_mm,
                          "mm", f"pinch at z={spar_z:.0f} mm"))

    area0 = 0.5 * abs(np.dot(stack.contours[0][:, 0],
                             np.roll(stack.contours[0][:, 1], -1))
                      - np.dot(stack.contours[0][:, 1],
                               np.roll(stack.contours[0][:, 0], -1)))
    floor = (s.elevon_first_layer_mm2 if stack.role == "elevon"
             else s.min_first_layer_mm2)
    gates.append(Gate("first-layer area", area0 >= floor, area0, floor, "mm2",
                      "bed adhesion" + (" (brim)" if stack.role == "elevon"
                                        else "")))
    return Printability(gates)


# ------------------------------------------------------------ segmentation


def arc_length_mm(plan: Planform, eta0: float = 0.0, eta1: float = 1.0,
                  n: int = 400) -> float:
    """True length along the panel from eta0 to eta1, dihedral included.

    Print height is arc length, not projected span: a panel with 5 deg of
    dihedral is 0.4% taller than its span suggests, and the Z envelope
    gate is decided in millimetres."""
    e = np.linspace(eta0, eta1, n)
    y = e * plan.half_span_m
    z = np.array([plan.at(float(v)).z_le_m for v in e])
    return float(np.hypot(np.diff(y), np.diff(z)).sum()) * 1000.0


def panel_etas(plan: Planform, settings: PrintSettings,
               z_margin_mm: float = 8.0,
               min_panel_mm: float = 5.0) -> list[tuple[float, float]]:
    """Split the half-span into panels that each fit the Z envelope.

    Breaks land on the planform's own stations first -- a kink is where
    the shape changes fastest and where a spar joint is least intrusive
    anyway -- and any remaining over-tall panel is divided into equal
    pieces. Equal pieces, not greedy-fill, so the joints stay symmetric
    and one spar length serves them all."""
    limit = settings.bed_z_mm - z_margin_mm
    # CONTROL stations, not every station: the faired loft emits ~35
    # dense stations to carry its curves, and splitting the print at
    # each of them would turn three panels into thirty.
    #
    # And the ELEVON station, whenever there is one. The elevon begins at
    # a spanwise station, so the wing's trailing edge has to disappear
    # there -- and a surface normal to the span is a roof in this print
    # orientation, which spiralize cannot build. Breaking the print at
    # `elevon_eta` makes every panel wholly plain or wholly truncated and
    # turns that roof into a print joint.
    breaks = sorted({0.0, 1.0} | set(plan.controls))
    if 0.0 < settings.elevon_eta < 1.0 and settings.elevon_chord > 1e-6:
        # Merge, do not just add. Micro's hinge station is eta 0.438 and a
        # planform control station sits at 0.436 -- 0.2 mm of arc length
        # apart -- so adding it produced a panel 0.2 mm tall weighing
        # 0.0 g, which is a part in the parts list and a joint in the
        # assembly that cannot exist. The HINGE station wins any tie: it
        # is where the trailing edge has to stop, while a control station
        # is only where the loft's curvature is described.
        e = float(settings.elevon_eta)
        breaks = [b for b in breaks
                  if b in (0.0, 1.0)
                  or arc_length_mm(plan, min(b, e), max(b, e)) > min_panel_mm]
        breaks = sorted(set(breaks) | {e})
    return _split_to_envelope(plan, breaks, limit)


def _split_to_envelope(plan: Planform, breaks, limit: float) -> list:
    """Divide any over-tall span between breaks into equal pieces."""
    out: list[tuple[float, float]] = []
    for a, b in zip(breaks, breaks[1:]):
        length = arc_length_mm(plan, a, b)
        n = max(int(np.ceil(length / limit)), 1)
        edges = np.linspace(a, b, n + 1)
        out.extend((float(u), float(v)) for u, v in zip(edges, edges[1:]))
    return out


def build_panels(plan: Planform, settings: PrintSettings,
                 z_margin_mm: float = 8.0,
                 z_step_mm: float | None = None,
                 bays=()) -> list[LayerStack]:
    """Every printable part of the right half wing, root outboard.

    `bays` are (name, x0, x1, eta1, depth_mm, ramp_mm) for the openings
    to cut, with the ramp length ALREADY measured by the caller on a bare
    panel. Measuring it here would be circular: the ramp's own dive and
    climb walls move in Z, so a budget measured on a panel that already
    has the cut comes back as zero and the ramp as infinite. That is
    exactly what the first version did. A
    bay is cut only into the panel that CONTAINS it -- a bay straddling a
    joint is already an infeasible design and the gate reports it -- and
    its span is converted into that panel's own z, because the depth
    profile is a function of print height.
    """
    spans = panel_etas(plan, settings, z_margin_mm)
    out = []
    for i, (a, b) in enumerate(spans):
        cuts = []
        for name, x0, x1, eta1, depth_mm, ramp_mm in bays:
            if not (a <= eta1 <= b + 1e-9):
                continue
            if not np.isfinite(ramp_mm):
                continue                       # cannot be closed: no cut
            cuts.append(BaySpec(name, x0, x1,
                                arc_length_mm(plan, a, min(eta1, b)),
                                ramp_mm, depth_mm,
                                settings.extrusion_width_mm))
        out.append(build_stack(plan, settings, a, b,
                               name=f"{plan.name}_p{i}",
                               z_step_mm=z_step_mm, bay_cuts=tuple(cuts)))
    return out
