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
from .ribs import POINTS_PER_RIB, RibSpec, insert_ribs, min_clearance_mm


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
        """Skin-to-skin distance at every paired chord station, (L, n-1).

        Paired points are the pre-rotation upper/lower pair at the same
        x/c, and twist is a RIGID rotation, so their distance is the true
        section thickness however the layer is turned. The leading-edge
        point is dropped: the two surfaces share it, so its 'thickness'
        is trivially zero and would sink every design."""
        n = (self.contours.shape[1] + 1) // 2
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

    step = z_step_mm or settings.layer_h_mm
    n_layers = max(int(round(panel_len_mm / step)), 2)
    z = np.arange(n_layers) * step
    z = z[z <= panel_len_mm + 1e-9]
    eta = eta0 + (z / panel_len_mm) * (eta1 - eta0)

    rib_spec = RibSpec(n_ribs=settings.rib_count,
                       pitch_mm=settings.rib_pitch_mm,
                       max_overhang_deg=settings.max_overhang_deg,
                       enabled=settings.ribs,
                       avoid=tuple(settings.spar_avoid))
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
    n_pts = 2 * settings.contour_points - 1 + (
        POINTS_PER_RIB * settings.rib_count if settings.ribs else 0)
    contours = np.empty((len(z), n_pts, 2))
    for k, e in enumerate(eta):
        st = plan.at(float(e))
        chord_mm = st.chord_m * 1000.0
        loop = st.airfoil.coords(settings.contour_points)
        loop = thicken_for_nozzle(loop, chord_mm, settings)
        if settings.ribs:
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
    flat = contours.reshape(-1, 2)
    contours -= 0.5 * (flat.min(0) + flat.max(0))
    return LayerStack(z_mm=z, eta=eta, contours=contours,
                      settings=settings, name=name, z_step_mm=step,
                      has_ribs=settings.ribs)


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


def spar_fit(stack: LayerStack) -> tuple[float, float]:
    """Largest spar tube the cavity accepts, and the Z of the pinch point.

    At the spar station the usable diameter is the local skin-to-skin
    thickness minus two wall widths (the bead sits inside the contour)
    minus the fit clearance.
    """
    s = stack.settings
    best, pinch_z = np.inf, 0.0
    n = (stack.contours.shape[1] + 1) // 2
    for k, layer in enumerate(stack.contours):
        upper, lower = layer[:n][::-1], layer[n - 1:]
        chord = layer[:, 0].max() - layer[:, 0].min()
        x_target = layer[:, 0].min() + s.spar_x_frac * chord
        i = int(np.argmin(np.abs(upper[:, 0] - x_target)))
        gap = (upper[i, 1] - lower[i, 1]) - 2 * s.extrusion_width_mm - s.spar_clearance_mm
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

    spar, spar_z = spar_fit(stack)
    gates.append(Gate("spar bore", spar >= s.spar_d_mm, spar, s.spar_d_mm, "mm",
                      f"pinch at z={spar_z:.0f} mm"))

    area0 = 0.5 * abs(np.dot(stack.contours[0][:, 0],
                             np.roll(stack.contours[0][:, 1], -1))
                      - np.dot(stack.contours[0][:, 1],
                               np.roll(stack.contours[0][:, 0], -1)))
    gates.append(Gate("first-layer area", area0 >= 300.0, area0, 300.0, "mm2",
                      "bed adhesion"))
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
               z_margin_mm: float = 8.0) -> list[tuple[float, float]]:
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
    breaks = sorted({0.0, 1.0} | set(plan.controls))
    out: list[tuple[float, float]] = []
    for a, b in zip(breaks, breaks[1:]):
        length = arc_length_mm(plan, a, b)
        n = max(int(np.ceil(length / limit)), 1)
        edges = np.linspace(a, b, n + 1)
        out.extend((float(u), float(v)) for u, v in zip(edges, edges[1:]))
    return out


def build_panels(plan: Planform, settings: PrintSettings,
                 z_margin_mm: float = 8.0,
                 z_step_mm: float | None = None) -> list[LayerStack]:
    """Every printable part of the right half wing, root outboard."""
    spans = panel_etas(plan, settings, z_margin_mm)
    return [build_stack(plan, settings, a, b, name=f"{plan.name}_p{i}",
                        z_step_mm=z_step_mm)
            for i, (a, b) in enumerate(spans)]
