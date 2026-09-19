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
from .bays import (BaySpec, bay_point_budget, insert_detours,
                   clamp_band as _bay_clamp_band)
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
    def lid_mm(self) -> float:
        """Hatch lid thickness, and therefore the depth of the ledge the
        lid rests on. DERIVED: the lid is a vase part, a lens of two
        beads one `rib_clearance_factor` apart so they weld into one
        curved plate -- the rule the rib slits already use -- and the
        ledge is cut to that. A declared 1.35 mm ledge under a printed
        0.95 mm lid left the lid 0.4 mm below the skin."""
        return self.extrusion_width_mm * (1.0 + self.rib_clearance_factor)

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
    has_cuts: bool = False
    origin_mm: tuple = (0.0, 0.0)
    """The planform point (mm aft of the root LE, mm above the chord
    line) that sits at the part's printer-frame origin. Parts are
    centred on the bed, and a cut at an absolute station has to be found
    again in the centred frame."""
    bay_specs: tuple = ()
    """The `BaySpec`s cut into this stack, in its own z. Carried so a
    companion part -- a lid -- can be built to the opening as cut, and
    so a failing layer can be inspected against the spec that shaped it."""
    """Whether any bay, pocket or groove detour is in the loop.

    Like `has_ribs`, this decides how wall separation is measured. A
    cut inserts six vertices into ONE skin, so the upper/lower index
    pairing the plain-section test relies on is off by six from the cut
    aft, and the gate then measures a vertex against the wrong mirror
    point. The search's first pass -- bare shell, bays cut, no ribs yet --
    was being judged that way."""
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


def _avoid_band(cut: BaySpec, z: np.ndarray, chord_mm: np.ndarray,
                x_le_mm: np.ndarray, layout_chord_mm: float,
                pad_mm: float) -> tuple[float, float] | None:
    """(centre chord fraction, half-width mm) of a cut, for the ribs.

    The ENVELOPE of the cut's band over the layers this panel will
    actually BUILD, in each layer's own chord, because the rib layout is
    solved once per panel while a cut at an absolute station slides
    across the chord fractions as the wing sweeps under it: on the
    trainer's centre body the pack's band runs from 0.29c-0.59c at the
    root to 0.15c-0.48c where its ramp ends, and a rib clear of the band
    at the panel's mid station sat inside it at the root.

    Over the layers BUILT, and every one of them, which makes the
    envelope exact rather than an estimate. Sampling nine stations of a
    thousand left the trainer's servo cut reaching 0.7175c while the
    envelope said 0.7089c, and `with_layout` duly placed a rib at 0.7117c
    -- in the sliver between the estimate's edge and the rib band's aft
    limit, and inside the real opening. `rebuild_skin` then refused the
    layer, which is a crash in the geometry rather than a gate with a
    number. A detour is only ever cut at one of these z values, so an
    envelope taken over exactly them cannot be exceeded by anything that
    prints.

    Over the layers where the bay is OPEN or still scribing its groove,
    which is where its walls are geometry that may not move. A bay keeps
    its six vertices for the whole panel, but once it has closed they lie
    on the skin and `pack_detours` is free to park them in whatever chord
    the openings and the truss have left. Excluding the closed run as
    well cost the trainer's centre body its entire rib truss -- the two
    bays and two spar corridors between them covered all of 0.20c to
    0.72c -- and the buckling pitch the structure had sized was then
    silently not delivered.

    `pad_mm` is the clearance a rib's slit keeps from the opening's wall.
    Returns None when the bay is open nowhere on this panel.
    """
    lo, hi = np.inf, -np.inf
    for k in range(len(z)):
        zk = float(z[k])
        if cut.depth_frac(zk) <= 0.0 and cut.groove_frac(zk) <= 0.0:
            continue                     # closed: its vertices may move
        band = cut.band(float(chord_mm[k]), float(x_le_mm[k]))
        lo, hi = min(lo, band[0]), max(hi, band[1])
    if not np.isfinite(lo):
        return None
    return (0.5 * (lo + hi), 0.5 * (hi - lo) * layout_chord_mm + pad_mm)


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

    # Every layer's station, once: the exclusion envelopes below and the
    # contour loop beneath both want them, and `plan.at` is the expensive
    # call in this function.
    stations = [plan.at(float(e)) for e in eta]
    chord_of = np.array([s_.chord_m * 1000.0 for s_ in stations])
    x_le_of = np.array([s_.x_le_m * 1000.0 for s_ in stations])
    layout_chord = plan.at(0.5 * (eta0 + eta1)).chord_m * 1000.0
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
                           band for band in (
                               _avoid_band(c, z, chord_of, x_le_of,
                                           layout_chord,
                                           settings.extrusion_width_mm
                                           * settings.rib_clearance_factor)
                               for c in bay_cuts)
                           if band is not None))
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
    for k, st in enumerate(stations):
        chord_mm = float(chord_of[k])
        loop = st.airfoil.coords(settings.contour_points)
        loop = thicken_for_nozzle(loop, chord_mm, settings)
        if truncated:
            from .elevons import hinge_x, truncate_loop
            loop = truncate_loop(loop, hinge_x(settings))
        if cuts or n_rib_eff > 0:
            # ONE pass for every detour on the section -- bays and ribs
            # together. Cutting bays one at a time handed the second call
            # a loop whose leading edge was no longer at index n-1; and
            # inserting the ribs after the bays re-interpolated a skin
            # that already had vertical walls in it, which put a ledge
            # vertex 7.2 mm down its wall between two adjacent layers.
            clr = settings.extrusion_width_mm * settings.rib_clearance_factor
            loop = insert_detours(loop, chord_mm, float(z[k]), cuts,
                                  rib_spec if n_rib_eff > 0 else None,
                                  slit_mm=clr, gap_mm=clr,
                                  min_groove_mm=settings.extrusion_width_mm,
                                  x_le_mm=st.x_le_m * 1000.0)
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
    origin = 0.5 * (flat.min(0) + flat.max(0))
    contours -= origin
    return LayerStack(z_mm=z, eta=eta, contours=contours,
                      settings=settings, name=name, z_step_mm=step,
                      has_ribs=n_rib_eff > 0, has_cuts=bool(cuts),
                      origin_mm=(float(origin[0]), float(origin[1])),
                      bay_specs=tuple(cuts), spar_x_local=local)


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


def _dist_to_segments(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Distance from each of `pts` to the nearest segment of the closed
    polygon `poly`. Vectorised: one (m, n) matrix per call."""
    a = poly
    ab = np.roll(poly, -1, axis=0) - a
    denom = np.einsum("ij,ij->i", ab, ab)
    denom[denom < 1e-12] = 1e-12
    ap = pts[:, None, :] - a[None, :, :]
    t = np.clip(np.einsum("ijk,jk->ij", ap, ab) / denom[None, :], 0.0, 1.0)
    closest = a[None, :, :] + t[:, :, None] * ab[None, :, :]
    return np.linalg.norm(pts[:, None, :] - closest, axis=2).min(1)


def overhang_deg(stack: LayerStack, stride: int = 4) -> tuple[float, float]:
    """Worst overhang angle from vertical, and the Z where it happens.

    Measured as the distance from each vertex of layer k+1 to the nearest
    SEGMENT of layer k -- 'is there a bead under me' -- not to the same
    vertex index, and not to the nearest vertex either. Index distance
    counts tangential sliding (which happens on every tapered section as
    the chord shrinks) as if it were an overhang. Nearest-vertex distance
    is right only while vertices are dense on both layers: a pocket's
    floor is one 30 mm segment with no vertex on it, and the layer above,
    where the pocket had faded to plain skin, put every vertex along that
    run 13 mm from the nearest vertex below and 0.1 mm from the bead. The
    bead is continuous along its segments; the printer sees segments.
    """
    worst, worst_z = 0.0, 0.0
    c = stack.contours
    for k in range(0, len(c) - stride, stride):
        d = _dist_to_segments(c[k + stride], c[k])
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
                stride: int = 4, side: str = "upper") -> float:
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
    skin the bay opens from -- the upper for a hatch, the LOWER for a
    servo pocket -- not from the analytic bound, because that bound
    tracks the leading and trailing edges only and runs 1.21x to 1.82x
    low. The two skins move differently as the section tapers, and
    measuring the upper for a pocket in the lower let the trainer's servo
    pocket ramp in at 1.25 mm/mm against a 1.19 limit: 51.3 degrees on
    both sides of the p0/p1 joint, a failure of five percent that the
    right skin's budget would have caught.

    Returns 0.0 if the wing alone has already spent the whole budget
    somewhere in the band, which means no feature can ramp there at all.
    """
    s = stack.settings
    lim = np.tan(np.radians(s.max_overhang_deg))
    c = stack.contours
    worst = lim
    for k in range(0, len(c) - stride, stride):
        a, b = c[k], c[k + stride]
        rise = float(stack.z_mm[k + stride] - stack.z_mm[k])
        if rise <= 0.0:
            continue
        d = np.linalg.norm(b[:, None, :] - a[None, :, :], axis=2)
        dv = b - a[d.argmin(1)]
        # the skin in question, found by geometry: the LE is the vertex of
        # minimum x, the upper runs before it and the lower after
        i_le = int(np.argmin(b[:, 0]))
        rows = np.arange(0, i_le + 1) if side == "upper" else np.arange(i_le, len(b))
        skin = b[rows]
        lo_x, hi_x = b[:, 0].min(), b[:, 0].max()
        frac = (skin[:, 0] - lo_x) / max(hi_x - lo_x, 1e-9)
        sel = (frac >= min(x0, x1)) & (frac <= max(x0, x1))
        if not sel.any():
            continue
        dx = np.abs(dv[rows][sel, 0]) / rise
        avail = np.sqrt(np.maximum(lim * lim - dx * dx, 0.0))
        worst = min(worst, float(avail.min()))
    return float(max(worst, 0.0))

def cut_budget_profile(stack: LayerStack, x_abs0: float, x_abs1: float,
                       stride: int = 4, side: str = "upper", n_x: int = 9,
                       margin_frac: float = 0.02
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Depth a cut may fade per mm of Z, station by station, with the
    band CLIPPED to the section as `BaySpec.band` clips it.

    -> (z_mm, rate, valid, rise). `valid` is False where nothing of the
    band is inside the section, so no cut can exist; `rate` is 0 where
    the wing alone spends the whole budget; `rise` is the skin's own rise
    across the opening there, which the floor's deepest corner travels on
    top of the box depth. Inside the band the cost is the
    skin's vertical drift at fixed absolute stations (`skin_drift_abs`),
    subtracted linearly. At an edge the clipping has moved onto the
    skin, the wall's corner rides the skin itself, so its cost is the
    skin's own motion there -- the distance from the edge point to the
    bead below, which the overhang gate will measure the same way --
    also subtracted linearly, which is conservative against the true
    quadrature and is the honest side to err on."""
    lim = np.tan(np.radians(stack.settings.max_overhang_deg))
    c = stack.contours
    ox = stack.origin_mm[0]
    xa0, xa1 = x_abs0 - ox, x_abs1 - ox

    def skin(layer):
        i_le = int(np.argmin(layer[:, 0]))
        pts = layer[:i_le + 1][::-1] if side == "upper" else layer[i_le:]
        o = np.argsort(pts[:, 0])
        return pts[o]

    zs, rate, valid, rise_mm = [], [], [], []
    for k in range(0, len(c) - stride, stride):
        rise = float(stack.z_mm[k + stride] - stack.z_mm[k])
        if rise <= 0.0:
            continue
        sa, sb = skin(c[k]), skin(c[k + stride])
        x_le, x_te = float(sb[0, 0]), float(sb[-1, 0])
        chord = x_te - x_le
        # the SAME rule the cutter uses, so the budget never calls a
        # station unusable that `insert_detours` will happily cut
        f0, f1 = _bay_clamp_band(0.5 * (xa0 + xa1 - 2.0 * x_le) / max(chord, 1e-9),
                                 0.5 * (xa1 - xa0) / max(chord, 1e-9), chord)
        lo, hi = x_le + f0 * chord, x_le + f1 * chord
        zs.append(0.5 * float(stack.z_mm[k] + stack.z_mm[k + stride]))
        if hi - lo < 1e-6:
            rate.append(0.0)
            valid.append(False)
            rise_mm.append(0.0)
            continue
        xs = np.linspace(lo, hi, n_x)
        ya = np.interp(xs, sa[:, 0], sa[:, 1])
        yb = np.interp(xs, sb[:, 0], sb[:, 1])
        cost = float(np.abs(yb - ya).max()) / rise
        for x_edge, clipped in ((lo, lo > xa0 + 1e-9), (hi, hi < xa1 - 1e-9)):
            if clipped:
                p = np.array([[x_edge, float(np.interp(x_edge, sb[:, 0], sb[:, 1]))]])
                d = float(_dist_to_segments(p, c[k])[0])
                cost = max(cost, d / rise)
        rate.append(float(max(lim - cost, 0.0)))
        valid.append(True)
        # how far the floor's deepest corner travels BEYOND the box depth
        # at this station: the skin's own rise across the opening
        rise_mm.append(float(yb.max() - yb.min()))
    return (np.array(zs), np.array(rate), np.array(valid, dtype=bool),
            np.array(rise_mm))


def solve_ramp_from_rates(dist_mm: np.ndarray, rate: np.ndarray,
                          rise_mm: np.ndarray, depth_mm: float,
                          margin: float) -> float:
    """Shortest ramp that fades a bay of `depth_mm` against a profile.

    A ramp of length L runs at ONE linear rate, because `depth_frac` is
    linear, so every station it crosses must afford it. At station k the
    floor's deepest corner travels `depth_mm + rise_mm[k]`, so the ramp
    must be at least `(depth + rise[k]) / (rate[k] * margin)` long for
    each k it covers, and the answer is the running maximum of that. inf
    if no L in the profile satisfies its own stations."""
    if len(dist_mm) == 0:
        return float("inf")
    with np.errstate(divide="ignore", invalid="ignore"):
        need = (depth_mm + rise_mm) / np.maximum(rate * margin, 1e-12)
    worst = np.maximum.accumulate(need)
    ok = np.where(dist_mm >= worst)[0]
    if len(ok) == 0:
        return float("inf")
    return float(worst[int(ok[0])])

def ramp_span_mm(stack: LayerStack, depth_mm: float,
                 x0: float, x1: float, side: str = "upper") -> float:
    """Span needed to fade a feature `depth_mm` deep in or out. inf if it
    cannot be done in this panel at all."""
    rate = ramp_budget(stack, x0, x1, side=side)
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
    if stack.has_ribs or stack.has_cuts:
        # With ribs or cuts the upper/lower index pairing no longer
        # describes the section, so clearance is measured the general
        # way: every vertex against every non-adjacent segment. Sampled,
        # because it is O(n^2) per layer and the geometry varies smoothly
        # with Z.
        step = max(len(stack.contours) // 40, 1)
        t_min = min(min_clearance_mm(c, skip=8)
                    for c in stack.contours[::step])
        detail = ("general contour clearance"
                  + (f", {stack.settings.rib_count} ribs" if stack.has_ribs else "")
                  + (", cut" if stack.has_cuts else ""))
    else:
        t_min = float(stack.wall_separation_mm().min())
        detail = "=> single simple loop per layer"
    gates.append(Gate("min wall separation", t_min >= s.min_wall_mm, t_min,
                      s.min_wall_mm, "mm", detail))

    if stack.role == "wing":
        spar, spar_z = spar_fit(stack)
        gates.append(Gate("spar bore", spar >= s.spar_d_mm, spar, s.spar_d_mm,
                          "mm", f"pinch at z={spar_z:.0f} mm"))

    # Consecutive vertices, which `min wall separation` cannot see: it
    # skips neighbours within 8 indices, exactly so that a contour is not
    # judged against its own next point. A coincident PAIR is invisible
    # to it and fatal to the mesher -- a zero-length edge stalls the
    # ear-clipper into an invalid fan, and six non-manifold edges came
    # out of `stl.export`, which is the only thing in the program that
    # was checking. One array op per layer.
    step = max(len(stack.contours) // 40, 1)
    gap = min(float(np.linalg.norm(
        np.roll(c, -1, axis=0) - c, axis=1).min())
        for c in stack.contours[::step])
    floor_mm = 1e-4
    gates.append(Gate("vertex spacing", gap >= floor_mm, gap, floor_mm, "mm",
                      "no zero-length edge may reach the mesher"))

    area0 = 0.5 * abs(np.dot(stack.contours[0][:, 0],
                             np.roll(stack.contours[0][:, 1], -1))
                      - np.dot(stack.contours[0][:, 1],
                               np.roll(stack.contours[0][:, 0], -1)))
    small = stack.role in ("elevon", "lid")
    floor = s.elevon_first_layer_mm2 if small else s.min_first_layer_mm2
    gates.append(Gate("first-layer area", area0 >= floor, area0, floor, "mm2",
                      "bed adhesion" + (" (brim)" if small else "")))
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


@dataclass(frozen=True)
class BayCut:
    """One opening to cut, in planform coordinates.

    `x0`/`x1` are root-chord fractions and give the CENTRE; `length_mm`,
    when set, is the opening's true chordwise extent and is applied at
    every layer's own chord. Both ramp lengths are ALREADY measured by
    the caller on bare panels -- the ramp-in against the panels it
    crosses inboard of eta0, the ramp-out against those outboard of eta1
    -- and the dead spans are panels with no overhang budget, which the
    bay crosses fully open before its ramp begins."""

    name: str
    x0: float
    x1: float
    eta0: float
    eta1: float
    depth_mm: float
    ramp_out_mm: float
    ramp_in_mm: float
    open_from: str = "upper"
    ledge_mm: float = 0.0
    dead_out_mm: float = 0.0
    dead_in_mm: float = 0.0
    length_mm: float | None = None
    x_abs_mm: float | None = None


def build_panels(plan: Planform, settings: PrintSettings,
                 z_margin_mm: float = 8.0,
                 z_step_mm: float | None = None,
                 bays=()) -> list[LayerStack]:
    """Every printable part of the right half wing, root outboard.

    `bays` are `BayCut`s (a plain tuple in the same field order is
    accepted) for the openings to cut. Their ramps are measured by the
    caller: measuring here would be circular, because the ramp's own dive
    and climb walls move in Z, so a budget measured on a panel that
    already has the cut comes back as zero and the ramp as infinite.

    **The ramp may run on past a print joint.** What must not straddle a
    joint is the BOX -- half a battery in each shell is not a thing --
    and that is gated separately. The taper that closes the opening is
    just geometry, and both panels are lofted from the same planform, so
    the contours match across the joint by construction. Requiring the
    ramp to finish inside the panel holding the box is a constraint
    nothing physical asks for, and it is what made micro's centre body
    impossible: 24.5 mm of panel, a pack reaching 15 mm up it, 10 mm left
    and 22 mm needed. Given the whole span outboard to close in, it has
    room.

    Each panel therefore receives the cut if the bay's influence -- its
    ramp in, its full-depth run and its ramp out -- reaches into that
    panel, with both stations rebased into the panel's own z. A panel the
    bay is already tapering through gets stations below z = 0, which
    `depth_frac` reads as "part way along the profile at z = 0".

    A bay at the centreline starts at the root face and ramps once. A bay
    out in the wing -- a servo pocket -- starts at its own eta0 and ramps
    IN as well as out; treating it as a root bay cut it into the centre
    body at full depth.
    """
    spans = panel_etas(plan, settings, z_margin_mm)
    bays = [b if isinstance(b, BayCut) else BayCut(*b) for b in bays]
    out = []
    for i, (a, b) in enumerate(spans):
        cuts = []
        base_mm = arc_length_mm(plan, 0.0, a)      # this panel's z origin
        top_mm = arc_length_mm(plan, 0.0, b)
        for c in bays:
            if not (np.isfinite(c.ramp_out_mm) and np.isfinite(c.ramp_in_mm)):
                continue                       # cannot be opened/closed: no cut
            root = c.eta0 <= 1e-9
            # the full-depth run extends across any budget-less panels
            start_mm = (0.0 if root
                        else arc_length_mm(plan, 0.0, c.eta0) - c.dead_in_mm)
            full_mm = arc_length_mm(plan, 0.0, c.eta1) + c.dead_out_mm
            first_mm = 0.0 if root else start_mm - c.ramp_in_mm
            if base_mm >= full_mm + c.ramp_out_mm - 1e-9:
                continue                       # closed before this panel
            if top_mm <= first_mm + 1e-9:
                continue                       # not yet begun in this panel
            cuts.append(BaySpec(c.name, c.x0, c.x1,
                                start_mm - base_mm, full_mm - base_mm,
                                c.ramp_out_mm, c.depth_mm,
                                settings.extrusion_width_mm,
                                open_from=c.open_from, ledge_mm=c.ledge_mm,
                                ramp_in_mm=0.0 if root else c.ramp_in_mm,
                                length_mm=c.length_mm, x_abs_mm=c.x_abs_mm))
        out.append(build_stack(plan, settings, a, b,
                               name=f"{plan.name}_p{i}",
                               z_step_mm=z_step_mm, bay_cuts=tuple(cuts)))
    return out
