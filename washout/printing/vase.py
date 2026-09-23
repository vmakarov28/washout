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
from .detours import insert_detours
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
    spar_lines: tuple = ()
    """Every fitted spar as the straight tube it is, flight millimetres:
    (root_x, root_z, dx/dy, dz/dy, reach_y, rib_half_width, diameter).

    When present this REPLACES `spar_avoid` and `spar_corridors`, which
    describe a tube at a fixed chord fraction -- the tube that bent with
    the wing (ROADMAP-CAD.md section 0.1). From the line, each panel keeps
    its truss out of the chord band the tube actually sweeps through
    inside that panel, and the bore gate measures at the tube's actual
    centre in each layer, and only in the layers it reaches."""
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
    origin_mm: tuple = (0.0, 0.0)
    """The planform point (mm aft of the root LE, mm above the chord
    line) that sits at the part's printer-frame origin. Parts are
    centred on the bed."""
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
    spar_xy: np.ndarray | None = None
    """(tubes, layers, 2): each straight tube's centre in each layer, print
    X/Y as on the bed; NaN in the layers it does not reach. Filled from
    `settings.spar_lines`, and when it is, the bore gate measures here."""
    spar_ellipse: tuple = ()
    """Per tube, (ux, uy, cos_gamma, diameter): a tube that crosses the
    layer plane at an angle gamma cuts it in an ellipse, 1/cos(gamma)
    longer along the in-plane direction (ux, uy)."""
    role: str = "wing"
    """"wing" or "elevon". Not decoration: two of the printability gates
    are wing-panel rules and are meaningless on a control surface. An
    elevon has no spar, so asking whether an 8 mm tube fits its 6 mm
    section rejects a part that was never meant to hold one; and the
    300 mm2 bed-adhesion floor is sized for a 111 mm tall centre body,
    not for a 25 mm wide trailing-edge wedge."""
    frame: object | None = None
    """This part's `frames.PanelFrame`: where it sits in the aircraft.

    Print X is flight x less `origin_mm[0]`; print Y is height square to
    the panel axis less `origin_mm[1]`; print Z is distance along the
    axis. `frame.to_flight` undoes it, which is how the build sheet
    reports the joints and how the CAD export and the conformance gate
    put a printed part back where it flies."""

    def to_flight(self, X, Y, Z) -> np.ndarray:
        """Print coordinates, as they are on the bed -> flight mm."""
        ox, oy = self.origin_mm
        return self.frame.to_flight(np.asarray(X) + ox, np.asarray(Y) + oy, Z)

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
        sep = np.linalg.norm(up - lo, axis=2)[:, 1:]
        # The round NOSE is not two walls. Pairs within a few beads of the
        # leading edge, measured ALONG the loop, are one bead turning round
        # it, and their straight-line distance says nothing about whether
        # two walls weld: at 0.04 mm aft of the vertex it is 0.3 mm on
        # every section, by geometry. The nose used to pass only because
        # the nozzle floor blunted it into a 1 mm flat (thicken_for_nozzle).
        def along(skin):
            seg = np.linalg.norm(np.diff(skin, axis=1), axis=2)
            return np.cumsum(seg, axis=1)
        path = along(up) + along(lo)
        return np.where(path < 4.0 * self.settings.extrusion_width_mm,
                        np.inf, sep)

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
    min_te_mm: float | None = None,
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

    `min_te_mm` overrides the setting for one section. A layer is cut
    square to a tilted panel, which is thinner than the `y = const`
    section it is carried from by cos(delta - phi) / cos(delta), so the
    floor is raised by that factor here for the printed edge to keep it.
    """
    n = (len(loop_unit) + 1) // 2
    upper = loop_unit[:n][::-1]                 # LE -> TE
    lower = loop_unit[n - 1:]                   # LE -> TE
    x = upper[:, 0]
    cam = 0.5 * (upper[:, 1] + lower[:, 1])
    t = upper[:, 1] - lower[:, 1]

    t_min = (settings.min_te_mm if min_te_mm is None else min_te_mm) / chord_mm
    eps = max(settings.te_blend / chord_mm, 1e-6)
    t_floor = 0.5 * (t + t_min + np.sqrt((t - t_min) ** 2 + eps**2))
    # The TRAILING edge only, blended in over 0.35c-0.65c where the section
    # is many beads thick and the floor changes nothing. Applied along the
    # whole chord, it also raised the NOSE -- where the vertical thickness
    # is zero by construction, because the nose is round -- to the floor:
    # the shared leading-edge vertex went to +0.5 mm and the next point
    # below it to -0.47 mm, 0.04 mm aft. Every printed section carried a
    # 1 mm vertical flat for a nose, and the CAD export could not fit one
    # smooth skin through it without a seam at the leading edge.
    w = np.clip((x - 0.35) / 0.30, 0.0, 1.0)
    w = w * w * (3.0 - 2.0 * w)
    t_new = t + w * (t_floor - t)

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
    frame=None,
) -> LayerStack:
    """Slice a spanwise panel of the planform into printable layers.

    Print Z is distance along the panel's AXIS, and each layer is the
    loft cut square to that axis -- see `frames.py`. The panel follows
    the dihedral curve inside itself as a lean in print Y, so a panel
    printed straight up the Z axis is the loft, not a straightened copy of
    it; only the joints between panels are straight lines, and they are
    reported as the wedges they leave.

    `frame` is this panel's `frames.PanelFrame`. `build_panels` solves one
    for every panel together, because a joint's pivot depends on both
    sides of it; without one, the panel gets the frame it would have alone
    (horizontal at the centreline, chord line to chord line elsewhere).
    """
    from . import frames as fr
    if frame is None:
        frame = fr.standalone(plan, eta0, eta1, settings)
    panel_len_mm = float(frame.length_mm)

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

    tubes = _tubes_in_panel(plan, frame, z, settings) if settings.spar_lines else None
    avoid = (tuple(b for b in (t["band"] for t in tubes) if b is not None)
             if tubes is not None else tuple(settings.spar_avoid))
    rib_spec = RibSpec(n_ribs=settings.rib_count,
                       pitch_mm=settings.rib_pitch_mm,
                       max_overhang_deg=settings.max_overhang_deg,
                       enabled=settings.ribs,
                       avoid=avoid)
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
        # The panel's LEAN -- the dihedral curve's departure from the
        # panel axis -- moves every point in print Y as Z rises, and that
        # motion is square to the ribs' chordwise sweep, so it comes off
        # the budget in QUADRATURE (ROADMAP-BUILD.md section 0), not by
        # subtraction.
        lean = fr.max_lean_rate(plan, frame)
        allowed_x = float(np.sqrt(max(allowed * allowed - lean * lean, 0.0)))
        # The analytic bound is a LOWER bound: it tracks the leading and
        # trailing edges but not twist, thickness change or the trailing-
        # edge thickening, all of which also move contour points. Measured
        # against the real overhang across nine panels it runs 1.21x to
        # 1.82x low, so 2.0 is the factor that keeps it conservative. The
        # cost is a shallower truss, which is the right way to be wrong.
        rib_spec = replace(rib_spec,
                           rate_mm_per_mm=max(allowed_x - 2.0 * used, 0.02))
    # How many ribs this panel can actually take. Per panel, because the
    # spar corridors it must dodge are per panel -- and the point budget
    # follows it, so the invariant holds within a stack without
    # pretending every panel has the same truss.
    n_rib_eff = ribs_that_fit(
        rib_spec, plan.at(0.5 * (eta0 + eta1)).chord_m * 1000.0,
        settings.extrusion_width_mm * settings.rib_clearance_factor)
    # Solved ONCE, at the panel's mid station: the gaps move with the
    # local chord and the share between them is an integer, so a layout
    # recomputed per layer made ribs hop between gaps from one layer to
    # the next.
    rib_spec = replace(rib_spec, n_ribs=n_rib_eff).with_layout(
        plan.at(0.5 * (eta0 + eta1)).chord_m * 1000.0)
    clr = settings.extrusion_width_mm * settings.rib_clearance_factor
    if truncated:
        from .elevons import hinge_x, truncate_loop
        xh_cut = hinge_x(settings)

    def unit_loop(st, chord_mm, s_mm, te_scale):
        loop = st.airfoil.coords(settings.contour_points)
        loop = thicken_for_nozzle(loop, chord_mm, settings,
                                  min_te_mm=settings.min_te_mm * te_scale)
        if truncated:
            loop = truncate_loop(loop, xh_cut)
        if n_rib_eff > 0:
            # ONE pass for every slit on the section. Inserting them one
            # at a time re-interpolated a skin that already had vertical
            # walls in it, which put a vertex 7.2 mm down a wall between
            # two adjacent layers.
            # The floor's gap to the lower skin is a THICKNESS, and the cut
            # square to a tilted panel shrinks every thickness by the same
            # factor the trailing edge is raised against -- which took the
            # trainer's tip-panel floor from 0.50 mm to 0.43, under a bead.
            # The slit's width runs along the chord and is not touched.
            loop = insert_detours(loop, chord_mm, s_mm, rib_spec,
                                  slit_mm=clr, gap_mm=clr * te_scale,
                                  min_groove_mm=settings.extrusion_width_mm)
        return loop

    contours, eta = fr.slice_layers(plan, frame, z, unit_loop)

    # The solved corridors, in this loop's own chord. A truncated panel's
    # loop spans [0, x_hinge] of the original chord, so a spar at 0.54c
    # sits at 0.54/x_hinge of what remains.
    xh = (1.0 - settings.elevon_chord) if truncated else 1.0
    local = tuple(float(np.clip(c / max(xh, 1e-6), 0.0, 1.0))
                  for c, reach in settings.spar_corridors
                  if reach >= eta0 + 1e-9)

    # centre the whole part on the bed
    flat = contours.reshape(-1, 2)
    origin = 0.5 * (flat.min(0) + flat.max(0))
    contours -= origin
    spar_xy, ellipse = None, ()
    if tubes is not None:
        local = ()
        if tubes:
            spar_xy = np.stack([t["xy"] for t in tubes]) - origin
            ellipse = tuple(t["ellipse"] for t in tubes)
    return LayerStack(z_mm=z, eta=eta, contours=contours,
                      settings=settings, name=name, z_step_mm=step,
                      has_ribs=n_rib_eff > 0,
                      origin_mm=(float(origin[0]), float(origin[1])),
                      spar_x_local=local, frame=frame,
                      spar_xy=spar_xy, spar_ellipse=ellipse)


def _tubes_in_panel(plan, frame, z_mm, settings) -> list:
    """Where each straight spar crosses each layer of one panel.

    The tube's axis is (x0 + sx y, y, z0 + sz y); the layer at print
    height s is the plane (y - y_p) cos(phi) + (z - z_p) sin(phi) = s, so
    the crossing is at

        y = (s + y_p cos(phi) + (z_p - z0) sin(phi)) / (cos(phi) + sz sin(phi))

    and a tube exists in the layer while 0 <= y <= its reach. From the
    crossings: the print X/Y of its centre per layer, the chord band it
    sweeps through in this panel -- the truss has to keep out of all of
    it, since the rib layout is solved once per panel -- and the ellipse
    it cuts in a layer, from the angle between the tube and the panel
    axis."""
    from .frames import reference_etas
    out = []
    ca, sa = frame.cos, frame.sin
    ay, az = frame.origin_yz_mm
    etas = reference_etas(plan, frame, z_mm)
    st = [plan.at(float(np.clip(e, 0.0, 1.0))) for e in etas]
    x_le = np.array([s_.x_le_m for s_ in st]) * 1000.0
    chord = np.array([s_.chord_m for s_ in st]) * 1000.0
    for (x0, z0, sx, sz, reach_y, half, d) in settings.spar_lines:
        y = (np.asarray(z_mm) + ay * ca + (az - z0) * sa) / (ca + sz * sa)
        x, zz = x0 + sx * y, z0 + sz * y
        X = x
        Y = -(y - ay) * sa + (zz - az) * ca
        here = (y >= -1e-9) & (y <= reach_y + 1e-9)
        xy = np.where(here[:, None], np.stack([X, Y], 1), np.nan)
        band = None
        if here.any():
            f = (x[here] - x_le[here]) / chord[here]
            c_mid = float(np.median(chord[here]))
            band = (float(0.5 * (f.min() + f.max())),
                    float(half + 0.5 * (f.max() - f.min()) * c_mid))
        # the tube's direction in print coordinates, and its angle to Z
        dX, dY, dS = sx, -sa + sz * ca, ca + sz * sa
        n_in = float(np.hypot(dX, dY))
        cos_g = float(dS / np.sqrt(dX * dX + dY * dY + dS * dS))
        ux, uy = ((dX / n_in, dY / n_in) if n_in > 1e-12 else (1.0, 0.0))
        out.append({"xy": xy, "band": band,
                    "ellipse": (float(ux), float(uy), cos_g, float(d))})
    return out


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


def check_insert(ins, settings: PrintSettings) -> Printability:
    """A joint wedge insert (printing/inserts.py), printed flat on its face
    A in normal mode. Four gates, each with its number:

      * it fits the bed, at the orientation it is drawn in;
      * the mesh is watertight -- a slicer fills what it cannot close;
      * its thinnest printed edge is two beads, where the knife edge at
        the pivot was cut off (the crest left is glue, and reported);
      * every tube bore has a bead of material round it on both faces --
        a bore that breaks out of the insert's outline reads negative."""
    from .stl import manifold_report
    out = Printability()
    if getattr(ins, "glue_fill", False):
        # nothing prints: the wedge is filled (see JointInsert.glue_fill)
        out.gates.append(Gate("wedge is a glue fill", True, ins.max_gap_mm, 0.0,
                              "mm", f"a tube severs it; {ins.fill_g():.1f} g of filler"))
        return out
    size = ins.size_mm()
    foot = float(max(size[0], size[1]))
    lim = float(min(settings.bed_x_mm, settings.bed_y_mm))
    out.gates.append(Gate("insert fits the bed", foot <= lim and size[2] <= settings.bed_z_mm,
                          foot, lim, "mm", f"{size[0]:.0f} x {size[1]:.0f} x {size[2]:.1f} mm"))
    tight = manifold_report(ins.tris)["watertight"]
    out.gates.append(Gate("insert watertight", bool(tight), float(tight), 1.0, "",
                          f"{len(ins.tris)} triangles"))
    beads = 2.0 * settings.extrusion_width_mm
    out.gates.append(Gate("insert thinnest edge", ins.crest_mm >= beads - 1e-9,
                          ins.crest_mm, beads, "mm",
                          "cut square where the wedge thins; the crest is glue"))
    for b in ins.bores:
        if b.get("kind") == "notch":
            # the tube sits against the skin, so the insert is cut round it;
            # what matters is the material left between notch and far skin
            out.gates.append(Gate(f"insert neck at notch ({b['name']})",
                                  b["neck_mm"] >= 2.0 * settings.extrusion_width_mm,
                                  b["neck_mm"], 2.0 * settings.extrusion_width_mm, "mm",
                                  f"{b['d_mm']:.0f} mm tube against the skin: notched"))
        else:
            out.gates.append(Gate(f"insert bore wall ({b['name']})",
                                  b["wall_mm"] >= settings.extrusion_width_mm,
                                  b["wall_mm"], settings.extrusion_width_mm, "mm",
                                  f"{b['d_mm']:.0f} mm tube"))
    return out


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
    c = stack.contours
    ks = np.arange(0, len(c) - stride, stride)
    if ks.size == 0:
        return 0.0, 0.0
    # rise is the ACTUAL Z between the two sampled contours, not the
    # nominal layer height: when the search samples coarsely those differ
    # by 8x, and using the nominal value reports 82 deg of overhang on a
    # wing that really has 42.
    rise = np.maximum(stack.z_mm[ks + stride] - stack.z_mm[ks], 1e-9)
    # EXACT, and pruned. The full vertex-to-segment search is an (N, N)
    # matrix per pair of layers and was a quarter of a design evaluation.
    # The worst point is almost always near the segments of the same
    # index on the layer below -- the point count is constant and the
    # contour moves smoothly -- so the distance to that window of
    # segments is an UPPER bound on each point's true distance, computed
    # for every pair at once. Pairs whose bound cannot beat the worst
    # angle found so far are skipped, and within a pair only the points
    # whose bound can still beat it get the full search. The answer is the
    # one the full search gives; `test_overhang_pruning_is_exact` pins it.
    upper = _window_bound(c, ks, stride) / rise[:, None]      # tan, (P, N)
    order = np.argsort(-upper.max(1), kind="stable")
    worst, worst_k, best_tan = 0.0, int(ks[0]), -1.0
    for p in order:
        bound = upper[p]
        if bound.max() < best_tan:
            break                       # no remaining pair can beat it
        k = int(ks[p])
        cand = np.flatnonzero(bound >= best_tan)
        cand = cand[np.argsort(-bound[cand], kind="stable")]
        d_max = -1.0
        for chunk in np.array_split(cand, max(len(cand) // 32, 1)):
            if bound[chunk[0]] < max(best_tan, d_max / rise[p]):
                break                   # nothing left here can raise it
            d_max = max(d_max, float(_dist_to_segments(c[k + stride][chunk],
                                                       c[k]).max()))
        if d_max < 0.0:
            continue
        # the angle exactly as the unpruned loop computed it, and its tie
        # rule: the LOWEST layer wins among equal angles
        ang = float(np.degrees(np.arctan2(d_max, rise[p])))
        if ang > worst or (ang == worst and ang > 0.0 and k < worst_k):
            worst, worst_k = ang, k
        best_tan = max(best_tan, d_max / rise[p])
    return worst, (float(stack.z_mm[worst_k]) if worst > 0.0 else 0.0)


def _window_bound(c: np.ndarray, ks: np.ndarray, stride: int,
                  w: int = 3) -> np.ndarray:
    """Distance from each vertex of layer k+stride to the nearest of the
    2w+1 segments of layer k around the same index -> (len(ks), N).

    An upper bound on the vertex-to-segment distance `overhang_deg`
    wants, because it is a minimum over a subset of the segments."""
    n = c.shape[1]
    a = c[ks]                                   # (P, N, 2) the layer below
    ab = np.roll(a, -1, axis=1) - a
    denom = np.maximum(np.einsum("pij,pij->pi", ab, ab), 1e-12)
    pts = c[ks + stride]                        # (P, N, 2)
    best = np.full(pts.shape[:2], np.inf)
    idx = np.arange(n)
    for off in range(-w, w + 1):
        j = (idx + off) % n
        aj, abj, dj = a[:, j], ab[:, j], denom[:, j]
        ap = pts - aj
        t = np.clip(np.einsum("pij,pij->pi", ap, abj) / dj, 0.0, 1.0)
        d = np.linalg.norm(ap - t[..., None] * abj, axis=2)
        best = np.minimum(best, d)
    return best


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

def ramp_span_mm(stack: LayerStack, depth_mm: float,
                 x0: float, x1: float, side: str = "upper") -> float:
    """Span needed to fade a feature `depth_mm` deep in or out. inf if it
    cannot be done in this panel at all."""
    rate = ramp_budget(stack, x0, x1, side=side)
    return float(depth_mm / rate) if rate > 0.0 else float("inf")


def _inscribed_gap(contour: np.ndarray, x: float, n: int = 17,
                   only_mid: bool = False) -> float:
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

    `only_mid` scores the middle centre alone. Its clearance is one of the
    fifteen the full scan maximises over, so it is a LOWER bound on the
    answer, at a fifteenth of the cost -- which is what `spar_fit` prunes
    with.
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
    if only_mid:
        cy = cy[len(cy) // 2:len(cy) // 2 + 1]
    c = np.stack([np.full(cy.shape, x), cy], 1)            # (m, 2)

    ap = c[:, None, :] - a[None, :, :]                      # (m, N, 2)
    tt = np.clip(np.einsum("mnk,nk->mn", ap, ab) / denom[None, :], 0.0, 1.0)
    closest = a[None, :, :] + tt[:, :, None] * ab[None, :, :]
    d = np.linalg.norm(c[:, None, :] - closest, axis=2).min(1)   # (m,)
    return float(2.0 * d.max())


def _mid_gaps(contours: np.ndarray, xs: np.ndarray, n: int = 17) -> np.ndarray:
    """`_inscribed_gap(contours[k], xs[k], only_mid=True)` for every layer
    at once -> (L,).

    The middle centre is placed exactly as `np.linspace` places it --
    start plus index times step -- so it is one of the centres the full
    scan scores, and the bound is a true lower bound rather than one a
    rounding away from it."""
    x0, y0 = contours[:, :, 0], contours[:, :, 1]
    x1, y1 = np.roll(x0, -1, axis=1), np.roll(y0, -1, axis=1)
    x = xs[:, None]
    hit = (((x0 - x) * (x1 - x)) <= 0.0) & (np.abs(x1 - x0) > 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        tt = (x - x0) / (x1 - x0)
    ys = y0 + tt * (y1 - y0)
    lo = np.where(hit, ys, np.inf).min(1)
    hi = np.where(hit, ys, -np.inf).max(1)
    ok = hit.sum(1) >= 2
    mid = (n - 1) // 2
    cy = mid * ((hi - lo) / (n - 1)) + lo
    a = contours
    ab = np.roll(a, -1, axis=1) - a
    denom = np.einsum("lij,lij->li", ab, ab)
    denom[denom < 1e-12] = 1e-12
    cpt = np.stack([xs, cy], 1)[:, None, :]                 # (L, 1, 2)
    ap = cpt - a
    t = np.clip(np.einsum("lij,lij->li", ap, ab) / denom, 0.0, 1.0)
    d = np.linalg.norm(cpt - (a + t[..., None] * ab), axis=2).min(1)
    return np.where(ok, 2.0 * d, 0.0)


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
    if stack.spar_xy is not None:
        return _tube_bore(stack)
    fracs = stack.spar_x_local or (s.spar_x_frac,)
    # the inscribed circle already stops one bead inside the contour's
    # centreline, so only the fit clearance is deducted.
    #
    # Pruned, and exact: the full scan tries 15 centres per layer, and any
    # ONE of them -- the middle -- gives a LOWER bound on the diameter.
    # Layers are visited in order of that bound and the scan stops once
    # no remaining layer's bound can undercut the pinch already found.
    # `test_bore_pruning_is_exact` pins the answer to the full scan's,
    # including its tie rule (the lowest layer wins).
    c = stack.contours
    lo_x, hi_x = c[:, :, 0].min(1), c[:, :, 0].max(1)
    rows = []
    for f in fracs:
        xs = lo_x + float(f) * (hi_x - lo_x)
        lbs = _mid_gaps(c, xs) - s.extrusion_width_mm - s.spar_clearance_mm
        rows += [(float(lb), k, float(x))
                 for k, (lb, x) in enumerate(zip(lbs, xs))]
    best, best_k = np.inf, None
    for lb, k, x in sorted(rows, key=lambda r: (r[0], r[1])):
        if lb > best or (lb == best and best_k is not None and k > best_k):
            break
        # subtracted in the same order the full scan did, so the answer is
        # the same float and not one rounding away from it
        gap = (_inscribed_gap(stack.contours[k], x)
               - s.extrusion_width_mm - s.spar_clearance_mm)
        if gap < best or (gap == best and best_k is not None and k < best_k):
            best, best_k = float(gap), k
    return best, (float(stack.z_mm[best_k]) if best_k is not None else 0.0)


def _tube_bore(stack: LayerStack) -> tuple[float, float]:
    """The bore at each straight tube's actual centre, in the layers it
    reaches: the usable diameter is twice the distance from the centre
    to the nearest contour segment, less a bead and the fit clearance --
    the inscribed-circle rule, asked at the one centre that matters.

    The tube crosses a layer at an angle, so its section is an ellipse,
    1/cos(gamma) longer along its in-plane direction. Compressing the
    contour along that direction by cos(gamma), about the centre, turns
    the ellipse into the circle and leaves the question a distance.
    -> (smallest margin-adjusted diameter, the Z where it is)."""
    s = stack.settings
    best, best_z = np.inf, 0.0
    c = stack.contours
    for xy, (ux, uy, cos_g, d) in zip(stack.spar_xy, stack.spar_ellipse):
        here = np.flatnonzero(np.isfinite(xy[:, 0]))
        if here.size == 0:
            continue
        rel = c[here] - xy[here][:, None, :]
        along = rel[..., 0] * ux + rel[..., 1] * uy
        rel = rel - (1.0 - cos_g) * along[..., None] * np.array([ux, uy])
        a = rel
        ab = np.roll(a, -1, axis=1) - a
        den = np.maximum(np.einsum("lij,lij->li", ab, ab), 1e-12)
        t = np.clip(-np.einsum("lij,lij->li", a, ab) / den, 0.0, 1.0)
        dist = np.linalg.norm(a + t[..., None] * ab, axis=2).min(1)
        # against `spar_d_mm`, the tube the structure sized, which is what
        # `check` compares it with -- as the chord-fraction gate did
        gap = 2.0 * dist - s.extrusion_width_mm - s.spar_clearance_mm
        k = int(np.argmin(gap))
        if gap[k] < best:
            best, best_z = float(gap[k]), float(stack.z_mm[here[k]])
    return best, best_z


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
        # With ribs the upper/lower index pairing no longer
        # describes the section, so clearance is measured the general
        # way: every vertex against every non-adjacent segment. Sampled,
        # because it is O(n^2) per layer and the geometry varies smoothly
        # with Z.
        step = max(len(stack.contours) // 40, 1)
        t_min = min(min_clearance_mm(c, skip=8,
                                     min_path_mm=4.0 * s.extrusion_width_mm)
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

    Two kinds of break, and the difference between them is the whole
    function.

    MANDATORY: the root, the tip, and the ELEVON's root station. The
    wing's trailing edge has to stop where the elevon begins, and a
    surface normal to the span is a roof in this print orientation, which
    spiralize cannot build. Breaking the print there makes every panel
    wholly plain or wholly truncated and turns that roof into a joint.

    OPTIONAL: the planform's own control stations. These are where the
    loft's curvature is DESCRIBED, which is not the same as where the
    aircraft has to come apart -- and breaking at all of them regardless
    cut a 480 mm wing into four panels a side when every one of them was
    under a third of the envelope. A joint is not free: two more faces to
    bond, a step for the air to find, and mass. So a control station is
    used only where a span is genuinely too tall, and the one nearest the
    middle is preferred, because an even split leaves the most room on
    both sides.

    Anything still over the envelope is divided into equal pieces --
    equal, not greedy-fill, so the joints stay symmetric and one spar
    length serves them all."""
    limit = settings.bed_z_mm - z_margin_mm
    mandatory = {0.0, 1.0}
    if 0.0 < settings.elevon_eta < 1.0 and settings.elevon_chord > 1e-6:
        mandatory.add(float(settings.elevon_eta))
    optional = [float(e) for e in sorted(set(plan.controls)) if 0.0 < e < 1.0]
    out: list[tuple[float, float]] = []
    edges = sorted(mandatory)
    for a, b in zip(edges, edges[1:]):
        out.extend(_fit_span(plan, a, b, optional, limit, min_panel_mm))
    return out


def _fit_span(plan: Planform, a: float, b: float, optional, limit: float,
              min_panel_mm: float) -> list[tuple[float, float]]:
    """One mandatory span, divided only as far as the envelope demands."""
    if arc_length_mm(plan, a, b) <= limit:
        return [(a, b)]
    # A control station inside it, nearest the middle, that leaves no
    # sliver either side. A sliver is a part in the parts list and a
    # joint in the assembly that cannot exist: micro's hinge station and
    # a control station once sat 0.2 mm apart and produced a panel 0.2 mm
    # tall weighing 0.0 g.
    mid = 0.5 * (a + b)
    inside = [e for e in optional if a < e < b
              and arc_length_mm(plan, a, e) >= min_panel_mm
              and arc_length_mm(plan, e, b) >= min_panel_mm]
    if inside:
        cut = min(inside, key=lambda e: abs(e - mid))
        return (_fit_span(plan, a, cut, optional, limit, min_panel_mm)
                + _fit_span(plan, cut, b, optional, limit, min_panel_mm))
    n = max(int(np.ceil(arc_length_mm(plan, a, b) / limit)), 1)
    cuts = np.linspace(a, b, n + 1)
    return [(float(u), float(v)) for u, v in zip(cuts, cuts[1:])]



def build_panels(plan: Planform, settings: PrintSettings,
                 z_margin_mm: float = 8.0,
                 z_step_mm: float | None = None) -> list[LayerStack]:
    """Every printable part of the right half wing, root outboard.

    Each panel is an open-ended tube: one contour per layer, ribs welded
    across it, and nothing cut through the skin. The wing is loaded
    through those open ends before the panels are bonded, which is why
    there is no opening to place and no ramp to budget."""
    from .frames import panel_frames
    spans = panel_etas(plan, settings, z_margin_mm)
    frames = panel_frames(plan, spans, settings)
    return [build_stack(plan, settings, a, b, name=f"{plan.name}_p{i}",
                        z_step_mm=z_step_mm, frame=f)
            for i, ((a, b), f) in enumerate(zip(spans, frames))]
