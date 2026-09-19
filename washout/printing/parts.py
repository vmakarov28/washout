"""Companion parts: what one spiral cannot be, generated to the shell as cut.

A vase-mode panel is a single closed contour per layer. Some of what an
aeroplane needs is not that: a hatch lid has two faces, a control horn
has a hole through it, a motor mount has five. Those are separate
printed parts, and the rule for them is the same as for everything else
in the program -- every dimension comes from the geometry that was
scored, and every claim about fit is a gate with a number.

Two kinds of geometry live here.

  * A **flat plate with holes**, printed in normal mode lying on the
    bed: the horn, the motor mount's web. Triangulated by Delaunay with
    boundary recovery -- any boundary edge the triangulation missed is
    split at its midpoint and the triangulation repeated, which
    terminates because every split halves an edge -- then the triangles
    outside the outline or inside a hole are dropped by centroid.
  * A **swept prism**: a closed section carried along an axis, one
    section per station, skinned exactly as `stl.skin` skins a panel.
    The motor mount's flanges are this, and they FUSE to the web by
    sharing its cap vertices: the web's forward face is triangulated
    with the flange roots as holes and the flanges start on those rings,
    so the union is watertight by construction and no boolean is needed.

The hatch lid is neither. It follows the wing's own skin over the bay,
so it is a vase-style stack built from the same station loops as the
panel and lives in `hatch_lid` below; it prints on its edge, span up,
exactly like the wing, as a two-bead lens.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import Delaunay

from .. import linkage as _lkg
from . import stl as _stl
from .bays import BaySpec, UPPER, _detour, split_skins
from .vase import LayerStack, PrintSettings, arc_length_mm, thicken_for_nozzle

SOLID_PLA_GCC = 1.24
"""Density the companion parts are weighed at. They print in ordinary
PLA, not the foamed kind the shell uses: a motor mount in foamed PLA is
a motor mount that pulls its bolts out. Declared, as every material
number in the project is."""

BEAD_HALF_MM = 0.225
"""Half an extrusion width at the fleet's 0.45 mm bead: how far the
printed surface stands outside the contour the toolpath follows."""


# ---------------------------------------------------------------- polygons


def signed_area(poly: np.ndarray) -> float:
    p = np.asarray(poly, dtype=float)
    q = np.roll(p, -1, axis=0)
    return 0.5 * float((p[:, 0] * q[:, 1] - q[:, 0] * p[:, 1]).sum())


def ccw(poly: np.ndarray) -> np.ndarray:
    p = np.asarray(poly, dtype=float)
    return p if signed_area(p) > 0.0 else p[::-1].copy()


def cw(poly: np.ndarray) -> np.ndarray:
    return ccw(poly)[::-1].copy()


def points_in_polygon(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Ray casting, vectorised over `pts`."""
    p = np.asarray(pts, dtype=float)
    a = np.asarray(poly, dtype=float)
    b = np.roll(a, -1, axis=0)
    x, y = p[:, 0][:, None], p[:, 1][:, None]
    ax, ay, bx, by = a[:, 0][None], a[:, 1][None], b[:, 0][None], b[:, 1][None]
    crosses = ((ay > y) != (by > y)) & (
        x < (bx - ax) * (y - ay) / np.where(by - ay == 0, 1e-300, by - ay) + ax)
    return crosses.sum(1) % 2 == 1


def circle_poly(cx: float, cy: float, r: float, n: int = 32) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], 1)


def stadium_poly(cx: float, cy: float, w: float, h: float,
                 n: int = 10) -> np.ndarray:
    """A slot: a rectangle w x h with semicircular ends on its LONG axis."""
    if w >= h:
        r, run = 0.5 * h, 0.5 * (w - h)
        t = np.linspace(-0.5 * np.pi, 0.5 * np.pi, n)
        right = np.stack([cx + run + r * np.cos(t), cy + r * np.sin(t)], 1)
        left = np.stack([cx - run - r * np.cos(t), cy - r * np.sin(t)], 1)
        return np.concatenate([right, left], 0)
    r, run = 0.5 * w, 0.5 * (h - w)
    t = np.linspace(0.0, np.pi, n)
    top = np.stack([cx + r * np.cos(t), cy + run + r * np.sin(t)], 1)
    bot = np.stack([cx - r * np.cos(t), cy - run - r * np.sin(t)], 1)
    return np.concatenate([top, bot], 0)


def rounded_rect(w: float, h: float, r: float, n: int = 6,
                 cx: float = 0.0, cy: float = 0.0) -> np.ndarray:
    r = min(r, 0.5 * w, 0.5 * h)
    out = []
    for sx, sy, a0 in ((1, 1, 0.0), (-1, 1, 0.5 * np.pi),
                       (-1, -1, np.pi), (1, -1, 1.5 * np.pi)):
        t = np.linspace(a0, a0 + 0.5 * np.pi, n)
        out.append(np.stack([cx + sx * (0.5 * w - r) + r * np.cos(t),
                             cy + sy * (0.5 * h - r) + r * np.sin(t)], 1))
    return np.concatenate(out, 0)


# ------------------------------------------------------------ triangulation


def triangulate(outer: np.ndarray, holes=(), max_rounds: int = 12):
    """A polygon with holes -> (points (M,2), triangles (T,3), loops).

    `loops` are index lists into `points`: the outer loop first (CCW),
    then each hole (CW), possibly with midpoints inserted by boundary
    recovery. The triangles are CCW and cover exactly the outline minus
    the holes: a triangle whose centroid falls outside the outline or
    inside a hole is dropped, which is sound only once every boundary
    edge is present -- hence the recovery loop.
    """
    def dedupe(poly):
        p = np.asarray(poly, dtype=float)
        keep = np.linalg.norm(p - np.roll(p, 1, axis=0), axis=1) > 1e-9
        return p[keep]

    loops = ([list(map(tuple, ccw(dedupe(outer))))]
             + [list(map(tuple, cw(dedupe(h)))) for h in holes])
    for _ in range(max_rounds):
        pts, idx = [], []
        for lp in loops:
            start = len(pts)
            pts.extend(lp)
            idx.append(list(range(start, start + len(lp))))
        pts = np.asarray(pts, dtype=float)
        tri = Delaunay(pts)
        edges = set()
        for s in tri.simplices:
            for a, b in ((s[0], s[1]), (s[1], s[2]), (s[2], s[0])):
                edges.add((min(a, b), max(a, b)))
        missing = []
        for li, lp_idx in enumerate(idx):
            m = len(lp_idx)
            for k in range(m):
                a, b = lp_idx[k], lp_idx[(k + 1) % m]
                if (min(a, b), max(a, b)) not in edges:
                    missing.append((li, k))
        if not missing:
            break
        # split every missing edge at its midpoint, highest index first so
        # earlier insertions do not shift the ones still to do
        for li, k in sorted(missing, key=lambda t: (t[0], -t[1])):
            lp = loops[li]
            a, b = lp[k], lp[(k + 1) % len(lp)]
            mid = (0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]))
            lp.insert(k + 1, mid)
    else:
        raise ValueError(f"boundary recovery did not converge: {len(missing)} "
                         f"edges still missing after {max_rounds} rounds")

    cent = pts[tri.simplices].mean(1)
    keep = points_in_polygon(cent, pts[idx[0]])
    for lp_idx in idx[1:]:
        keep &= ~points_in_polygon(cent, pts[lp_idx])
    tris = tri.simplices[keep]
    # CCW every triangle
    a, b, c = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
    cwise = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]) < 0
    tris = tris.copy()
    tris[cwise] = tris[cwise][:, ::-1]
    return pts, tris.astype(np.int64), idx


# ------------------------------------------------------------ mesh builder


class MeshBuilder:
    """Vertices and triangles accumulated with explicit index offsets, so
    parts can share rings and the result is one watertight mesh."""

    def __init__(self) -> None:
        self.verts: list[np.ndarray] = []
        self.tris: list[np.ndarray] = []
        self._n = 0

    def add_vertices(self, v: np.ndarray) -> int:
        v = np.asarray(v, dtype=float).reshape(-1, 3)
        self.verts.append(v)
        off = self._n
        self._n += len(v)
        return off

    def verts_at(self, i: int) -> np.ndarray:
        """One accumulated vertex, by global index."""
        n = 0
        for block in self.verts:
            if i < n + len(block):
                return block[i - n]
            n += len(block)
        raise IndexError(i)

    def add_triangles(self, t: np.ndarray) -> None:
        t = np.asarray(t, dtype=np.int64).reshape(-1, 3)
        if len(t):
            self.tris.append(t)

    def walls(self, ring_a, ring_b) -> None:
        """Quads between two rings of equal length, ring_a below ring_b
        in the sweep direction, both in the same cyclic order."""
        a, b = np.asarray(ring_a), np.asarray(ring_b)
        j_a, j_b = np.roll(a, -1), np.roll(b, -1)
        self.add_triangles(np.stack([a, j_a, j_b], 1))
        self.add_triangles(np.stack([a, j_b, b], 1))

    def build(self) -> tuple[np.ndarray, np.ndarray]:
        verts = np.concatenate(self.verts, 0)
        tris = np.concatenate(self.tris, 0)
        return verts, _stl._orient_outward(verts, tris)


def extrude_plate(mb: MeshBuilder, outer: np.ndarray, holes, thickness: float,
                  frame, connected: tuple = ()) -> dict:
    """A flat plate with holes, thickness along the frame's third axis.

    `frame(u, v, w)` maps plate coordinates to the mesh's 3-D frame.
    `connected` names holes (by index into `holes`) that exist on the TOP
    cap only: no wall is built for them and the bottom cap is solid
    there, because another solid starts on that ring. Returns the top
    ring's vertex indices for every connected hole, in CCW order, so the
    solid that continues them can share the vertices."""
    pts, tris_top, loops = triangulate(outer, holes)
    solid_holes = [h for i, h in enumerate(holes) if i not in connected]
    pts_b, tris_bot, loops_b = triangulate(outer, solid_holes)
    top_off = mb.add_vertices(np.array([frame(u, v, thickness) for u, v in pts]))
    bot_off = mb.add_vertices(np.array([frame(u, v, 0.0) for u, v in pts_b]))
    mb.add_triangles(tris_top + top_off)
    mb.add_triangles(tris_bot[:, ::-1] + bot_off)
    # walls: the outer loop and the solid holes. The two triangulations
    # may have split boundary edges differently, so walls are built on
    # the finer of each pair of rings with the other resampled onto it.
    for k, (lt, lb) in enumerate(zip(loops, loops_b)):
        if k > 0 and (k - 1) in connected:
            continue
        ring_t = pts[lt]
        ring_b = pts_b[lb]
        if len(ring_t) != len(ring_b) or not np.allclose(ring_t, ring_b):
            raise ValueError("plate caps disagree on a boundary ring")
        mb.walls(np.array(lb) + bot_off, np.array(lt) + top_off)
    out = {}
    for i in connected:
        ring = np.array(loops[i + 1]) + top_off      # CW as a hole ...
        out[i] = ring[::-1]                          # ... CCW as a boundary
    return out


def sweep(mb: MeshBuilder, sections: np.ndarray, start_ring=None,
          cap_start: bool = True, cap_end: bool = True) -> None:
    """Skin (L, N, 3) sections in order; each a closed loop in the same
    cyclic sense. `start_ring` gives vertex indices already in the mesh
    for section 0 (a plate's connected hole), in which case section 0 is
    not added and no start cap is built."""
    sections = np.asarray(sections, dtype=float)
    L, N, _ = sections.shape
    rings = []
    if start_ring is not None:
        rings.append(np.asarray(start_ring))
        first = 1
    else:
        off = mb.add_vertices(sections[0])
        rings.append(np.arange(N) + off)
        first = 0
    for k in range(first, L):
        off = mb.add_vertices(sections[k])
        rings.append(np.arange(N) + off)
    for a, b in zip(rings, rings[1:]):
        mb.walls(a, b)

    def cap(ring, sec, reverse):
        # ear-clip in the section's own plane: drop the sweep axis, which
        # is the direction of least extent across the loop
        spread = sec.max(0) - sec.min(0)
        keep = np.argsort(spread)[1:]
        tri = _stl._ear_clip(sec[:, sorted(keep)])
        tri = np.asarray(ring)[tri]
        mb.add_triangles(tri[:, ::-1] if reverse else tri)

    if cap_start and start_ring is None:
        cap(rings[0], sections[0], reverse=True)
    if cap_end:
        cap(rings[-1], sections[-1], reverse=False)


# ------------------------------------------------------------------ parts


@dataclass
class SolidPart:
    """A companion part printed in normal mode: its mesh, and what a
    build sheet needs to say about it."""

    name: str
    verts: np.ndarray
    tris: np.ndarray
    role: str
    orientation: str
    quantity: int = 1
    notes: tuple[str, ...] = ()
    gates: list = field(default_factory=list)

    @property
    def volume_mm3(self) -> float:
        return _stl.volume_mm3(self.verts, self.tris)

    def mass_g(self, density_gcc: float = SOLID_PLA_GCC) -> float:
        return self.volume_mm3 / 1000.0 * density_gcc

    @property
    def watertight(self) -> bool:
        return bool(_stl.manifold_report(self.tris)["watertight"])

    @property
    def footprint_mm(self) -> tuple[float, float, float]:
        e = self.verts.max(0) - self.verts.min(0)
        return float(e[0]), float(e[1]), float(e[2])

    def centroid_mm(self) -> np.ndarray:
        """Volume centroid by the divergence theorem over signed
        tetrahedra from the origin."""
        v = self.verts[self.tris]
        vol = np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2])) / 6.0
        c = v.mean(1)
        return (c * vol[:, None]).sum(0) / max(vol.sum(), 1e-12)

    def export_stl(self, path) -> dict:
        rep = _stl.manifold_report(self.tris)
        if not rep["watertight"]:
            raise ValueError(f"{self.name}: mesh is not watertight: {rep}")
        _stl.write_stl(path, self.verts, self.tris, header=f"washout {self.name}")
        rep["volume_cm3"] = self.volume_mm3 / 1000.0
        rep["path"] = str(path)
        return rep


def _gate(name, passed, value, limit, units, detail=""):
    from .vase import Gate
    return Gate(name, bool(passed), float(value), float(limit), units, detail)


# ------------------------------------------------------------ control horn

HORN_T_MM = 1.5
"""Horn plate thickness. Bearing stress on the 1.8 mm hole from a 9 g
servo's stall pull, about 13 N at an 11 mm arm, is 13 / (1.8 * 1.5) =
4.8 MPa against PLA's ~50: the plate is sized by the socket, not the
load."""
HORN_HOLE_MM = 1.8
"""For a 1.0-1.2 mm pushrod wire with a Z-bend. Printed vertically in
the plate, so it comes out round."""
HORN_TONGUE_MIN_MM = 8.0
HORN_TONGUE_MAX_MM = 24.0
"""Chordwise length of the tongue, between which it is SIZED to the load.

A moulded nylon horn's base is fifteen to twenty millimetres long, so
this range is ordinary hardware; what is not ordinary is choosing one
number out of it in advance, which is what `HORN_TONGUE_MM = 10` did.
The capacity goes as the length squared and the depth reachable falls as
the elevon's tail thins, so the shortest tongue that carries the load is
a small search, not a guess."""

HORN_TONGUE_MM = 10.0
"""Nominal, for the parts that only need a size to draw."""
SOCKET_DEPTH_FACTOR = 3.0
"""The socket is this many horn thicknesses deep when the section
allows: the tongue is held in bearing by the skin, and the glue only
keeps it there."""
HORN_BOND_MPA = 1.0
"""Shear a bond between the horn and the socket's wall is credited with.
The same DECLARED allowable the panel joints and the motor mount use,
and for the same reason: the foam is the weak side, and it has not been
tested here."""

HORN_SAFETY = 1.0
"""Factor on the servo's STALL torque.

Stall is already the worst case the mechanism can produce: it is what a
servo delivers into a jammed surface, not what it sees in flight, where
the hinge moment is a small fraction of it. A factor on top of a worst
case is two margins stacked, and `Mission.min_aeroelastic_margin` makes
exactly this argument for exactly this reason -- it rejects designs for
arithmetic rather than for physics.

So the requirement is that the horn is NOT the fuse: under an abuse the
servo's own gears are meant to give first, and at 1.0 the socket holds
while they do. It is not a licence, either. micro still fails it at
0.9x, because a 28 mm elevon chord has 1.5 mm of section to socket
into, and that is a finding about micro rather than about the number."""


def socket_moment_capacity_nmm(length_mm: float, depth_mm: float,
                               allow_mpa: float = HORN_BOND_MPA) -> float:
    """What a socket of this size can resist before the horn levers out.

    The pushrod pulls the blade at a height above the surface, so the
    buried tongue is loaded as a COUPLE, not in shear: the bond on the
    forward half pulls one way and the aft half the other. Taking the
    bond stress as triangular along the tongue, peaking at its ends, the
    couple its two faces can carry is

        M = tau * (2 * L * d) * L / 4

    -- bonded area times the allowable times the lever between the two
    halves' centroids. A thickness ratio cannot express this: it has no
    length in it, and the tongue's LENGTH is most of the strength."""
    return float(allow_mpa * (2.0 * length_mm * depth_mm) * length_mm / 4.0)
SOCKET_CLEAR_MM = 0.15
"""Fit clearance each side of the tongue in the socket's full-depth run."""


def socket_depth_mm(section_thick_mm: float, wall_mm: float) -> float:
    """How deep the socket can be: the tongue's wish, or what the section
    leaves after both skins and a bead of floor clearance."""
    return float(max(min(SOCKET_DEPTH_FACTOR * HORN_T_MM,
                         section_thick_mm - 2.0 * wall_mm - wall_mm), 0.0))


def horn_geometry(plan, link, eta_h: float, wall_mm: float,
                  applied_nmm: float = 0.0):
    """Where the horn sits and how deep its socket can be.

    -> (x_mid_mm, socket_len_mm, socket_depth_mm, horn_above_mm), the
    station in mm aft of the ROOT leading edge.

    The hole has to end up where the four-bar put it: `hinge_x_mm` aft of
    the local leading edge plus `horn_dx_mm` aft of the axis, which is
    how far the horn stands clear of the hinge tape. The blade's lobe is
    half a strap wide, so the tongue's forward end is that much ahead of
    the hole, and the socket is the tongue plus a fit clearance at each
    end."""
    st = plan.at(float(np.clip(eta_h, 0.0, 1.0)))
    c_mm, x_le = st.chord_m * 1000.0, st.x_le_m * 1000.0
    x_hole = x_le + link.hinge_x_mm + link.horn_dx_mm
    x0 = x_hole - 0.5 * _lkg.HORN_STRAP_MM

    def depth_for(length_mm):
        xs = np.clip(np.linspace(x0 - x_le, x0 + length_mm - x_le, 7)
                     / max(c_mm, 1e-9), 0.0, 1.0)
        return socket_depth_mm(float(st.airfoil.thickness(xs).min()) * c_mm,
                               wall_mm)

    # The tongue is SIZED, not declared. Its capacity goes as the length
    # squared while the depth it can reach falls as the tail thins, so
    # there is a shortest length that carries the load and it is not
    # obvious by eye: 10 mm was a number written before the load was
    # computed, and it left every aircraft in the fleet at 1.4x against
    # a 1.5x requirement. The shortest that passes wins, because a longer
    # tongue is more of the elevon removed; if none does, the longest is
    # returned and the gate prices the shortfall. Same shape as
    # `structure.select` picking the lightest tube that survives.
    best = (HORN_TONGUE_MIN_MM, depth_for(HORN_TONGUE_MIN_MM))
    for L in np.arange(HORN_TONGUE_MIN_MM, HORN_TONGUE_MAX_MM + 1e-9, 1.0):
        d = depth_for(float(L))
        if socket_moment_capacity_nmm(float(L), d) > socket_moment_capacity_nmm(*best):
            best = (float(L), d)
        if (applied_nmm > 0.0
                and socket_moment_capacity_nmm(float(L), d)
                >= HORN_SAFETY * applied_nmm):
            best = (float(L), d)
            break
    length, depth = best
    return (x0 + 0.5 * length, length + 2.0 * SOCKET_CLEAR_MM, depth,
            float(link.horn_arm_mm), length)


def horn_for(plan, link, eta_h: float, wall_mm: float, name: str,
             applied_nmm: float = 0.0) -> SolidPart:
    """The horn, built to the elevon's own surface at its station.

    The tongue's top edge follows that surface rather than sitting flat
    on its crown, so the blade stands square to the section instead of
    rocking. The section is taken untwisted, which is the frame the horn
    is glued in: twist rotates the whole section and the horn with it."""
    x_mid, _, depth, above, length = horn_geometry(plan, link, eta_h,
                                                   wall_mm, applied_nmm)
    st = plan.at(float(np.clip(eta_h, 0.0, 1.0)))
    c_mm, x_le = st.chord_m * 1000.0, st.x_le_m * 1000.0
    x0 = x_mid - 0.5 * length

    def y_up(x_abs):
        f = float(np.clip((x_abs - x_le) / max(c_mm, 1e-9), 0.0, 1.0))
        return float(st.airfoil.y_upper(np.array([f]))[0]) * c_mm

    y0 = y_up(x0)
    return control_horn(name, depth, above, _lkg.HORN_STRAP_MM,
                        lambda x: y_up(x0 + x) - y0,
                        applied_nmm=applied_nmm, tongue_mm=length)


def control_horn(name: str, socket_depth: float, horn_above_mm: float,
                 strap_mm: float, surface_y_at,
                 applied_nmm: float = 0.0,
                 tongue_mm: float = HORN_TONGUE_MM) -> SolidPart:
    """The horn as a flat plate in the section plane (x aft, y up), the
    elevon's upper surface at y = 0 over the tongue.

    Outline, forward to aft: the tongue from x = 0 to `tongue_mm`,
    `socket_depth` below the surface; the blade a strap `strap_mm` wide
    rising from the tongue's forward end to a semicircular lobe around
    the hole, `horn_above_mm` above the surface. The tongue's top edge
    follows the actual surface (`surface_y_at(x)`, mm relative to the
    forward end), so the plate sits on the skin rather than rocking on
    its crown. Prints lying flat; the hole is vertical."""
    if socket_depth <= 0.0:
        raise ValueError(
            f"{name}: a horn with a {socket_depth:.2f} mm tongue is not a "
            f"horn -- the section at its station has no depth to socket "
            f"into, and the outline would come back degenerate")
    r = 0.5 * strap_mm
    xs_t = np.linspace(0.0, tongue_mm, 7)
    surf = np.array([surface_y_at(x) for x in xs_t])
    # the socket's floor: parallel to the surface, socket_depth below
    pts = [(x, y - socket_depth) for x, y in zip(xs_t, surf)]           # floor, fwd->aft
    pts += [(tongue_mm, surf[-1])]                                 # aft wall up to surface
    aft_x = min(strap_mm, tongue_mm)
    pts += [(x, float(np.interp(x, xs_t, surf)))
            for x in np.linspace(tongue_mm, aft_x, 5)[1:]]         # along the surface, aft->fwd
    pts += [(aft_x, horn_above_mm)]                                     # blade's aft edge up
    t = np.linspace(0.0, np.pi, 12)
    pts += [(r + r * np.cos(a), horn_above_mm + r * np.sin(a)) for a in t]   # the lobe
    pts += [(0.0, surf[0])]                                             # forward edge down to the surface
    outline = np.array(pts)
    hole = circle_poly(r, horn_above_mm, 0.5 * HORN_HOLE_MM, 24)
    mb = MeshBuilder()
    extrude_plate(mb, outline, [hole], HORN_T_MM,
                  frame=lambda u, v, w: (u, v, w))
    verts, tris = mb.build()
    part = SolidPart(name, verts, tris, role="horn",
                     orientation="flat on the bed, tongue and blade in the "
                                 "bed plane; the hole prints vertical",
                     quantity=2,
                     notes=(f"tongue {tongue_mm:.0f} x {socket_depth:.1f} mm "
                            f"into the elevon's socket; hole {HORN_HOLE_MM} mm",))
    cap = socket_moment_capacity_nmm(tongue_mm, socket_depth)
    if applied_nmm > 0.0:
        part.gates.append(_gate(
            "horn socket holds", cap >= HORN_SAFETY * applied_nmm,
            cap / applied_nmm, HORN_SAFETY, "x",
            f"{cap:.0f} N.mm of couple from a {tongue_mm:.0f} x "
            f"{socket_depth:.1f} mm tongue at {HORN_BOND_MPA} MPa "
            f"(declared) against {applied_nmm:.0f} the servo can lever"))
    return part


# ------------------------------------------------------------- motor mount

WEB_T_MM = 3.0
FLANGE_T_MM = 1.5
FLANGE_L_MM = 25.0
MOUNT_W_MM = 30.0
"""The saddle: a web normal to the thrust line just aft of the trailing
edge, with two flanges lying on the upper and lower skins. All declared
dimensions of a printed part; what is gated is what they bond to."""
BOLT_SLOT_MM = (3.4, 4.9)
"""M3 clearance slots long enough to take both the 16 x 16 and the
16 x 19 mm patterns a 2205 comes with."""
BOLT_PATTERN_MM = (16.0, 17.5)
CENTRE_HOLE_MM = 8.0
"""Clearance for the shaft end and its circlip proud of the base."""
ROOT_CLEAR_MM = 1.0
"""Web material kept between the upper flange's root and the nearest
bolt slot. The thrust axis is placed as low as this allows: the flanges
meet the web's forward face within a few millimetres of the trailing
edge's mid-line, and a bolt pattern centred there would cut through
their roots. The axis therefore stands clear above the wing -- where a
pusher's motor sits on a wing in any case -- and the offset between the
thrust line and the mid-line is REPORTED. The CG's height is not
modelled, so the pitching moment of the thrust is not either."""
GLUE_GAP_MM = 0.2
BOND_ALLOWABLE_MPA = 1.0
"""Shear a CA or epoxy bond to foamed PLA is credited with. A DECLARED
allowance, deliberately far below any published figure for the
adhesive itself, because the foam is the weak side and has not been
tested here. The gates report the margin against it."""
BOND_SAFETY = 3.0


def motor_mount(name: str, skin_y_upper, skin_y_lower, x_te_of,
                span_mm: float = MOUNT_W_MM, prop_plane_mm: float = 26.0,
                channel_w_mm: float = 0.0) -> SolidPart:
    """The saddle over the blunt trailing edge.

    `skin_y_upper(x, z)` / `skin_y_lower(x, z)` give the skins' OUTER
    surfaces in planform mm (x aft of the root LE, z along the span);
    `x_te_of(z)` the trailing edge's station. The web stands 0.5 mm aft
    of the aft-most trailing edge under it, normal to x, and the flanges
    run forward from its face along both skins with `GLUE_GAP_MM` of
    adhesive under them. Print: web face down, flanges standing.
    `channel_w_mm` of the upper flange's width is over the motor-wire
    channel and does not bond."""
    zs = np.linspace(-0.5 * span_mm, 0.5 * span_mm, 13)
    x_web = float(max(x_te_of(z) for z in zs)) + 0.5
    y_mid = 0.5 * (skin_y_upper(x_web - 1.0, 0.0) + skin_y_lower(x_web - 1.0, 0.0))

    # part frame: u = span, v = thickness (up), w = forward from the web's
    # aft face. Print Z is w: the web lies on the bed, the flanges stand.
    def frame(u, v, w):
        return (u, v, w)

    # flange root rings on the web's forward face -- INSET from the web's
    # sides. A flange as wide as the web puts its root's end vertices
    # exactly on the web outline's side edges, and a boundary edge with
    # a point lying on it can never be recovered by any triangulation.
    m = 9
    inset = 2.0
    us = np.linspace(-0.5 * span_mm + inset, 0.5 * span_mm - inset, m)

    def flange_section(w, sign):
        x = x_web - w
        skin = skin_y_upper if sign > 0 else skin_y_lower
        vin = np.array([skin(min(x, x_te_of(u) - 0.5), u) + sign * GLUE_GAP_MM
                        for u in us])
        vout = vin + sign * FLANGE_T_MM
        loop = np.concatenate([np.stack([us, vin], 1),
                               np.stack([us[::-1], vout[::-1]], 1)], 0)
        return ccw(loop)

    def flange_at(w, sign, ring_uv=None):
        """The flange's section at `w`, in the ring's own parameterisation.

        `extrude_plate` triangulates the web with the flange roots as
        holes, and its boundary recovery SPLITS any hole edge the
        triangulation missed -- so the ring it hands back has more
        vertices than the polygon that was passed in, and in an order
        only it knows. The sweep has to start on that ring, so every
        later section is built by carrying each of ITS points forward:
        the point's u is kept, and its v is recomputed from the skin at
        the new w, on whichever of the two edges it started on.

        Rebuilding the section from `us` instead silently assumed the
        recovery had changed nothing. It had not on a synthetic wedge or
        on the trainer, and it had on micro, whose thinner root needed
        the extra points -- `np.stack` then refused to wall a 20-point
        ring against an 18-point section, and the mount was skipped."""
        x = x_web - w
        skin = skin_y_upper if sign > 0 else skin_y_lower
        vin = np.array([skin(min(x, x_te_of(u) - 0.5), u) + sign * GLUE_GAP_MM
                        for u in us])
        if ring_uv is None:
            vout = vin + sign * FLANGE_T_MM
            return ccw(np.concatenate([np.stack([us, vin], 1),
                                       np.stack([us[::-1], vout[::-1]], 1)], 0))
        out = []
        for u, v in ring_uv:
            inner = float(np.interp(u, us, vin))
            was_in = float(np.interp(u, us, vin_root))
            # which edge did this vertex start on? the one it is nearer
            on_outer = abs(v - (was_in + sign * FLANGE_T_MM)) < abs(v - was_in)
            out.append((u, inner + (sign * FLANGE_T_MM if on_outer else 0.0)))
        return np.array(out, dtype=float)

    roots = [flange_section(WEB_T_MM, +1), flange_section(WEB_T_MM, -1)]
    root_top = float(max(v for _, v in roots[0]))
    root_bot = float(min(v for _, v in roots[1]))
    # the axis as low as the bolt slots allow above the upper root
    y_axis = (root_top + ROOT_CLEAR_MM
              + 0.5 * BOLT_PATTERN_MM[1] + 0.5 * BOLT_SLOT_MM[1])
    web_top = y_axis + 0.5 * BOLT_PATTERN_MM[1] + 0.5 * BOLT_SLOT_MM[1] + 3.0
    web_bot = root_bot - 6.0
    web_h = web_top - web_bot
    outer = rounded_rect(span_mm, web_h, 3.0, cy=0.5 * (web_top + web_bot))
    holes = []
    for su in (-1, 1):
        for sv in (-1, 1):
            holes.append(stadium_poly(su * 0.5 * BOLT_PATTERN_MM[0],
                                      y_axis + sv * 0.5 * BOLT_PATTERN_MM[1],
                                      BOLT_SLOT_MM[0], BOLT_SLOT_MM[1]))
    holes.append(circle_poly(0.0, y_axis, 0.5 * CENTRE_HOLE_MM, 32))

    holes += roots
    mb = MeshBuilder()
    rings = extrude_plate(mb, outer, holes, WEB_T_MM, frame,
                          connected=(len(holes) - 2, len(holes) - 1))
    for sign, key in ((+1, len(holes) - 2), (-1, len(holes) - 1)):
        ring = rings[key]
        ring_uv = [(float(mb.verts_at(i)[0]), float(mb.verts_at(i)[1]))
                   for i in ring]
        x0f = x_web - WEB_T_MM
        skin0 = skin_y_upper if sign > 0 else skin_y_lower
        vin_root = np.array([skin0(min(x0f, x_te_of(u) - 0.5), u)
                             + sign * GLUE_GAP_MM for u in us])
        ws = np.linspace(WEB_T_MM, WEB_T_MM + FLANGE_L_MM, 12)
        secs = np.array([[frame(u, v, w)
                          for u, v in flange_at(w, sign, ring_uv)]
                         for w in ws])
        sweep(mb, secs, start_ring=ring, cap_end=True)
    verts, tris = mb.build()
    part = SolidPart(name, verts, tris, role="motor mount",
                     orientation="web face down on the bed, flanges standing",
                     notes=(f"web {span_mm:.0f} x {web_h:.0f} x {WEB_T_MM:.0f} mm at "
                            f"x = {x_web:.1f} mm; flanges {FLANGE_L_MM:.0f} mm "
                            f"forward on both skins; M3 slots for a 16x16 or 16x19 "
                            f"pattern; {CENTRE_HOLE_MM:.0f} mm centre hole",))
    part.x_web_mm = x_web
    part.y_mid_mm = float(y_mid)
    part.y_axis_mm = float(y_axis)
    part.axis_offset_mm = float(y_axis - y_mid)
    # the lowest bolt slot must clear the upper flange's root, or the web
    # has a hole through the flange's attachment
    slot_bot = y_axis - 0.5 * BOLT_PATTERN_MM[1] - 0.5 * BOLT_SLOT_MM[1]
    part.gates.append(_gate("bolt slots clear flange root",
                            slot_bot >= root_top + ROOT_CLEAR_MM - 1e-9,
                            slot_bot - root_top, ROOT_CLEAR_MM, "mm",
                            f"thrust axis {y_axis - y_mid:.1f} mm above the "
                            f"trailing edge mid-line"))
    part.prop_plane_x_mm = x_web + WEB_T_MM + prop_plane_mm
    part.bond_area_mm2 = float(FLANGE_L_MM * (2.0 * (span_mm - 2.0 * inset)
                                              - channel_w_mm))
    return part


def mount_gates(part: SolidPart, thrust_n: float, torque_nm: float,
                x_te_of, prop_r_mm: float, belly_y_mm: float,
                span_limit_mm: float, clearance_mm: float = 10.0) -> list:
    """Every claim about the mount, with a number.

    - bond: the flanges' bonded area at `BOND_ALLOWABLE_MPA`, against
      `BOND_SAFETY` times static thrust as shear, and the motor's torque
      reacted as a shear couple across the two flanges.
    - prop clearance: the disc plane to the trailing edge everywhere
      the disc reaches along the span.
    - belly clearance is REPORTED, not gated: on every pusher wing the
      prop reaches below the belly, and the build sheet says to stop it
      horizontal with the ESC brake."""
    gates = []
    cap_n = part.bond_area_mm2 * BOND_ALLOWABLE_MPA
    gates.append(_gate("mount bond vs thrust", cap_n >= BOND_SAFETY * thrust_n,
                       cap_n / max(thrust_n, 1e-9), BOND_SAFETY, "x",
                       f"{part.bond_area_mm2:.0f} mm2 at {BOND_ALLOWABLE_MPA} MPa "
                       f"(declared) vs {thrust_n:.1f} N static thrust"))
    # torque about the thrust axis: a shear couple across the flanges,
    # lever arm the section thickness at the flange, taken as 2 mm at
    # worst -- the thinnest a blunt trailing edge gets
    couple_n = torque_nm * 1000.0 / 2.0
    gates.append(_gate("mount bond vs torque", cap_n >= BOND_SAFETY * couple_n,
                       cap_n / max(couple_n, 1e-9), BOND_SAFETY, "x",
                       f"{torque_nm * 1000:.0f} N.mm of motor torque as a couple"))
    ys = np.linspace(-prop_r_mm, prop_r_mm, 25)
    ys = ys[np.abs(ys) <= span_limit_mm]
    gap = float(min(part.prop_plane_x_mm - x_te_of(y) for y in ys))
    gates.append(_gate("prop to trailing edge", gap >= clearance_mm, gap,
                       clearance_mm, "mm",
                       f"disc plane at x = {part.prop_plane_x_mm:.0f} mm, "
                       f"radius {prop_r_mm:.0f} mm"))
    part.belly_clearance_mm = float((part.y_axis_mm - prop_r_mm) - belly_y_mm)
    return gates


# --------------------------------------------------------------- hatch lid


def skin_functions(plan, settings: PrintSettings, n: int = 25):
    """(y_upper, y_lower, x_te) over the planform, in millimetres.

    Each takes (x_mm aft of the root leading edge, z_mm from the
    centreline) and returns the OUTER surface, built from the same
    thickened, twisted station loops the panels are. `x_te` takes z
    alone.

    Tabulated over a span of stations and interpolated, because the
    motor mount asks for a few hundred samples over the 30 mm at the
    centreline and rebuilding a CST loop for each would dominate the
    export. Dihedral is ignored: the mount lives within a few
    centimetres of the centreline, where it is under a tenth of a
    millimetre, and a mount that spanned enough for it to matter would
    be a mount the trailing edge is curved under anyway."""
    half_mm = plan.half_span_m * 1000.0
    zs = np.linspace(0.0, min(0.25 * half_mm, half_mm), n)
    tab = []
    for z in zs:
        st, chord_mm, loop = _station_loop(plan, float(z) / max(half_mm, 1e-9),
                                           settings)
        pts = _to_planform_mm(loop, st, chord_mm)
        i_le = int(np.argmin(pts[:, 0]))
        up = pts[:i_le + 1][::-1]
        lo = pts[i_le:]
        tab.append((up[np.argsort(up[:, 0])], lo[np.argsort(lo[:, 0])]))

    def at(z_mm, which):
        k = float(np.clip(abs(z_mm), zs[0], zs[-1]))
        i = int(np.clip(np.searchsorted(zs, k) - 1, 0, len(zs) - 2))
        f = (k - zs[i]) / max(zs[i + 1] - zs[i], 1e-9)
        return tab[i][which], tab[i + 1][which], f

    def y_upper(x_mm, z_mm):
        a, b, f = at(z_mm, 0)
        return float((1 - f) * np.interp(x_mm, a[:, 0], a[:, 1])
                     + f * np.interp(x_mm, b[:, 0], b[:, 1]))

    def y_lower(x_mm, z_mm):
        a, b, f = at(z_mm, 1)
        return float((1 - f) * np.interp(x_mm, a[:, 0], a[:, 1])
                     + f * np.interp(x_mm, b[:, 0], b[:, 1]))

    def x_te(z_mm):
        a, b, f = at(z_mm, 0)
        return float((1 - f) * a[-1, 0] + f * b[-1, 0])

    return y_upper, y_lower, x_te


def te_station(plan):
    """x of the trailing edge, in mm aft of the root leading edge, as a
    function of z in mm from the centreline.

    From the planform alone -- leading edge plus chord -- so it costs two
    interpolations rather than a rebuilt section loop, and the score can
    afford to ask it. Dihedral is ignored, as it is for the mount.
    """
    half_mm = plan.half_span_m * 1000.0

    def x_te(z_mm):
        st = plan.at(float(np.clip(abs(z_mm) / max(half_mm, 1e-9), 0.0, 1.0)))
        return (st.x_le_m + st.chord_m) * 1000.0

    return x_te


def disc_plane_mm(plan, span_mm: float = MOUNT_W_MM) -> float:
    """Where the propeller disc sits, mm aft of the root leading edge:
    the mount's web just aft of the trailing edge under it, its own
    thickness, and the motor's shaft and hub."""
    x_te = te_station(plan)
    zs = np.linspace(-0.5 * span_mm, 0.5 * span_mm, 13)
    return float(max(x_te(z) for z in zs)) + 0.5 + WEB_T_MM + PROP_PLANE_MM


def prop_clearance_mm(plan, prop_diam_in: float, n: int = 25) -> float:
    """Closest the disc plane comes to the trailing edge, anywhere the
    disc reaches along the span.

    A pusher on a SWEPT wing loses this outboard: the trailing edge runs
    aft as the disc runs out, so the tips of the blades are the part in
    danger, not the root. Measured over the disc's own radius, clipped to
    the half span."""
    x_te = te_station(plan)
    plane = disc_plane_mm(plan)
    r = 0.5 * prop_diam_in * 25.4
    half_mm = plan.half_span_m * 1000.0
    zs = np.linspace(-r, r, n)
    zs = zs[np.abs(zs) <= half_mm]
    if not len(zs):
        return float(plane - x_te(0.0))
    return float(min(plane - x_te(float(z)) for z in zs))


def mount_for(plan, settings: PrintSettings, powertrain, name: str,
              span_mm: float = MOUNT_W_MM):
    """The motor mount and its gates, from the design that was scored.

    -> (part, gates). The thrust is the powertrain's own STATIC figure,
    which is the load case a mount sees on the bench and on a hand
    launch; the torque is the shaft power at the loaded rpm, which is
    what tries to twist it off the trailing edge."""
    y_up, y_lo, x_te = skin_functions(plan, settings)
    part = motor_mount(name, y_up, y_lo, x_te, span_mm=span_mm,
                       prop_plane_mm=PROP_PLANE_MM)
    thrust_n = powertrain.power_limited_thrust_n(0.0)
    omega = 2.0 * np.pi * powertrain.rpm() / 60.0
    p_shaft = powertrain.motor.max_power_w * powertrain.eta_esc * 0.80
    torque_nm = p_shaft / max(omega, 1e-9)
    prop_r = 0.5 * powertrain.prop.diameter_in * 25.4
    belly = min(y_lo(x_te(0.0) - 5.0, 0.0), y_lo(0.5 * x_te(0.0), 0.0))
    # the SAME trailing edge the score's clearance gate uses, so the two
    # can never disagree about where the wing ends
    gates = mount_gates(part, thrust_n, torque_nm, te_station(plan), prop_r,
                        belly, plan.half_span_m * 1000.0)
    return part, gates


PROP_PLANE_MM = 26.0
"""Disc plane aft of the mount's forward face: a 2205's shaft length
plus the hub. A declared dimension of the hardware."""


def lid_thickness_mm(settings: PrintSettings) -> float:
    """Two beads that touch: the lens is one bead inside the other with
    `rib_clearance_factor` of the bead width between the toolpaths, so
    the two walls weld into a solid curved plate -- the same rule the
    rib slits use. The ledge the lid rests on is cut this deep."""
    return settings.extrusion_width_mm * (1.0 + settings.rib_clearance_factor)


def _station_loop(plan, eta: float, settings: PrintSettings):
    st = plan.at(float(eta))
    chord_mm = st.chord_m * 1000.0
    loop = thicken_for_nozzle(st.airfoil.coords(settings.contour_points),
                              chord_mm, settings)
    return st, chord_mm, loop


def _to_planform_mm(pts_unit: np.ndarray, st, chord_mm: float) -> np.ndarray:
    p = pts_unit - np.array([0.25, 0.0])
    a = np.radians(-st.twist_deg)
    ca, sa = np.cos(a), np.sin(a)
    rot = np.stack([p[:, 0] * ca - p[:, 1] * sa,
                    p[:, 0] * sa + p[:, 1] * ca], 1)
    return rot * chord_mm + np.array([st.x_le_m * 1000.0 + 0.25 * chord_mm, 0.0])


def hatch_lid(plan, settings: PrintSettings, cut, name: str,
              z_step_mm: float | None = None,
              n_pts: int = 41) -> LayerStack | None:
    """The lid over a root bay, as a vase-style stack across BOTH halves.

    At each span station the lid is a lens: the upper skin's contour
    between the bay's walls, and the same arc one bead lower, closed by
    two blunt ends. Printed on its edge, span up, like the panel it sits
    on -- so it follows the wing's own curvature with no support and no
    second print mode -- as two beads that weld into one plate.

    It covers the span over which the ledge is at its full depth. Beyond
    that the ramp has made the opening shallower than the lid and the
    ledge has become the floor; what remains there is a taper the lid
    cannot sit in.

    ONE PER HALF, printed from the centreline outward, and the pair meets
    over the centre joint the panels already have. Mirroring it into a
    single part across both halves was the first version and it failed
    two gates at once: 259 mm of print height against a 250 mm envelope,
    and a first layer of 32 mm2 against the 40 mm2 a small part needs to
    stay on the bed -- because a tip-to-tip lid begins at an OUTBOARD end,
    where the bay has narrowed to almost nothing. Begun at the
    centreline it starts on the widest, thickest section it has."""
    if cut.open_from != UPPER or cut.ledge_mm <= 0.0 or cut.eta0 > 1e-9:
        return None
    step = z_step_mm or settings.layer_h_mm
    # planform-level spec: full depth from the root face, one ramp out
    span_mm = arc_length_mm(plan, 0.0, cut.eta1) + cut.dead_out_mm
    spec = BaySpec(cut.name, cut.x0, cut.x1, 0.0, span_mm, cut.ramp_out_mm,
                   cut.depth_mm, settings.extrusion_width_mm,
                   open_from=UPPER, ledge_mm=cut.ledge_mm,
                   length_mm=cut.length_mm, x_abs_mm=cut.x_abs_mm)
    z_max = span_mm + cut.ramp_out_mm
    zs = np.arange(0.0, z_max + 1e-9, step)
    # z -> eta by arc length along the half span
    e_tab = np.linspace(0.0, 1.0, 400)
    s_tab = np.array([0.0] + [arc_length_mm(plan, 0.0, e, n=60) for e in e_tab[1:]])
    t_lid = lid_thickness_mm(settings)
    contours, z_keep = [], []
    for z in zs:
        eta = float(np.interp(z, s_tab, e_tab))
        st, c_mm, loop = _station_loop(plan, eta, settings)
        upper, lower = split_skins(loop)
        x_le = st.x_le_m * 1000.0
        det = _detour(spec, upper, lower, c_mm, float(z),
                      settings.extrusion_width_mm, x_le)
        x0, x1 = spec.band(c_mm, x_le)
        y_l0 = float(np.interp(x0, upper[:, 0], upper[:, 1])) - spec.ledge_mm / c_mm
        y_l1 = float(np.interp(x1, upper[:, 0], upper[:, 1])) - spec.ledge_mm / c_mm
        if abs(det[0, 1] - y_l0) > 1e-9 or abs(det[5, 1] - y_l1) > 1e-9:
            break                                   # ledge clamped: lid ends
        xs = x0 + (x1 - x0) * 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n_pts)))
        ys = np.interp(xs, upper[:, 0], upper[:, 1])
        outer = np.stack([xs, ys], 1)
        inner = np.stack([xs, ys - t_lid / c_mm], 1)
        lens = np.concatenate([outer[::-1], inner], 0)      # 2n, two blunt ends
        contours.append(_to_planform_mm(lens, st, c_mm))
        z_keep.append(float(z))
    if len(contours) < 3:
        return None
    full = np.array(contours)
    z_full = np.array(z_keep)
    flat = full.reshape(-1, 2)
    origin = 0.5 * (flat.min(0) + flat.max(0))
    full = full - origin
    eta_full = np.array([float(np.interp(z, s_tab, e_tab)) for z in z_full])
    return LayerStack(z_mm=z_full, eta=eta_full, contours=full, settings=settings,
                      name=name, z_step_mm=step, has_ribs=False, role="lid",
                      n_upper=n_pts, origin_mm=(float(origin[0]), float(origin[1])))
