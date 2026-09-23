"""Joint wedge inserts: the piece of wing a turning joint leaves out.

## Why there is a gap at all

Each panel prints root-down in vase mode, so both of its end faces are
square to its OWN axis (frames.py). Where the dihedral turns at a joint,
the inner panel's tip face and the outer panel's root face are two planes
through the joint's pivot at the kink angle between the axes -- and the
loft between them belongs to neither part. On micro_fpv's gen7 winner that
is a 17.3-degree wedge opening to 6.25 mm at the lower skin, a real piece
of the aeroplane's surface. The build sheet told the builder to fill it
with glue; CAD showed two parts that did not share a face.

Neither end face can be tilted instead (ROADMAP-BUILD.md section 0): a
sloped ROOT face is a sloped bottom on the bed, and a sloped TIP face is a
cut that moves across a hollow vase part by ~0.8 mm a layer, printed over
air. So the missing loft is its own part.

## What the insert is

Exactly the loft between the two joint planes. Both faces are slices of
the same unit loop (`frames.slice_layers`) through the two planes, and a
loop point is the same chordwise station in each, so point i of one face
and point i of the other bound one straight line of the skin between them
-- at most a few millimetres long on a loft that is smooth there. Face A
is the outer panel's root face, face B the inner panel's tip face: glued
to them, the three parts share faces exactly.

Near the pivot the wedge thins to nothing. A knife edge cannot print, so
the insert stops where it is `min_thickness_mm` thick -- two beads -- and
the thin crest it leaves, never wider than that, is glue. Reported.

Where a joint is also the elevon's root, the elevon moves and the insert
must not bond to it: the insert covers the wing forward of the hinge cut
(the outer panel is truncated there too), and the elevon's own root wedge
is left as its clearance, reported with its size.

Printed flat on face A in NORMAL mode -- a solid part, not a vase: its
top, face B, is an upward-facing ramp the slicer caps with top layers.
Tubes that cross the joint pass through it in a bore, an exact
cylinder-plane cut on both faces.

## Mass

Charged in the search, at its centroid, both sides. A normal-mode part is
a shell plus sparse infill; the estimate below takes a 0.8 mm shell (two
0.45 mm walls; three 0.25 mm layers top and bottom) and 15% infill, which
for a wedge a few millimetres thick is nearly solid anyway. Declared, not
measured: the slicer's own estimate is the one to trust once it exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SHELL_MM = 0.8
INFILL = 0.15
FILL_DENSITY_GCC = 0.6
"""Microballoon-thickened epoxy, the filler for a wedge too thin for a
printed insert. DECLARED -- the class of material, not a weighed batch."""


@dataclass
class JointInsert:
    """One joint's wedge insert (the right wing's; print two, mirror one)."""

    name: str
    joint: int                     # the insert fills p{joint-1} / p{joint}
    eta: float
    kink_deg: float
    pivot: str
    flight_verts: np.ndarray       # (V,3) flight mm, starboard
    verts: np.ndarray              # (V,3) print mm, face A on the bed
    tris: np.ndarray               # (T,3)
    max_gap_mm: float              # thickest point, at the far skin
    crest_mm: float                # the unfilled crest left at the pivot
    elevon_gap_mm: float = 0.0     # elevon root clearance wedge, if any
    bores: list = field(default_factory=list)
    """Per tube: {"name", "d_mm", "wall_mm"} -- wall is the least material
    between the bore and the insert's outline, on either face."""
    arc_a: np.ndarray | None = None      # face A's outline (flight mm), open
    arc_b: np.ndarray | None = None      # face B's, point for point
    bore_axes: list = field(default_factory=list)
    """(point, unit direction, radius) of each bore CUT, flight mm."""
    print_R: np.ndarray | None = None
    print_t: np.ndarray | None = None
    """print = print_R @ left + print_t, where `left` is the flight point
    mirrored to the left wing (y -> -y): a PROPER rigid motion, so a CAD
    kernel can place the part without a reflection."""
    glue_fill: bool = False
    """The wedge is FILLED, not printed: a tube severs it -- a thin wedge
    along the skin with the spar seated in it, which no notch leaves in
    one piece. The mesh is then the fill's volume and nothing prints."""

    def fill_g(self) -> float:
        return FILL_DENSITY_GCC * self.volume_mm3 / 1000.0

    @property
    def volume_mm3(self) -> float:
        v = self.verts[self.tris]
        return float(abs(np.einsum("ij,ij->i", v[:, 0],
                                   np.cross(v[:, 1], v[:, 2])).sum()) / 6.0)

    @property
    def area_mm2(self) -> float:
        v = self.verts[self.tris]
        return float(0.5 * np.linalg.norm(
            np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1).sum())

    def mass_g(self, density_gcc: float) -> float:
        """ONE insert, printed: shell plus infill (module docstring) -- or,
        for a glue fill, the filler's own mass."""
        if self.glue_fill:
            return self.fill_g()
        v = self.volume_mm3
        shell = min(v, self.area_mm2 * SHELL_MM)
        return density_gcc * (shell + INFILL * (v - shell)) / 1000.0

    def centroid_flight_mm(self) -> np.ndarray:
        """Volume centroid, flight mm (starboard)."""
        v = self.flight_verts[self.tris]
        vol = np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2])) / 6.0
        c = v.sum(axis=1) / 4.0
        return (vol[:, None] * c).sum(0) / vol.sum()

    def size_mm(self) -> np.ndarray:
        return self.verts.max(0) - self.verts.min(0)


# ------------------------------------------------------------ triangulation


def _cross2(o, a, b) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _ear_clip_pts(pts: np.ndarray) -> list[tuple[int, int, int]]:
    """Ear clipping that tolerates repeated coordinates -- the two ends of
    a keyhole bridge. A candidate ear is blocked only by a point strictly
    inside it that is not a copy of one of its own corners.

    The blocking test is one vector operation per candidate; the loop in
    Python over every point made a 240-point cap cost two seconds, and the
    search builds one per turning joint per design."""
    n = len(pts)
    idx = list(range(n))
    area2 = float(np.dot(pts[:, 0], np.roll(pts[:, 1], -1))
                  - np.dot(pts[:, 1], np.roll(pts[:, 0], -1)))
    if area2 < 0:
        idx.reverse()
    out = []
    guard = 0
    eps = 1e-12
    k = 0
    while len(idx) > 3 and guard < 4 * n * n:
        guard += 1
        m = len(idx)
        k %= m
        a, b, c = idx[k - 1], idx[k], idx[(k + 1) % m]
        pa, pb, pc = pts[a], pts[b], pts[c]
        if _cross2(pa, pb, pc) > eps:
            others = np.array([p for p in idx if p not in (a, b, c)])
            P = pts[others]
            copy = (np.all(np.abs(P - pa) < 1e-9, 1) | np.all(np.abs(P - pb) < 1e-9, 1)
                    | np.all(np.abs(P - pc) < 1e-9, 1))
            d1 = (pb[0] - pa[0]) * (P[:, 1] - pa[1]) - (pb[1] - pa[1]) * (P[:, 0] - pa[0])
            d2 = (pc[0] - pb[0]) * (P[:, 1] - pb[1]) - (pc[1] - pb[1]) * (P[:, 0] - pb[0])
            d3 = (pa[0] - pc[0]) * (P[:, 1] - pc[1]) - (pa[1] - pc[1]) * (P[:, 0] - pc[0])
            if not np.any((d1 > eps) & (d2 > eps) & (d3 > eps) & ~copy):
                out.append((a, b, c))
                idx.pop(k)
                continue
        k += 1
        if k >= 2 * m and guard > n * m:
            raise ValueError("cap triangulation stuck: the outline self-intersects")
    if len(idx) == 3:
        out.append(tuple(idx))
    return out


def _segments_cross(p1, p2, q1, q2) -> bool:
    d1, d2 = _cross2(q1, q2, p1), _cross2(q1, q2, p2)
    d3, d4 = _cross2(p1, p2, q1), _cross2(p1, p2, q2)
    return (d1 * d2 < -1e-18) and (d3 * d4 < -1e-18)


def cap_triangles(outer: np.ndarray, holes: list[np.ndarray]) -> np.ndarray:
    """Triangles of a planar cap: `outer` (M,2) with `holes` (each (K,2))
    cut out, as indices into concatenate([outer, *holes]).

    Holes are joined to the outline by keyhole bridges -- the hole's
    right-most point to the nearest outline point it can see -- and the
    single polygon that makes is ear-clipped."""
    outer = np.asarray(outer, dtype=float)
    ccw = lambda p: float(np.dot(p[:, 0], np.roll(p[:, 1], -1))   # noqa: E731
                          - np.dot(p[:, 1], np.roll(p[:, 0], -1))) > 0
    ids = list(range(len(outer)))
    if not ccw(outer):
        ids.reverse()
    allpts = [outer]
    base = len(outer)
    edges = [(outer[ids[i]], outer[ids[(i + 1) % len(ids)]]) for i in range(len(ids))]
    for h in holes:
        h = np.asarray(h, dtype=float)
        hid = list(range(base, base + len(h)))
        if ccw(h):                         # holes run clockwise
            hid.reverse()
        for i in range(len(h)):
            edges.append((h[i], h[(i + 1) % len(h)]))
        allpts.append(h)
        base += len(h)
        P = np.concatenate(allpts)
        # right-most hole point, then the nearest visible outline point
        j = max(range(len(hid)), key=lambda k: P[hid[k]][0])
        hp = P[hid[j]]
        best = None
        for k, oid in sorted(enumerate(ids), key=lambda kv: np.linalg.norm(P[kv[1]] - hp)):
            op = P[oid]
            if any(_segments_cross(hp, op, e0, e1) for e0, e1 in edges
                   if not (np.allclose(e0, op) or np.allclose(e1, op)
                           or np.allclose(e0, hp) or np.allclose(e1, hp))):
                continue
            best = k
            break
        if best is None:
            raise ValueError("no bridge from a hole to the outline")
        ring = hid[j:] + hid[:j]
        ids = ids[:best + 1] + ring + [ring[0]] + ids[best:]
    P = np.concatenate(allpts)
    pts = P[ids]
    tri = _ear_clip_pts(pts)
    return np.array([[ids[a], ids[b], ids[c]] for a, b, c in tri], dtype=np.int64)


# ------------------------------------------------------------------ build


def _plain_unit_loop(settings, truncated: bool):
    from .elevons import hinge_x, truncate_loop
    from .vase import thicken_for_nozzle
    xh = hinge_x(settings)

    def unit_loop(st, chord_mm, s_mm, te_scale):
        loop = st.airfoil.coords(settings.contour_points)
        loop = thicken_for_nozzle(loop, chord_mm, settings,
                                  min_te_mm=settings.min_te_mm * te_scale)
        if truncated:
            loop = truncate_loop(loop, xh)
        return loop
    return unit_loop


def _face(plan, frame, s_mm: float, unit_loop) -> np.ndarray:
    """The loft cut by one panel's layer plane at print height s -> (N,3)
    flight mm."""
    from .frames import slice_layers
    c, _ = slice_layers(plan, frame, np.array([float(s_mm)]), unit_loop)
    c = c[0]
    return frame.to_flight(c[:, 0], c[:, 1], np.full(len(c), float(s_mm)))


def _plane_hit(p0, d, q, n) -> np.ndarray:
    """Points p0 (K,3) moved along d (3,) onto the plane through q with
    normal n."""
    t = ((q - p0) @ n) / float(d @ n)
    return p0 + t[:, None] * d[None, :]


def joint_inserts(plan, settings, panels, min_thickness_mm: float | None = None,
                  n_bore: int = 32) -> list[JointInsert]:
    """Every turning joint's insert, root outboard. Joints that mate flat
    need none; a chord-line joint (which interpenetrates rather than
    gaps) gets none and is gated elsewhere."""
    wing = [p for p in panels if getattr(p, "role", "wing") == "wing"
            and p.frame is not None]
    t_min = (2.0 * settings.extrusion_width_mm if min_thickness_mm is None
             else float(min_thickness_mm))
    out = []
    for j, (inner, outer) in enumerate(zip(wing, wing[1:]), start=1):
        f_in, f_out = inner.frame, outer.frame
        if f_out.pivot not in ("upper", "lower") or abs(f_out.kink_deg) < 0.05:
            continue
        eta = float(f_out.eta0)
        truncated = (settings.elevon_chord > 1e-6
                     and eta >= settings.elevon_eta - 1e-9)
        loop = _plain_unit_loop(settings, truncated)
        A = _face(plan, f_out, 0.0, loop)
        B = _face(plan, f_in, f_in.length_mm, loop)
        if A.shape != B.shape:
            raise ValueError(f"joint {j}: faces sampled differently")
        # THICKNESS is the height above face A -- the bed face, so the
        # height the printer builds -- not the distance between matching
        # points, which on a swept wing is mostly the chordwise slide of
        # the skin between the two planes (9.4 mm against a 6.3 mm wedge
        # on micro_fpv).
        na = np.array([0.0, f_out.cos, f_out.sin])
        qa = np.array([0.0, *f_out.origin_yz_mm])
        gap = np.abs((B - qa) @ na)

        elevon_gap = 0.0
        if truncated and eta <= settings.elevon_eta + 1e-9:
            full = _plain_unit_loop(settings, False)
            Af = _face(plan, f_out, 0.0, full)
            Bf = _face(plan, f_in, f_in.length_mm, full)
            from .elevons import hinge_x
            xh = hinge_x(settings)
            st = plan.at(eta)
            aft = Af[:, 0] > st.x_le_m * 1000.0 + xh * st.chord_m * 1000.0
            if aft.any():
                qa_ = np.array([0.0, *f_out.origin_yz_mm])
                na_ = np.array([0.0, f_out.cos, f_out.sin])
                elevon_gap = float(np.abs((Bf[aft] - qa_) @ na_).max())

        # the thin run around the pivot, where the wedge is under t_min
        thin = gap < t_min
        if thin.all() or not thin.any():
            continue
        n = len(gap)
        # rotate so the loop starts just after the thin run ends
        starts = [i for i in range(n) if thin[i] and not thin[(i + 1) % n]]
        if len(starts) != 1:
            raise ValueError(f"joint {j}: the thin part of the wedge is not one "
                             f"run ({len(starts)} runs)")
        r1 = starts[0]
        order = [(r1 + 1 + k) % n for k in range(n)]
        keep = [i for i in order if not thin[i]]
        i_first, i_last = keep[0], keep[-1]
        prev_first = (i_first - 1) % n
        next_last = (i_last + 1) % n

        def cross(i_thin, i_thick):
            u = (t_min - gap[i_thin]) / max(gap[i_thick] - gap[i_thin], 1e-12)
            return (A[i_thin] + u * (A[i_thick] - A[i_thin]),
                    B[i_thin] + u * (B[i_thick] - B[i_thin]))

        a0, b0 = cross(prev_first, i_first)
        a1, b1 = cross(next_last, i_last)
        Ak = np.vstack([a0, A[keep], a1])
        Bk = np.vstack([b0, B[keep], b1])
        M = len(Ak)

        # face B's plane, square to the inner axis at its tip
        nb = np.array([0.0, f_in.cos, f_in.sin])
        qb = np.array([0.0, *f_in.origin_yz_mm]) + f_in.length_mm * nb

        # bores: every tube that crosses the joint
        holes_a, holes_b, bores = [], [], []
        y_joint = float(Ak[:, 1].mean())
        for ti, (x0, z0, sx, sz, reach_y, half, d) in enumerate(settings.spar_lines):
            if reach_y < y_joint + 0.5 * float(gap.max()) + 1.0:
                continue
            r = half - 0.5 * settings.extrusion_width_mm
            dvec = np.array([sx, 1.0, sz])
            dvec /= np.linalg.norm(dvec)
            e1 = np.cross(dvec, [1.0, 0.0, 0.0])
            if np.linalg.norm(e1) < 1e-9:
                e1 = np.cross(dvec, [0.0, 0.0, 1.0])
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(dvec, e1)
            th = np.linspace(0.0, 2.0 * np.pi, n_bore, endpoint=False)
            ring = (np.array([x0, 0.0, z0])[None, :]
                    + r * (np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2))
            ha = _plane_hit(ring, dvec, qa, na)
            hb = _plane_hit(ring, dvec, qb, nb)
            holes_a.append(ha)
            holes_b.append(hb)
            bores.append({"name": f"tube {ti + 1}", "d_mm": float(d), "r_mm": float(r)})

        # 2D coordinates of each cap in its own plane
        def to2d(P, frame, s):
            q = frame.to_print(P)
            return q[:, :2]

        A2 = to2d(Ak, f_out, 0.0)
        B2 = to2d(Bk, f_in, f_in.length_mm)
        HA = [to2d(h, f_out, 0.0) for h in holes_a]
        HB = [to2d(h, f_in, f_in.length_mm) for h in holes_b]

        # Bore walls: least material between each hole and the outline. A
        # tube SEATED against the skin -- which is how the spars are placed
        # -- leaves its bore within a bead of the outline or through it,
        # and a closed hole there is a sliver no printer lays. So such a
        # bore becomes a NOTCH: the outline is cut round the tube, a
        # C-shaped channel with the carbon showing through a groove one
        # wedge long. Both faces are notched and their outlines resampled
        # to matching segments, so the side band still pairs point i with
        # point i. The notch is gated on the NECK it leaves.
        from .vase import _dist_to_segments
        w_bead = settings.extrusion_width_mm
        keep_h = []
        glue = False
        A2_0, B2_0 = A2.copy(), B2.copy()
        for bi, (ha2, hb2) in enumerate(zip(HA, HB)):
            walls = []
            for outline, hole in ((A2, ha2), (B2, hb2)):
                inside = _points_in_polygon(hole, outline)
                w = float(_dist_to_segments(hole, outline).min())
                walls.append(w if inside.all() else -w)
            bores[bi]["wall_mm"] = float(min(walls))
            if min(walls) >= w_bead:
                bores[bi]["kind"] = "bore"
                keep_h.append(bi)
                continue
            bores[bi]["kind"] = "notch"
            try:
                res = _notch_both(A2, B2, ha2, hb2, w_bead)
            except (ValueError, IndexError, FloatingPointError):
                res = None                   # an outline the cut cannot parse
            if res is None or res[2] < 2.0 * w_bead:
                # the tube SEVERS the wedge (or leaves a neck no printer
                # lays): a thin wedge along the skin with the spar in it.
                # That joint is filled with glue, not printed -- declared,
                # charged by weight, and nothing is gated for print.
                glue = True
                break
            A2, B2, neck = res
            bores[bi]["neck_mm"] = float(neck)
        if glue:
            A2, B2, keep_h = A2_0, B2_0, []
            for b in bores:
                b["kind"] = "through the fill"
                b.setdefault("wall_mm", float("nan"))
        holes_a = [holes_a[i] for i in keep_h]
        holes_b = [holes_b[i] for i in keep_h]
        HA = [HA[i] for i in keep_h]
        HB = [HB[i] for i in keep_h]
        # the (possibly notched) outlines back into the flight frame
        if len(A2) != len(Ak):
            Ak = f_out.to_flight(A2[:, 0], A2[:, 1], np.zeros(len(A2)))
            Bk = f_in.to_flight(B2[:, 0], B2[:, 1], np.full(len(B2), f_in.length_mm))
            M = len(Ak)

        tri_a = cap_triangles(A2, HA)
        tri_b = cap_triangles(B2, HB)
        K = [len(h) for h in holes_a]
        # vertex layout: A cap [Ak, holes_a...], then B cap [Bk, holes_b...]
        VA = np.vstack([Ak] + holes_a)
        VB = np.vstack([Bk] + holes_b)
        nA = len(VA)
        tris = [tri_a, tri_b[:, ::-1] + nA]
        side = []
        for k in range(M):                       # skin band + truncation face
            k2 = (k + 1) % M
            side += [(k, k2, nA + k2), (k, nA + k2, nA + k)]
        off = M
        for kk in K:                             # bore walls
            for m in range(kk):
                m2 = (m + 1) % kk
                a_, b_ = off + m, off + m2
                side += [(a_, nA + b_, b_), (a_, nA + a_, nA + b_)]
            off += kk
        tris.append(np.array(side, dtype=np.int64))
        tris = np.vstack(tris)
        flight = np.vstack([VA, VB])

        # consistent winding: caps from the triangulator are CCW in their
        # own 2D frames, whose handedness differs; fix by edge agreement
        tris = _make_consistent(tris)
        # print frame: face A on the bed, as the LEFT half (see module)
        P = f_out.to_print(flight)
        verts = np.stack([-P[:, 0], P[:, 1], -P[:, 2]], 1)
        shift = np.array([0.5 * (verts[:, 0].min() + verts[:, 0].max()),
                          0.5 * (verts[:, 1].min() + verts[:, 1].max()),
                          verts[:, 2].min()])
        verts -= shift
        # the same map as a matrix: print = D P (flight - (0, ay, az)) - shift,
        # D = diag(-1, 1, -1), P = to_print's linear part; composed with the
        # y-mirror it is a proper rotation acting on the LEFT wing's points
        c, s_ = f_out.cos, f_out.sin
        Pm = np.array([[1.0, 0.0, 0.0], [0.0, -s_, c], [0.0, c, s_]])
        Dm = np.diag([-1.0, 1.0, -1.0])
        My = np.diag([1.0, -1.0, 1.0])
        R = Dm @ Pm @ My
        t = -Dm @ Pm @ np.array([0.0, *f_out.origin_yz_mm]) - shift
        from .stl import _orient_outward
        tris = _orient_outward(verts, tris)
        out.append(JointInsert(
            name=f"{plan.name}_insert{j}", joint=j, eta=eta,
            kink_deg=float(f_out.kink_deg), pivot=f_out.pivot,
            flight_verts=flight, verts=verts, tris=tris,
            max_gap_mm=float(gap.max()), crest_mm=float(t_min),
            elevon_gap_mm=elevon_gap, bores=bores,
            arc_a=Ak, arc_b=Bk,
            bore_axes=_kept_axes(settings, y_joint, float(gap.max()), keep_h),
            print_R=R, print_t=t, glue_fill=glue))
    return out


def _resample(poly: np.ndarray, m: int) -> np.ndarray:
    """An open polyline resampled to m points evenly by arc length (ends kept)."""
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-12:
        return np.repeat(poly[:1], m, axis=0)
    t = np.linspace(0.0, s[-1], m)
    return np.stack([np.interp(t, s, poly[:, k]) for k in range(poly.shape[1])], 1)


def _seg_cross(p0, p1, q0, q1):
    """Intersection parameter (t on p, u on q) of two 2D segments, or None."""
    r, s = p1 - p0, q1 - q0
    den = r[0] * s[1] - r[1] * s[0]
    if abs(den) < 1e-15:
        return None
    w = q0 - p0
    t = (w[0] * s[1] - w[1] * s[0]) / den
    u = (w[0] * r[1] - w[1] * r[0]) / den
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return t, u
    return None


def _notch_one(outline: np.ndarray, hole: np.ndarray, grow: float):
    """The open outline with a C-shaped bite round `hole`, grown by `grow`
    so no sliver under a bead is left between tube and skin.
    -> (pre, arc, post, centre, radius) or None if it does not cut
    cleanly (not exactly one entry and one exit on the open arc)."""
    c = hole.mean(0)
    h = c + (hole - c) * (1.0 + grow / max(np.linalg.norm(hole - c, axis=1).mean(), 1e-9))
    K = len(h)
    hits = []
    for i in range(len(outline) - 1):
        for k in range(K):
            x = _seg_cross(outline[i], outline[i + 1], h[k], h[(k + 1) % K])
            if x is not None:
                hits.append((i + x[0], k + x[1]))
    if len(hits) != 2:
        return None
    (se, he), (sx, hx) = sorted(hits)
    E = outline[int(se)] + (se % 1.0) * (outline[min(int(se) + 1, len(outline) - 1)] - outline[int(se)])
    X = outline[int(sx)] + (sx % 1.0) * (outline[min(int(sx) + 1, len(outline) - 1)] - outline[int(sx)])
    pre = np.vstack([outline[:int(se) + 1], E[None]])
    post = np.vstack([X[None], outline[int(sx) + 1:]])

    # the hole's vertices between the two crossings, either way round:
    # entry on hole edge ke (ke -> ke+1), exit on edge kx
    ke, kx = int(np.floor(he)) % K, int(np.floor(hx)) % K
    fwd = [h[(ke + 1 + i) % K] for i in range((kx - ke) % K)]
    bwd = [h[(ke - i) % K] for i in range((ke - kx) % K)]
    cand = [np.vstack([E] + fwd + [X]), np.vstack([E] + bwd + [X])]
    # the way round that runs INSIDE the insert (the outline, closed by its
    # crest cut, is the polygon): test a point half way along each
    inside = [bool(_points_in_polygon(a_[len(a_) // 2][None], outline)[0]) for a_ in cand]
    if inside.count(True) != 1:
        return None
    arc = cand[inside.index(True)]
    return pre, arc, post, c, float(np.linalg.norm(h - c, axis=1).mean())


def _notch_both(A2, B2, ha2, hb2, bead):
    """Notch both faces round one tube, resampled to matching segments so
    point i on face A still pairs with point i on face B.
    -> (A2', B2', neck) or None."""
    ra = _notch_one(A2, ha2, bead)
    rb = _notch_one(B2, hb2, bead)
    if ra is None or rb is None:
        return None
    parts = []
    for sa, sb in zip(ra[:3], rb[:3]):
        m = max(len(sa), len(sb), 3)
        parts.append((_resample(sa, m), _resample(sb, m)))
    A_new = np.vstack([parts[0][0], parts[1][0][1:-1], parts[2][0]])
    B_new = np.vstack([parts[0][1], parts[1][1][1:-1], parts[2][1]])
    # the neck: the least material between the notch and the REST of the
    # outline -- skin more than a notch radius away from the tube
    necks = []
    for (pre, arc, post, c, r), new in ((ra, A_new), (rb, B_new)):
        far = new[np.linalg.norm(new - c, axis=1) > r + 2.0 * bead]
        if len(far) < 2:
            return None
        d = np.linalg.norm(arc[1:-1, None, :] - far[None, :, :], axis=2).min()
        necks.append(float(d))
    return A_new, B_new, min(necks)


def _kept_axes(settings, y_joint: float, max_gap: float, keep: list) -> list:
    """(point, direction, radius) of the bores actually cut: the tubes that
    cross the joint, in `spar_lines` order, then only those whose bore kept
    its wall (`keep` indexes that list)."""
    crossing = [(np.array([x0, 0.0, z0]),
                 np.array([sx, 1.0, sz]) / np.linalg.norm([sx, 1.0, sz]),
                 float(half - 0.5 * settings.extrusion_width_mm))
                for (x0, z0, sx, sz, reach_y, half, d) in settings.spar_lines
                if reach_y >= y_joint + 0.5 * max_gap + 1.0]
    return [crossing[i] for i in keep]


def _points_in_polygon(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    x, y = pts[:, 0][:, None], pts[:, 1][:, None]
    x0, y0 = poly[:, 0][None, :], poly[:, 1][None, :]
    x1, y1 = np.roll(poly[:, 0], -1)[None, :], np.roll(poly[:, 1], -1)[None, :]
    crosses = ((y0 > y) != (y1 > y)) & (
        x < (x1 - x0) * (y - y0) / np.where(y1 != y0, y1 - y0, 1e-18) + x0)
    return crosses.sum(1) % 2 == 1


def _make_consistent(tris: np.ndarray) -> np.ndarray:
    """Flip triangles until every shared edge is used in opposite
    directions by its two faces -- a breadth-first walk from the first."""
    tris = tris.copy()
    from collections import defaultdict, deque
    edge_faces = defaultdict(list)
    for f, t in enumerate(tris):
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            edge_faces[(min(a, b), max(a, b))].append(f)
    seen = np.zeros(len(tris), dtype=bool)
    for start in range(len(tris)):
        if seen[start]:
            continue
        seen[start] = True
        q = deque([start])
        while q:
            f = q.popleft()
            t = tris[f]
            for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
                for g in edge_faces[(min(a, b), max(a, b))]:
                    if g == f or seen[g]:
                        continue
                    u = tris[g]
                    same = any((u[i] == a and u[(i + 1) % 3] == b) for i in range(3))
                    if same:
                        tris[g] = u[::-1]
                    seen[g] = True
                    q.append(g)
    return tris
