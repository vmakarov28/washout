"""A part's smooth surfaces as tensor-product B-splines, from its layers.

## Why from the layers, and why in pieces

The first STEP export lofted the PRINTED contour -- 247 points a loop,
rib slits one bead wide -- and every smooth fit rang through the slits
(one came back 75 m wide on a 240 mm part), so it fell back to a ruled
loft of 9,635 faces. The contour is a manufacturing encoding: one loop,
ribs as detours of the skin. So the surfaces here are fitted to the
panel's PLAIN layers -- the same frames and the same slices as the part
that prints, with no rib detours -- which makes the file the outer
mould line of the printed part and nothing else.

And each section is split into its SMOOTH pieces before fitting, at the
corners the geometry really has: the blunt trailing edge, a hinge cut, an
elevon's nose flat and its chamfer. A cubic fitted across a corner rounds
it; fitted up to it, it does not. One piece is one face in the file.

## The fit

Every piece of every section is fitted by least squares on ONE shared
knot vector with its end points pinned. Shared knots make the sections
compatible, so spanwise skinning never has to re-knot; pinned ends make
two adjacent pieces share their boundary control points, so their faces
share their edges exactly and sew without a gap. Spanwise, the control
points are interpolated between the sections, so the surface passes
through every fitted section's curve.

The knot count is the smallest that holds every section within the
tolerance, and the deviation is MEASURED -- on the fitted sections and on
held-out layers between them -- and reported as a gate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline, make_interp_spline


@dataclass
class Surface:
    """One face's geometry: poles (nu, nv, 3), clamped knot vectors."""

    name: str
    poles: np.ndarray
    u_knots: np.ndarray
    v_knots: np.ndarray
    u_deg: int
    v_deg: int
    fit_dev_mm: float = 0.0

    def evaluate(self, u, v) -> np.ndarray:
        """Points on the surface at parameters (u, v) -> (..., 3)."""
        u = np.atleast_1d(np.asarray(u, dtype=float))
        v = np.atleast_1d(np.asarray(v, dtype=float))
        Bu = BSpline.design_matrix(np.clip(u, 0.0, 1.0), self.u_knots, self.u_deg).toarray()
        Bv = BSpline.design_matrix(np.clip(v, 0.0, 1.0), self.v_knots, self.v_deg).toarray()
        return np.einsum("ai,ijk,aj->ak", Bu, self.poles, Bv)

    @staticmethod
    def knots_and_mults(knots: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        k, m = np.unique(np.round(knots, 12), return_counts=True)
        return k, m.astype(int)


def clamped_knots(n_ctrl: int, deg: int) -> np.ndarray:
    """Uniform interior knots, ends repeated deg+1 times, on [0, 1]."""
    n_int = n_ctrl - deg - 1
    inner = np.linspace(0.0, 1.0, n_int + 2)[1:-1] if n_int > 0 else np.zeros(0)
    return np.concatenate([np.zeros(deg + 1), inner, np.ones(deg + 1)])


def fit_curve(points: np.ndarray, t: np.ndarray, knots: np.ndarray,
              deg: int) -> tuple[np.ndarray, float]:
    """Least-squares poles for one section on fixed knots, ends pinned.

    -> (poles (n, 3), largest distance from a point to its fitted image)."""
    B = BSpline.design_matrix(np.clip(t, 0.0, 1.0), knots, deg).toarray()
    n = B.shape[1]
    if n == 2 or len(points) <= 2:
        # a straight piece: its two ends ARE its poles
        poles = np.array([points[0], points[-1]])
        B2 = BSpline.design_matrix(np.clip(t, 0.0, 1.0), knots, deg).toarray()
        return poles, float(np.linalg.norm(B2 @ poles - points, axis=1).max())
    rhs = points - np.outer(B[:, 0], points[0]) - np.outer(B[:, -1], points[-1])
    inner, *_ = np.linalg.lstsq(B[:, 1:-1], rhs, rcond=None)
    poles = np.vstack([points[0], inner, points[-1]])
    dev = float(np.linalg.norm(B @ poles - points, axis=1).max())
    return poles, dev


def _fit_shared(S: np.ndarray, t: np.ndarray, knots: np.ndarray,
                deg: int) -> tuple[list[np.ndarray], float]:
    """`fit_curve` for K sections that share their parameters -> (K poles,
    worst deviation). S is (K, m, 3)."""
    B = BSpline.design_matrix(np.clip(t, 0.0, 1.0), knots, deg).toarray()
    K, m, _ = S.shape
    first, last = S[:, 0, :], S[:, -1, :]                       # (K, 3)
    rhs = (S - B[:, 0][None, :, None] * first[:, None, :]
           - B[:, -1][None, :, None] * last[:, None, :])        # (K, m, 3)
    rhs = np.transpose(rhs, (1, 0, 2)).reshape(m, 3 * K)
    inner, *_ = np.linalg.lstsq(B[:, 1:-1], rhs, rcond=None)
    inner = inner.reshape(-1, K, 3).transpose(1, 0, 2)           # (K, n-2, 3)
    P = np.concatenate([first[:, None, :], inner, last[:, None, :]], axis=1)
    dev = float(np.linalg.norm(np.einsum("mi,kij->kmj", B, P) - S, axis=2).max())
    return list(P), dev


def _param(points: np.ndarray, by_index: bool) -> np.ndarray:
    if by_index or len(points) < 3:
        return np.linspace(0.0, 1.0, len(points))
    d = np.sqrt(np.linalg.norm(np.diff(points, axis=0), axis=1))   # centripetal
    s = np.concatenate([[0.0], np.cumsum(d)])
    return s / max(s[-1], 1e-12)


def _chord_dev(points: np.ndarray) -> float:
    """Largest distance from a point to the chord between the ends."""
    if len(points) <= 2:
        return 0.0
    a, b = points[0], points[-1]
    d = b - a
    L2 = float(np.dot(d, d))
    if L2 < 1e-24:
        return float("inf")
    t = np.clip((points - a) @ d / L2, 0.0, 1.0)
    return float(np.linalg.norm(a + np.outer(t, d) - points, axis=1).max())


def _straight(points: np.ndarray, tol_mm: float) -> bool:
    return _chord_dev(points) <= tol_mm


def fit_piece(name: str, sections: list[np.ndarray], v: np.ndarray,
              tol_mm: float = 0.01, deg: int = 3,
              counts=(8, 12, 16, 24, 32, 48, 64, 96, 128)) -> Surface:
    """One smooth piece across all its sections -> one Surface.

    `sections` are (m_k, 3) point runs in the same orientation on every
    section; `v` the spanwise parameter of each, in [0, 1]. A piece whose
    sections all have the same point count is parametrised by INDEX --
    the loops come off one cosine grid, so index is the same chord
    station on every section and dense where the nose curves -- anything
    else centripetally."""
    same = len({len(s) for s in sections}) == 1
    if all(_straight(s, tol_mm) for s in sections):
        # A straight piece is ruled: its two ends ARE its poles. The
        # elevon's floor and chamfer are straight; fitted as cubics, a
        # chamfer whose sections were sampled differently escalated to
        # interpolating every point and zigzagged 2.5 mm between them --
        # demon1's elevon0 tip cap crossed itself.
        u_deg, knots = 1, np.array([0.0, 0.0, 1.0, 1.0])
        poles_u = [np.array([s[0], s[-1]]) for s in sections]
        dev = max(_chord_dev(s) for s in sections)
    else:
        u_deg = deg
        ts = [_param(s, same) for s in sections]
        m_min = min(len(s) for s in sections)
        for n in counts:
            n = max(min(n, m_min), u_deg + 1)
            knots = clamped_knots(n, u_deg)
            if same:
                # one basis for every section: one factorisation, all the
                # sections solved as columns of one right-hand side
                poles_u, dev = _fit_shared(np.stack(sections), ts[0], knots, u_deg)
            else:
                fits = [fit_curve(s, t, knots, u_deg) for s, t in zip(sections, ts)]
                poles_u, dev = [f[0] for f in fits], max(f[1] for f in fits)
            if dev <= tol_mm or n >= m_min:
                break
    P = np.stack(poles_u)                               # (K, n, 3)
    K = len(sections)
    v_deg = min(3, K - 1)
    spl = make_interp_spline(np.asarray(v, dtype=float), P, k=v_deg, axis=0)
    return Surface(name=name, poles=np.transpose(spl.c, (1, 0, 2)).copy(),
                   u_knots=knots, v_knots=np.asarray(spl.t, dtype=float),
                   u_deg=u_deg, v_deg=v_deg, fit_dev_mm=float(dev))


# ------------------------------------------------------ a part's pieces


def _turns_deg(pts: np.ndarray) -> np.ndarray:
    """Turning angle at each interior vertex of an open run."""
    a = np.diff(pts[:, :2], axis=0)
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-15)
    cosang = np.clip(np.einsum("ij,ij->i", a[:-1], a[1:]), -1.0, 1.0)
    return np.degrees(np.arccos(cosang))


def piece_bounds(loop: np.ndarray, n_upper: int | None,
                 kink_deg: float = 30.0,
                 every_corner: bool = False) -> list[tuple[int, int]] | None:
    """Where one closed layer loop breaks into smooth pieces, as inclusive
    index runs, the last wrapping back to 0.

    A wing loop is its skin -- trailing edge round the nose and back, or
    hinge cut to hinge cut -- and the closing segment, which is the blunt
    trailing edge or the cut face. An elevon's is its upper skin, its nose
    flat, its lower surface -- broken again wherever the chamfer or the
    nozzle's floor puts a corner in it -- and its trailing edge.

    `every_corner` breaks the lower surface at EVERY corner of `kink_deg`
    or more, not only the chamfer's; `fit_part` uses it when every section
    agrees on how many there are."""
    N = len(loop)
    if n_upper is None:
        return [(0, N - 1), (N - 1, 0)]
    n = int(n_upper)
    # The lower surface is broken where the chamfer meets the skin -- a
    # corner of 60-70 degrees on every section, so it is split there and
    # only there. The nozzle's floor puts smaller corners near the nose on
    # the thin outboard sections and not on the thick ones; splitting at
    # those too gave a different number of pieces along the part, and a
    # piece has to exist on every section to be a face.
    # The chamfer's junction is the corner FURTHEST from the nose: the
    # floor's corners, when a section has them, are all nearer. Taking the
    # sharpest instead picked a floor corner on some of the trainer's
    # sections and the junction on others, and a surface skinned through
    # pieces that are not the same piece wandered 9.6 mm between them.
    turns = _turns_deg(loop[n:])
    big = np.flatnonzero(turns >= kink_deg)
    if big.size == 0:
        return [(0, n - 1), (n - 1, n), (n, N - 1), (N - 1, 0)]
    if every_corner:
        # A corner can fall between two vertices one short segment apart,
        # and then its turn is split between them -- demon1's elevon floor
        # turns 28.5 + 28.2 degrees on seven layers and 45 + 12 on the
        # rest. So a corner is a run of adjacent vertices each turning a
        # quarter of the kink or more, whose turns SUM to the kink; it is
        # cut at the sharpest vertex of the run.
        cuts, i = [], 0
        while i < len(turns):
            if turns[i] < 0.25 * kink_deg:
                i += 1
                continue
            j = i
            while j + 1 < len(turns) and turns[j + 1] >= 0.25 * kink_deg:
                j += 1
            if turns[i:j + 1].sum() >= kink_deg:
                cuts.append(n + 1 + i + int(np.argmax(turns[i:j + 1])))
            i = j + 1
    else:
        cuts = [n + 1 + int(big[-1])]
    ends = [n] + cuts + [N - 1]
    return ([(0, n - 1), (n - 1, n)] + list(zip(ends, ends[1:])) + [(N - 1, 0)])


def sharpen(loop: np.ndarray, n_upper: int | None, kink_deg: float = 30.0) -> np.ndarray:
    """The loop with a vertex put ON every lower-surface corner that falls
    between two grid points.

    The elevon's nose floor meets its chamfer at a corner no grid point is
    placed on, so along the part it slides from one vertex to the next,
    splitting its turn between two of them in between. A piece boundary
    that hops a grid point from section to section skins a surface that
    misses the held-out layers by 0.08 mm on demon1. The true corner is
    where the straight run into the pair meets the straight run out of
    it; it is INSERTED between them, so every printed point is kept and
    the corner is one vertex on every section -- the chamfer's own fix
    (elevons.py), made here for a corner the print does not place."""
    if n_upper is None:
        return loop
    n = int(n_upper)
    turns = _turns_deg(loop[n:])
    big = np.flatnonzero(turns >= kink_deg)
    # the chamfer's junction -- the furthest corner -- is a vertex already
    last = int(big[-1]) if big.size else len(turns)
    out, done = [loop[:n + 1]], n
    pairs = []
    for i in range(last - 1):
        if turns[i] < 0.25 * kink_deg or turns[i] < max(turns[max(i - 1, 0)], turns[i + 1]):
            continue                      # not the sharper vertex of a corner
        j = i - 1 if i > 0 and turns[i - 1] > turns[i + 1] else i + 1
        if j >= last or turns[j] < 0.5 or turns[i] + turns[j] < kink_deg:
            continue                      # on a vertex already, or no corner
        pairs.append(min(i, j))
    for i in pairs:
        a, b = n + 1 + i, n + 2 + i
        d1, d2 = loop[a, :2] - loop[a - 1, :2], loop[b + 1, :2] - loop[b, :2]
        M = np.array([d1, -d2]).T
        if abs(np.linalg.det(M)) < 1e-12:
            continue
        s, _ = np.linalg.solve(M, loop[b, :2] - loop[a - 1, :2])
        p = loop[a - 1].copy()
        p[:2] = loop[a - 1, :2] + s * d1
        seg = loop[b, :2] - loop[a, :2]
        t = float(np.dot(p[:2] - loop[a, :2], seg) / max(np.dot(seg, seg), 1e-24))
        if not 0.0 < t < 1.0:
            continue
        out += [loop[done + 1:a + 1], p[None, :]]
        done = a
    out.append(loop[done + 1:])
    return np.vstack(out)


def section_runs(loop3: np.ndarray, bounds) -> list[np.ndarray]:
    out = []
    for a, b in bounds:
        if b == 0 and a == len(loop3) - 1:
            out.append(loop3[[a, 0]])
        else:
            out.append(loop3[a:b + 1])
    return out


def _dist_to_iso(surf: Surface, pts: np.ndarray, v: float,
                 n_coarse: int = 400, n_fine: int = 41) -> float:
    """Largest distance from `pts` to the surface's u-curve at v: a coarse
    pass to find each point's neighbourhood, then a fine one inside it."""
    uu = np.linspace(0.0, 1.0, n_coarse)
    curve = surf.evaluate(uu, np.full(n_coarse, v))
    near = np.linalg.norm(pts[:, None, :] - curve[None, :, :], axis=2).argmin(1)
    h = 1.0 / (n_coarse - 1)
    u = np.clip(uu[near][:, None] + np.linspace(-h, h, n_fine)[None, :], 0.0, 1.0)
    c = surf.evaluate(u.ravel(), np.full(u.size, v)).reshape(len(pts), n_fine, 3)
    return float(np.linalg.norm(c - pts[:, None, :], axis=2).min(1).max())


def piece_names(n_upper: int | None, n_pieces: int, truncated: bool) -> list[str]:
    if n_upper is None:
        return ["skin", "hinge cut" if truncated else "trailing edge"]
    # the lower surface, nose to trailing edge: any floor pieces, then the
    # chamfer, then the skin
    n_lower = n_pieces - 3
    lower = ["lower skin"] if n_lower == 1 else ["hinge chamfer", "lower skin"]
    floors = n_lower - len(lower)
    lower = [("nose floor" if i == 0 else f"nose floor {i + 1}")
             for i in range(floors)] + lower
    return ["upper skin", "nose"] + lower + ["trailing edge"]


def fit_part(stack, every_mm: float = 6.0, tol_mm: float = 0.02,
             truncated: bool = False,
             target_mm: float = 0.04) -> tuple[list[Surface], dict]:
    """Every smooth piece of one PLAIN printed part, in its print frame.

    -> (surfaces, report). The report carries the fit deviation on the
    fitted sections and on HELD-OUT layers half way between them, which is
    the number the loft-fidelity gate reads: a surface that matches the
    sections it was fitted to and wanders between them would pass the
    first and fail the second.

    The spacing REFINES itself: while the held-out layers are further than
    `target_mm` from the surface, the sections are halved. The loft's
    edges are piecewise quadratic -- C1, with their curvature jumping at
    the planform's knots -- and a C2 cubic through sections 6 mm apart
    misses such a jump by a quarter of a millimetre on micro_fpv's centre
    body, while 3 mm holds it to 0.05. Where the wing is plain the coarse
    spacing is kept, and the file stays small."""
    floor = 2.0 * (stack.z_step_mm or 0.25)
    # Which corners to split at is decided by the same measurement: every
    # corner first, then the chamfer's only (see _fit_part_once). The
    # first that holds the target is kept; if none does, the closest.
    best = None
    for every_corner in ((True, False) if stack.n_upper is not None else (False,)):
        every = float(every_mm)
        while True:
            surfs, rep = _fit_part_once(stack, every, tol_mm, truncated, every_corner)
            if rep["held_out_dev_mm"] <= target_mm or every <= floor:
                break
            every = max(0.5 * every, floor)
        rep["every_mm"] = every
        if rep["held_out_dev_mm"] <= target_mm:
            return surfs, rep
        if best is None or rep["held_out_dev_mm"] < best[1]["held_out_dev_mm"]:
            best = (surfs, rep)
    return best


def _fit_part_once(stack, every_mm, tol_mm, truncated, every_corner=True):
    z = stack.z_mm
    L = len(z)
    k_step = max(int(round(every_mm / max(stack.z_step_mm or 0.25, 1e-9))), 1)
    idx = list(range(0, L, k_step))
    if idx[-1] != L - 1:
        idx.append(L - 1)
    if len(idx) < 4:
        idx = sorted(set(np.linspace(0, L - 1, min(L, 4)).round().astype(int)))
    loops3 = [np.column_stack([stack.contours[k], np.full(len(stack.contours[k]), z[k])])
              for k in range(L)]
    if every_corner:
        loops3 = [sharpen(lp, stack.n_upper) for lp in loops3]
    # Split at every corner when every section has the same ones: a cubic
    # fitted ACROSS a corner rings, and demon1's elevon has a 45-degree
    # floor corner inside its chamfer piece on all 225 layers -- the fit
    # folded back over the nose flat and the root cap came out an invalid,
    # self-intersecting face. (micro_fpv's has the same corner and its
    # fit across it holds 0.05 mm where the split one does not; fit_part
    # keeps whichever measures inside the target.) Where the floor's
    # corners come and go, split at the chamfer only; where the chamfer's
    # does too, fit the lower surface whole rather than pair up pieces
    # that are not the same.
    held = [(a + b) // 2 for a, b in zip(idx, idx[1:]) if b - a > 1]
    # (the held-out layers must agree too before splitting at every corner,
    # so the gate measures every piece on them)
    modes = ((30.0, True),) if every_corner else ()
    for kink, every in modes + ((30.0, False), (1e9, False)):
        check = idx + held if every else idx
        bounds = [piece_bounds(loops3[k], stack.n_upper, kink, every) for k in check]
        if len({len(b) for b in bounds}) == 1:
            break
    bounds = bounds[:len(idx)]
    runs = [section_runs(loops3[k], b) for k, b in zip(idx, bounds)]
    n_pieces = len(runs[0])
    v = (z[idx] - z[0]) / max(z[-1] - z[0], 1e-12)
    names = piece_names(stack.n_upper, n_pieces, truncated)
    surfs = [fit_piece(names[p], [r[p] for r in runs], v, tol_mm=tol_mm)
             for p in range(n_pieces)]
    # held out: the layer half way between each pair of fitted ones,
    # measured as the distance from each of its points to the surface's
    # iso-curve at that layer's own v -- found by a fine search around the
    # point's own parameter, not by the nearest of a few hundred samples,
    # whose spacing alone would read 0.2 mm on a 450 mm skin
    worst_held = 0.0
    for k in held:
        vk = (z[k] - z[0]) / max(z[-1] - z[0], 1e-12)
        rr = section_runs(loops3[k], piece_bounds(loops3[k], stack.n_upper, kink, every))
        if len(rr) != n_pieces:
            continue
        for s_, pts in zip(surfs, rr):
            worst_held = max(worst_held, _dist_to_iso(s_, pts, vk))
    report = {"sections": len(idx), "pieces": n_pieces,
              "fit_dev_mm": max(s_.fit_dev_mm for s_ in surfs),
              "held_out_dev_mm": worst_held,
              "poles": int(sum(s_.poles.shape[0] * s_.poles.shape[1] for s_ in surfs))}
    return surfs, report
