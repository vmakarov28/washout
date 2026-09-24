"""The inside of a printed wall: each layer's outline offset one bead in.

A vase part is a single bead, so the part is a WALL, not the solid its
outer surface encloses. The wall's inner surface is fitted exactly as the
outer one is, from layer outlines -- offset inward by the extrusion
width. The offset is OpenCASCADE's planar one, with intersection joins:
a vertex-normal offset written first made loops at the leading edge
wherever the points were dense, and pushed points into each other at the
hinge corners. The result is resampled onto the outline's own structure,
so the fit sees the same point correspondence it sees outside.
"""

from __future__ import annotations

import numpy as np
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeFace
from OCP.BRepOffsetAPI import BRepOffsetAPI_MakeOffset
from OCP.GeomAbs import GeomAbs_JoinType
from OCP.BRepAdaptor import BRepAdaptor_CompCurve
from OCP.GCPnts import GCPnts_QuasiUniformAbscissa
from OCP.TopAbs import TopAbs_WIRE
from OCP.TopoDS import TopoDS
from OCP.gp import gp_Pnt
from . import brep


def _loop_len(P, closed=True):
    d = np.linalg.norm(np.diff(np.vstack([P, P[:1]]) if closed else P, axis=0), axis=1)
    return d


def _solid_aft(P: np.ndarray, d: float, margin: float = 0.15,
               n_upper: int | None = None) -> np.ndarray:
    """The outline with its aft end cut where the section first gets
    thinner than two walls: aft of that the two beads merge and the part
    is solid, and the planar offset fails on the collapse rather than
    returning the shorter loop. Aft is +X, as every panel prints.

    Point 0 is the upper aft corner and point N-1 the lower one (Selig
    order round the leading edge), so the cut loop keeps that structure."""
    if n_upper is None:
        i_le = int(np.argmin(P[:, 0]))
        up = P[: i_le + 1][::-1]        # LE -> aft
        lo = P[i_le:]                   # LE -> aft
    else:                               # an elevon: a cut nose, no shared LE
        up = P[:n_upper][::-1]
        lo = P[n_upper:]
    x_front = max(up[0, 0], lo[0, 0])
    xs = np.linspace(x_front, min(up[-1, 0], lo[-1, 0]), 400)
    yu = np.interp(xs, up[:, 0], up[:, 1])
    yl = np.interp(xs, lo[:, 0], lo[:, 1])
    thin = np.flatnonzero((yu - yl) < 2.0 * d + margin)
    thin = thin[xs[thin] > xs[0] + 0.5 * (xs[-1] - xs[0])]
    if thin.size == 0:
        return P
    xc = float(xs[thin[0]])
    up_k = up[up[:, 0] < xc]
    lo_k = lo[lo[:, 0] < xc]
    up_c = np.array([xc, float(np.interp(xc, up[:, 0], up[:, 1]))])
    lo_c = np.array([xc, float(np.interp(xc, lo[:, 0], lo[:, 1]))])
    if n_upper is None:
        return np.vstack([up_c, up_k[::-1], lo_k[1:], lo_c])
    return np.vstack([up_c, up_k[::-1], lo_k, lo_c])


def offset_loop(P: np.ndarray, d: float, n_dense: int = 1500,
                n_upper: int | None = None) -> np.ndarray:
    """Offset a closed polygon (N, 2) inward by d with OpenCASCADE's planar
    offset (sharp, intersection joins), and resample it onto P's own
    structure: point 0 and point N-1 (the aft corners) map to the offset
    images of those corners, and every other point at the same fraction
    of the path length between them."""
    N = len(P)
    P_full = P
    P = _solid_aft(P, d, n_upper=n_upper)
    # Segments of a few hundredths of a mm (they occur at the blunt TE's
    # corners) make the planar offset return nothing. They are merged for
    # the offset only; the resampling below uses P's own points.
    keep = [0]
    for i in range(1, len(P)):
        if np.linalg.norm(P[i] - P[keep[-1]]) >= 0.1:
            keep.append(i)
    if np.linalg.norm(P[keep[-1]] - P[0]) < 0.1 and len(keep) > 3:
        keep.pop()
    # Rounded joins first. Inward, a join only acts at a concave corner, so
    # rounding is what a bead does anyway; the sharp (intersection) joins
    # failed outright on 59 of 153 layers of gen10's centre body, where
    # near-collinear points make the line intersections ill-conditioned.
    # Then sharp joins, then the same outline at half the points.
    # An elevon's cut nose has real corners (the chamfer) that the fit
    # splits the loop at; rounded inner corners made that split flip from
    # layer to layer and the pieces overlapped. It tries sharp joins first.
    arc, sharp = GeomAbs_JoinType.GeomAbs_Arc, GeomAbs_JoinType.GeomAbs_Intersection
    order = ((P[keep], arc), (P[keep], sharp), (P[keep][::2], arc))
    if n_upper is not None:
        order = ((P[keep], sharp), (P[keep][::2], sharp), (P[keep], arc))
    wires = []
    for pts, jt in order:
        poly = BRepBuilderAPI_MakePolygon()
        for x, y in pts:
            poly.Add(gp_Pnt(float(x), float(y), 0.0))
        poly.Close()
        try:
            face = BRepBuilderAPI_MakeFace(poly.Wire(), True).Face()
            mo = BRepOffsetAPI_MakeOffset(face, jt)
            mo.Perform(-float(d))
            if mo.IsDone():
                wires = brep._shapes(mo.Shape(), TopAbs_WIRE)
        except Exception:                            # noqa: BLE001
            wires = []
        if wires:
            break
    if not wires:
        raise RuntimeError("planar offset failed")
    best = None
    for wr in wires:
        cc = BRepAdaptor_CompCurve(TopoDS.Wire(wr))
        ab = GCPnts_QuasiUniformAbscissa(cc, n_dense)
        Q = np.array([[cc.Value(ab.Parameter(i)).X(), cc.Value(ab.Parameter(i)).Y()]
                      for i in range(1, ab.NbPoints() + 1)])
        area = 0.5 * abs(np.sum(Q[:, 0] * np.roll(Q[:, 1], -1) - np.roll(Q[:, 0], -1) * Q[:, 1]))
        if best is None or area > best[0]:
            best = (area, Q)
    Q = best[1]
    if np.linalg.norm(Q[0] - Q[-1]) < 1e-9:
        Q = Q[:-1]
    # same winding as P
    sa = lambda R: np.sum(R[:, 0] * np.roll(R[:, 1], -1) - np.roll(R[:, 0], -1) * R[:, 1])
    if sa(Q) * sa(P) < 0:
        Q = Q[::-1]
    # The corners the fit splits the loop at: the two aft ones, and for an
    # elevon the two at its cut nose. Each maps to the nearest point of the
    # offset loop, and every run of points between two corners is
    # resampled at its own arclength fractions onto the matching run --
    # a single run for the whole loop drifted an elevon's nose corners
    # into its skins, and the fit of that skin blew up.
    cut = P
    P = P_full
    N = len(P)
    corners = [0, N - 1] if n_upper is None else [0, n_upper - 1, n_upper, N - 1]
    targets = [cut[0], cut[-1]] if n_upper is None else [
        cut[0], P[n_upper - 1], P[n_upper], cut[-1]]
    ia = int(np.argmin(np.linalg.norm(Q - targets[0], axis=1)))
    Q = np.roll(Q, -ia, axis=0)
    idx = [0] + [int(np.argmin(np.linalg.norm(Q - t, axis=1))) for t in targets[1:]]
    # the aft corners can collapse onto one sample (a ~1 mm blunt TE
    # offsets to ~0.1 mm): the last run then goes all the way round
    if idx[-1] <= idx[-2]:
        idx[-1] = len(Q) - 1
    out = np.empty_like(P)
    for (ca, cb), (qa, qb) in zip(zip(corners, corners[1:]), zip(idx, idx[1:])):
        seg_p = P[ca:cb + 1]
        seg_q = Q[qa:qb + 1] if qb > qa else Q[qa:qa + 2]
        sp = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(seg_p, axis=0), axis=1))])
        sq = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(seg_q, axis=0), axis=1))])
        sp = sp / sp[-1] if sp[-1] > 0 else np.linspace(0, 1, len(sp))
        sq = sq / sq[-1] if sq[-1] > 0 else np.linspace(0, 1, len(sq))
        out[ca:cb + 1] = np.stack([np.interp(sp, sq, seg_q[:, 0]),
                                    np.interp(sp, sq, seg_q[:, 1])], 1)
    return out


def offset_stack(C: np.ndarray, d: float, n_upper: int | None = None) -> np.ndarray:
    return np.stack([offset_loop(P, d, n_upper=n_upper) for P in C])
