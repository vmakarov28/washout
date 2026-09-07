"""Skin a layer stack into a watertight triangle mesh, write binary STL.

Watertightness is checked, not assumed: a slicer handed a mesh with a
hole will silently produce a part with a hole in it, and in vase mode
that is a wing with a slot down the spar bay. `manifold_report` counts
every undirected edge; a closed orientable surface has each one used
exactly twice, once in each direction.

The caps are triangulated as a ladder between the upper and lower
surfaces at matching chord stations, not as a fan from the centroid. An
airfoil with reflex is not star-shaped about its centroid, so a fan
produces inverted triangles right where the trailing edge matters.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from .vase import LayerStack


def skin(stack: LayerStack) -> tuple[np.ndarray, np.ndarray]:
    """Layer stack -> (vertices (V,3), triangles (T,3) indices)."""
    contours, z = stack.contours, stack.z_mm
    L, N, _ = contours.shape
    verts = np.concatenate(
        [contours.reshape(-1, 2),
         np.repeat(z, N)[:, None]], axis=1).astype(np.float64)

    # --- side wall: one quad ring per layer gap, wrapping at the TE ---
    i = np.arange(N)
    j = (i + 1) % N
    k = np.arange(L - 1)[:, None] * N
    a, b = k + i, k + j                    # this layer
    c, d = a + N, b + N                    # next layer up
    side = np.concatenate([
        np.stack([a, b, d], -1).reshape(-1, 3),
        np.stack([a, d, c], -1).reshape(-1, 3),
    ])

    # --- caps ---
    # Ear clipping, not the old upper/lower ladder. The ladder assumed the
    # section is two graphs over a shared x, which a RIBBED contour is
    # emphatically not: its rib detours double back in x. Ear clipping
    # triangulates any simple polygon, so it covers both cases and there
    # is no reason to keep two code paths.
    cap = _ear_clip(contours[0])

    # _ear_clip normalises its output to counter-clockwise, which runs
    # WITH the side ring's cyclic direction -- the opposite of the old
    # ladder, whose boundary ran against it. So the reversal that used to
    # belong on the top now belongs on the bottom. Getting this backwards
    # costs exactly 2N inconsistent windings (482 for a 241-point
    # section) and a mesh the slicer will quietly patch for you.
    bottom = cap[:, ::-1]
    top = cap + (L - 1) * N

    tris = np.concatenate([side, bottom, top]).astype(np.int64)
    tris = _orient_outward(verts, tris)
    return verts, tris


def _ear_clip(poly: np.ndarray) -> np.ndarray:
    """Triangulate a simple polygon -> (T,3) indices into its own points.

    O(n^2) and unglamorous, but it makes no assumption about the shape,
    which is what a rib detour requires. Winding is normalised first so
    the returned triangles are consistently counter-clockwise."""
    n = len(poly)
    idx = list(range(n))
    area2 = float(np.dot(poly[:, 0], np.roll(poly[:, 1], -1))
                  - np.dot(poly[:, 1], np.roll(poly[:, 0], -1)))
    if area2 < 0:
        idx.reverse()

    def cross(o, a, b):
        return ((poly[a][0] - poly[o][0]) * (poly[b][1] - poly[o][1])
                - (poly[a][1] - poly[o][1]) * (poly[b][0] - poly[o][0]))

    def inside(a, b, c, p):
        d1 = cross(a, b, p)
        d2 = cross(b, c, p)
        d3 = cross(c, a, p)
        return (d1 >= 0 and d2 >= 0 and d3 >= 0) or (d1 <= 0 and d2 <= 0 and d3 <= 0)

    out: list[tuple[int, int, int]] = []
    guard = 0
    while len(idx) > 3 and guard < 4 * n:
        guard += 1
        clipped = False
        for k in range(len(idx)):
            a, b, c = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            if cross(a, b, c) <= 0:                 # reflex or collinear
                continue
            if any(inside(a, b, c, p) for p in idx
                   if p not in (a, b, c)):
                continue
            out.append((a, b, c))
            idx.pop(k)
            clipped = True
            break
        if not clipped:                             # numerically stuck: fan
            for k in range(1, len(idx) - 1):
                out.append((idx[0], idx[k], idx[k + 1]))
            idx = idx[:3]
            break
    if len(idx) == 3:
        out.append((idx[0], idx[1], idx[2]))
    return np.array(out, dtype=np.int64)


def _orient_outward(verts: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Flip the whole mesh if the signed volume came out negative.

    The construction is already consistent (every triangle wound the same
    way relative to its neighbours); only the global sense can be wrong,
    and the sign of the divergence-theorem volume is exactly that."""
    v = verts[tris]
    vol = np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2])).sum() / 6.0
    return tris[:, ::-1] if vol < 0 else tris


def manifold_report(tris: np.ndarray) -> dict:
    """Edge bookkeeping: the only honest test of a printable mesh."""
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    und = np.sort(e, axis=1)
    _, counts = np.unique(und, axis=0, return_counts=True)
    # each undirected edge exactly twice, and the two uses opposite in
    # direction (which np.unique on the DIRECTED edges detects as all
    # counts == 1)
    _, dcounts = np.unique(e, axis=0, return_counts=True)
    return {
        "triangles": int(len(tris)),
        "edges": int(len(counts)),
        "non_manifold_edges": int((counts != 2).sum()),
        "inconsistent_windings": int((dcounts != 1).sum()),
        "watertight": bool((counts == 2).all() and (dcounts == 1).all()),
    }


def volume_mm3(verts: np.ndarray, tris: np.ndarray) -> float:
    v = verts[tris]
    return float(abs(np.einsum("ij,ij->i", v[:, 0],
                               np.cross(v[:, 1], v[:, 2])).sum()) / 6.0)


def write_stl(path: str | Path, verts: np.ndarray, tris: np.ndarray,
              header: str = "planeforge") -> Path:
    """Binary STL. Normals written explicitly; some slicers trust them."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    v = verts[tris].astype(np.float32)
    nrm = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = np.divide(nrm, ln, out=np.zeros_like(nrm), where=ln > 0)

    # 50 bytes per facet: normal + 3 vertices (12 float32 = 48 B) then a
    # 2-byte attribute word left at zero.
    rec = np.zeros((len(tris), 50), dtype=np.uint8)
    body = np.concatenate([nrm.astype(np.float32)[:, None, :], v], axis=1)
    rec[:, :48] = body.reshape(len(tris), 12).view(np.uint8)

    with path.open("wb") as f:
        f.write(header.encode("ascii", "replace")[:80].ljust(80, b"\0"))
        f.write(struct.pack("<I", len(tris)))
        f.write(rec.tobytes())
    return path


def export(stack: LayerStack, path: str | Path) -> dict:
    """Skin, verify, write -- the one call the pipeline uses."""
    verts, tris = skin(stack)
    rep = manifold_report(tris)
    if not rep["watertight"]:
        raise ValueError(f"mesh is not watertight: {rep}")
    write_stl(path, verts, tris, header=f"planeforge {stack.name}")
    rep["volume_cm3"] = volume_mm3(verts, tris) / 1000.0
    rep["path"] = str(path)
    return rep
