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

    # --- caps: ladder between upper and lower at matching x ---
    n = (N + 1) // 2
    up = (n - 1) - np.arange(n)            # LE -> TE along the upper surface
    lo = (n - 1) + np.arange(n)            # LE -> TE along the lower surface
    q = np.arange(n - 1)
    quads = np.stack([up[q], up[q + 1], lo[q + 1], lo[q]], -1)
    cap = np.concatenate([
        np.stack([quads[:, 0], quads[:, 1], quads[:, 2]], -1),
        np.stack([quads[:, 0], quads[:, 2], quads[:, 3]], -1),
    ])
    cap = cap[(cap[:, 0] != cap[:, 1]) & (cap[:, 1] != cap[:, 2])
              & (cap[:, 0] != cap[:, 2])]          # drop the LE degenerate

    # Both cap boundary edges run counter to the side ring's cyclic
    # direction (the ladder walks the upper surface LE->TE, which is
    # DECREASING index), so `cap` is already wound as the bottom. The top
    # is its mirror. Getting this backwards costs exactly 2N inconsistent
    # windings and a mesh the slicer will quietly patch for you.
    bottom = cap
    top = cap[:, ::-1] + (L - 1) * N

    tris = np.concatenate([side, bottom, top]).astype(np.int64)
    tris = _orient_outward(verts, tris)
    return verts, tris


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
