"""Vortex-lattice method on the mean camber surface.

This is the in-the-loop evaluator. The LBM tunnel is the truth, but a
D3Q19 run of a whole aircraft is hours and an optimizer needs thousands
of evaluations, so the search runs on a VLM that answers in milliseconds
and the tunnel is spent where it changes decisions (see aero/tiers.py).

Panels sit on the MEAN CAMBER SURFACE, not on a flat plate. That is not
a refinement here, it is the entire point: a tailless aircraft trims on
its own section pitching moment, so a flat-plate lattice would report
Cm0 = 0 for every design and cheerfully hand back an untrimmable wing.
Reflex has to be visible to the solver that judges it.

Conventions: x aft, y starboard, z up. Cm is positive nose-up, taken
about a reference point, and is non-dimensionalised on MAC.

Known limits, stated because the objective weighs them: inviscid, so no
stall and no separation -- CL_max comes from the section data, never from
here; and the trailing legs are straight and streamwise, so post-stall or
heavily rolled cases are outside its reach.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geom.cst import cosine_x
from ..geom.planform import Planform

FOURPI = 4.0 * np.pi


# --------------------------------------------------------------- filaments


def _bound(p: np.ndarray, a: np.ndarray, b: np.ndarray,
           core: float = 0.0) -> np.ndarray:
    """Biot-Savart for a finite filament a->b of unit strength.

    p (M,3) field points, a/b (N,3) segment ends -> (M,N,3).

    `core` > 0 regularises the filament's line: the squared distance to it
    becomes d^2 + core^2 (a Scully-type core). The wing never uses one --
    core 0 is the classical kernel to the bit -- and the fin junction
    does; see VLM.__init__."""
    r1 = p[:, None, :] - a[None, :, :]
    r2 = p[:, None, :] - b[None, :, :]
    cr = np.cross(r1, r2)
    den = np.einsum("mnk,mnk->mn", cr, cr)
    if core > 0.0:
        seg = b - a
        den = den + core * core * np.einsum("nk,nk->n", seg, seg)[None, :]
    n1 = np.linalg.norm(r1, axis=2)
    n2 = np.linalg.norm(r2, axis=2)
    r0 = (b - a)[None, :, :]
    k = (np.einsum("mnk,mnk->mn", r0, r1) / np.maximum(n1, 1e-12)
         - np.einsum("mnk,mnk->mn", r0, r2) / np.maximum(n2, 1e-12))
    # the filament's own line is a singularity; a finite core radius keeps
    # the AIC matrix conditioned when a control point lands near a bound
    # segment, which happens on every high-taper tip panel.
    good = den > 1e-12
    coef = np.where(good, k / np.where(good, den, 1.0), 0.0) / FOURPI
    return coef[..., None] * cr


def _semi_infinite(p: np.ndarray, a: np.ndarray, d: np.ndarray,
                   core: float = 0.0) -> np.ndarray:
    """Filament from a to infinity along unit vector d -> (M,N,3).

    With a core, the same kernel in its (1 + cos)/d_perp^2 form -- equal to
    1/(n (n - d.r)) identically -- with d_perp^2 + core^2 underneath."""
    r = p[:, None, :] - a[None, :, :]
    n = np.linalg.norm(r, axis=2)
    cr = np.cross(d[None, None, :], r)
    if core > 0.0:
        perp2 = np.einsum("mnk,mnk->mn", cr, cr)
        cos = np.einsum("k,mnk->mn", d, r) / np.maximum(n, 1e-12)
        coef = (1.0 + cos) / (perp2 + core * core) / FOURPI
        return coef[..., None] * cr
    den = n * (n - np.einsum("k,mnk->mn", d, r))
    good = den > 1e-12
    coef = np.where(good, 1.0 / np.where(good, den, 1.0), 0.0) / FOURPI
    return coef[..., None] * cr


def horseshoe(p: np.ndarray, a: np.ndarray, b: np.ndarray,
              d: np.ndarray, core: float = 0.0) -> np.ndarray:
    """Bound segment a->b plus the two streamwise trailing legs."""
    return (_bound(p, a, b, core) + _semi_infinite(p, b, d, core)
            - _semi_infinite(p, a, d, core))


def _influence(p: np.ndarray, lat: "Lattice", d: np.ndarray) -> np.ndarray:
    """(N, N, 3) velocity each of the lattice's horseshoes induces at `p`,
    ONE POINT PER PANEL in panel order (control points, or bound-vortex
    midpoints), so the first `n_wing_panels` rows are wing points.

    Wing on wing and fin on fin are the classical kernel. ACROSS the
    junction -- a fin's vortices at a wing point, a wing's at a fin point
    -- the kernel is cored at `lat.fin_core`: the wing's last cosine strip
    puts its control points a fraction of a millimetre from the fin's root
    vortices, and the unregularised kernel there swung one design's fin
    side-force slope from -0.24 to +11.8 as the fin was refined. Coring
    the fin's influence on ITSELF as well was tried and is wrong: it
    shrinks each panel's own induced velocity, which is the fin's aspect
    ratio, and put a fin's slope past the reflection-plane limit."""
    if not lat.n_fin_strips:
        return horseshoe(p, lat.a, lat.b, d)
    nw, c = lat.n_wing_panels, lat.fin_core
    out = np.empty((len(p), lat.n_panels, 3))
    out[:nw, :nw] = horseshoe(p[:nw], lat.a[:nw], lat.b[:nw], d)
    out[:nw, nw:] = horseshoe(p[:nw], lat.a[nw:], lat.b[nw:], d, c)
    out[nw:, :nw] = horseshoe(p[nw:], lat.a[:nw], lat.b[:nw], d, c)
    out[nw:, nw:] = horseshoe(p[nw:], lat.a[nw:], lat.b[nw:], d)
    return out


# ------------------------------------------------------------------ lattice


@dataclass
class Lattice:
    """Panelled mean-camber surface of the FULL aircraft (both halves)."""

    a: np.ndarray          # (N,3) inboard end of each bound vortex
    b: np.ndarray          # (N,3) outboard end
    cp: np.ndarray         # (N,3) control points, 3/4 chord
    normal: np.ndarray     # (N,3) camber-surface normals
    dy: np.ndarray         # (N,) spanwise width
    x_qc: np.ndarray       # (N,) x of the bound-vortex midpoint
    strip: np.ndarray      # (N,) spanwise strip index
    y_strip: np.ndarray    # (S,) strip centre y
    dy_strip: np.ndarray   # (S,)
    z_strip: np.ndarray    # (S,) strip centre z at the trailing edge
    """Where the strip sheds its wake, vertically. The Trefftz plane is a
    two-dimensional problem in the (y, z) crossflow plane, and dropping z
    -- which this did -- flattens every dihedral and winglet onto the y
    axis. A winglet exists precisely to move shed vorticity OUT of that
    plane, so the planar version scores one at exactly zero benefit."""
    area: float
    mac: float
    span: float
    # Tip fins, when the lattice carries them. Their panels come AFTER
    # every wing panel and their strips are numbered after every wing
    # strip, so the wing's arrays above -- y_strip, dy_strip, z_strip --
    # are exactly what they were without fins, and every consumer of the
    # spanwise loading (tip-stall gate, strip-theory drag, structure)
    # still reads the wing and only the wing.
    fin_rel: np.ndarray | None = None
    """(F, 2, 2): each fin strip's start and end in the Trefftz (y, z)
    plane, RELATIVE to the wing-tip node it hangs from, ordered along its
    bound vortex (a -> b). F counts both fins."""
    fin_side: np.ndarray | None = None
    """(F,): +1 starboard, -1 port -- which tip node the strip hangs from."""
    fin_core: float = 0.0
    """Vortex core for every interaction involving a fin: half a fin strip."""

    @property
    def n_panels(self) -> int:
        return len(self.a)

    @property
    def n_wing_panels(self) -> int:
        return int((self.strip < len(self.y_strip)).sum())

    @property
    def n_wing_strips(self) -> int:
        return len(self.y_strip)

    @property
    def n_fin_strips(self) -> int:
        return 0 if self.fin_side is None else len(self.fin_side)


def _fin_panels(fins, z_root: float, nf: int, nc: int):
    """Starboard fin panels: vertical plates in the plane y = y_tip.

    THE ROOT IS FLAT, at `z_root` (the tip's trailing-edge height, where
    the wing's wake and the fin's meet). A fin's strips are stacked in z,
    so its chordwise lines must run streamwise at constant z: its trailing
    legs are straight and horizontal, and a chordwise line that climbs --
    following the tip's camber line was tried -- carries the legs of its
    forward panels down through the control points of the strips below.
    On fpv_micro_fpv_s4's swept fins that put control points 0.0-0.2 mm
    from a leg, and the fin's side-force slope went -0.19, -0.15, -0.01,
    -0.34 as it was refined. (A wing can follow its camber: its strips are
    stacked in y, and camber moves its legs out of the surface, not
    across strips.)

    The wing's own tip vortices leave at the tip camber heights, a few
    millimetres around `z_root`, and pass through the fin's lowest strip;
    that is what the junction core in `_influence` is sized to.

    Strips are UNIFORM up each part. Clustering them at the free tip, as
    the wing clusters its span, made tip strips 0.2 mm tall under 6 mm
    chordwise panels, and the circulation there alternated in sign from
    strip to strip.

    Each part (above the chord line, and below it when `below_frac` > 0)
    tapers from the root chord to the tip chord over its own height, with
    the leading edge swept, exactly as `TipFins.outline` draws it; heights
    are measured from the camber line.

    Every bound vortex runs UP (a -> b along +z), bottom of the fin to
    top, through the root; the port copy's is reversed by the mirror.
    -> panel lists and the per-strip (h0, h1) heights, bottom to top."""
    t = np.tan(np.radians(fins.sweep_deg))
    hu = (1.0 - fins.below_frac) * fins.height_m
    hd = fins.below_frac * fins.height_m
    cr, ct, x0, y = fins.root_chord_m, fins.tip_chord_m, fins.x_root_le_m, fins.y_m
    xc_edges = cosine_x(nc + 1)
    parts = []
    if hd > 1e-6:
        n_lo = max(2, int(round(nf * hd / fins.height_m)))
        parts.append((hd, n_lo, -1.0))
    parts.append((hu, max(2, nf - (parts[0][1] if parts else 0)), +1.0))

    def pt(h: float, frac: float, h_part: float) -> np.ndarray:
        ah = abs(h)
        c = cr + (ct - cr) * ah / h_part
        return np.array([x0 + ah * t + frac * c, y, z_root + h])

    A, B, CP, NRM, DY, XQC, strips = [], [], [], [], [], [], []
    for h_part, n_s, sgn in parts:
        e = np.linspace(0.0, 1.0, n_s + 1)
        hs = sgn * e * h_part
        if sgn < 0:
            hs = hs[::-1]                    # bottom -> root, running up
        for h0, h1 in zip(hs, hs[1:]):
            hm = 0.5 * (h0 + h1)
            strips.append((h0, h1))
            for c0, c1 in zip(xc_edges, xc_edges[1:]):
                q = c0 + 0.25 * (c1 - c0)
                A.append(pt(h0, q, h_part))
                B.append(pt(h1, q, h_part))
                CP.append(pt(hm, c0 + 0.75 * (c1 - c0), h_part))
                NRM.append(np.array([0.0, 1.0, 0.0]))
                DY.append(abs(h1 - h0))
                XQC.append(0.5 * (A[-1][0] + B[-1][0]))
    return A, B, CP, NRM, DY, XQC, strips


def build_lattice(plan: Planform, ns: int = 24, nc: int = 6,
                  fins=None, nf: int = 12, ncf: int = 6) -> Lattice:
    """Cosine-spaced spanwise strips, cosine-spaced chordwise panels.

    Cosine spanwise clustering puts panels where the loading gradient is
    -- the tip, and the blend kink. Uniform spacing on a BWB underloads
    the tip and quietly flatters the induced drag.

    `fins` (aero/fins.py TipFins) adds a vertical plate at each tip: `nf`
    strips up its height and `ncf` panels along its chord."""
    etas = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, ns + 1)))  # 0..1 edges
    xc_edges = cosine_x(nc + 1)

    # plan.at() blends two CST sections and allocates; build_lattice asks
    # for the same span station five times per chordwise panel, so at
    # 32x8 that is ~1300 blends of which ~33 are distinct. Cache them.
    _stations: dict[float, object] = {}

    def station(eta: float):
        st = _stations.get(eta)
        if st is None:
            st = _stations[eta] = plan.at(eta)
        return st

    def camber_point(eta: float, xc: float) -> np.ndarray:
        st = station(eta)
        yc = float(st.airfoil.camber(np.array([xc]))[0])
        p = np.array([xc - 0.25, yc])
        ang = np.radians(-st.twist_deg)
        ca, sa = np.cos(ang), np.sin(ang)
        rot = np.array([p[0] * ca - p[1] * sa, p[0] * sa + p[1] * ca])
        return np.array([st.x_le_m + (0.25 + rot[0]) * st.chord_m,
                         eta * plan.half_span_m,
                         st.z_le_m + rot[1] * st.chord_m])

    A, B, CP, NRM, DY, XQC, STRIP = [], [], [], [], [], [], []
    y_strip, dy_strip, z_strip = [], [], []
    s_index = 0
    for e0, e1 in zip(etas, etas[1:]):
        em = 0.5 * (e0 + e1)
        width = (e1 - e0) * plan.half_span_m
        y_strip.append(em * plan.half_span_m)
        dy_strip.append(width)
        z_strip.append(float(camber_point(em, 1.0)[2]))
        for c0, c1 in zip(xc_edges, xc_edges[1:]):
            # bound vortex at the panel's quarter chord, control point at
            # its three-quarter chord -- the classical arrangement that
            # makes a single chordwise panel reproduce thin-aerofoil theory
            qc0, qc1 = c0 + 0.25 * (c1 - c0), c1 - c1 + c0 + 0.25 * (c1 - c0)
            xa = camber_point(e0, c0 + 0.25 * (c1 - c0))
            xb = camber_point(e1, c0 + 0.25 * (c1 - c0))
            cpt = camber_point(em, c0 + 0.75 * (c1 - c0))
            # normal from the panel's own camber surface
            p_le = camber_point(em, c0)
            p_te = camber_point(em, c1)
            chordwise = p_te - p_le
            spanwise = camber_point(e1, 0.5 * (c0 + c1)) - camber_point(e0, 0.5 * (c0 + c1))
            n = np.cross(chordwise, spanwise)
            n /= max(np.linalg.norm(n), 1e-12)
            if n[2] < 0:
                n = -n
            A.append(xa); B.append(xb); CP.append(cpt); NRM.append(n)
            DY.append(width); XQC.append(0.5 * (xa[0] + xb[0])); STRIP.append(s_index)
        s_index += 1

    a = np.array(A); b = np.array(B); cp = np.array(CP); nrm = np.array(NRM)
    dy = np.array(DY); xqc = np.array(XQC); strip = np.array(STRIP)
    y_strip = np.array(y_strip); dy_strip = np.array(dy_strip)
    z_strip = np.array(z_strip)

    # mirror to port. The port half is the starboard half with y negated
    # and the bound vortex ends SWAPPED, so circulation keeps running the
    # same way round the wing.
    flip = np.array([1.0, -1.0, 1.0])
    a_m, b_m = b * flip, a * flip
    cp_m, nrm_m = cp * flip, nrm * flip
    ns_strips = len(y_strip)

    lat = Lattice(
        a=np.concatenate([a, a_m]), b=np.concatenate([b, b_m]),
        cp=np.concatenate([cp, cp_m]), normal=np.concatenate([nrm, nrm_m]),
        dy=np.concatenate([dy, dy]), x_qc=np.concatenate([xqc, xqc]),
        strip=np.concatenate([strip, strip + ns_strips]),
        y_strip=np.concatenate([y_strip, -y_strip]),
        dy_strip=np.concatenate([dy_strip, dy_strip]),
        z_strip=np.concatenate([z_strip, z_strip]),
        area=plan.area_m2, mac=plan.mac_m, span=plan.span_m,
    )
    if fins is None:
        return lat

    z_root = float(camber_point(1.0, 1.0)[2])
    fa, fb, fcp, fn, fdy, fx, fstrips = _fin_panels(fins, z_root, nf, ncf)
    fa, fb, fcp, fn = map(np.array, (fa, fb, fcp, fn))
    fdy, fx = np.array(fdy), np.array(fx)
    nfs = len(fstrips)
    fstrip = np.repeat(np.arange(nfs), ncf) + 2 * ns_strips
    # the port fin: y negated, ends swapped, so its vortices run DOWN
    fa_m, fb_m = fb * flip, fa * flip
    rel_s = np.array([[[0.0, h0], [0.0, h1]] for h0, h1 in fstrips])
    rel_p = rel_s[:, ::-1, :]                 # port: top -> bottom
    lat.a = np.concatenate([lat.a, fa, fa_m])
    lat.b = np.concatenate([lat.b, fb, fb_m])
    lat.cp = np.concatenate([lat.cp, fcp, fcp * flip])
    lat.normal = np.concatenate([lat.normal, fn, fn * flip])
    lat.dy = np.concatenate([lat.dy, fdy, fdy])
    lat.x_qc = np.concatenate([lat.x_qc, fx, fx])
    lat.strip = np.concatenate([lat.strip, fstrip, fstrip + nfs])
    lat.fin_rel = np.concatenate([rel_s, rel_p])
    lat.fin_side = np.concatenate([np.ones(nfs), -np.ones(nfs)])
    # The junction core covers the two distances at which the lattices
    # meet, and nothing more: the wing's last control points sit half a
    # strip width inboard of the fin's plane, and the wing's tip vortices
    # leave at the tip's camber heights, spread about the flat fin root.
    # The spread does not shrink as either lattice is refined, so neither
    # may the core. Sizing it to half a FIN strip instead was tried: on a
    # planar end plate it cost 10% of the end-plate gain at h/b = 0.2,
    # because the coupling it smears is the end-plate effect itself.
    tip_b = b[np.abs(b[:, 1] - plan.half_span_m) < 1e-12]
    spread = float(np.abs(tip_b[:, 2] - z_root).max()) if len(tip_b) else 0.0
    lat.fin_core = max(0.5 * float(dy_strip.min()), spread)
    return lat


# ------------------------------------------------------------------ Trefftz


def trefftz_cdi(pts: np.ndarray, edge: np.ndarray, g: np.ndarray,
                fin_rel: np.ndarray | None, fin_side: np.ndarray | None,
                g_fin: np.ndarray | None, area: float) -> float:
    """Induced drag of a wake sheet that may branch, in the Trefftz plane.

    `pts` (S,2) and `edge` (S+1,2) are the wing chain's strip centres and
    nodes, port to starboard, with `g` (S,) its strip circulations; fins
    hang from the chain's two END nodes. Each strip runs from a start node
    to an end node along its bound vortex, and a node sheds the jump in
    circulation across it,

        shed(node) = sum(G of strips ending there) - sum(G starting there),

    which on a plain chain is G[i-1] - G[i], the familiar sheet. The
    downwash normal to each strip -- the +90 degree turn of its running
    direction, the same convention the chain uses -- then gives

        CDi = -sum(G w ds) / S

    over every strip, wing and fin alike."""
    S = len(g)
    nodes = [p for p in edge]
    starts = list(range(S))
    ends = list(range(1, S + 1))
    cent = [p for p in pts]
    gam = list(g)
    if fin_rel is not None and len(fin_rel):
        junction = {+1.0: edge[-1], -1.0: edge[0]}     # starboard, port
        for rel, side, gf in zip(fin_rel, fin_side, g_fin):
            p0 = junction[float(side)] + rel[0]
            p1 = junction[float(side)] + rel[1]
            ids = []
            for p in (p0, p1):
                hit = next((i for i, q in enumerate(nodes)
                            if abs(q[0] - p[0]) < 1e-12 and abs(q[1] - p[1]) < 1e-12),
                           None)
                if hit is None:
                    nodes.append(np.asarray(p, dtype=float))
                    hit = len(nodes) - 1
                ids.append(hit)
            starts.append(ids[0])
            ends.append(ids[1])
            cent.append(0.5 * (p0 + p1))
            gam.append(float(gf))
    nodes = np.array(nodes)
    cent = np.array(cent)
    gam = np.array(gam)
    starts, ends = np.array(starts), np.array(ends)

    shed = np.zeros(len(nodes))
    np.add.at(shed, ends, gam)
    np.add.at(shed, starts, -gam)

    seg = nodes[ends] - nodes[starts]
    ds = np.linalg.norm(seg, axis=1)
    tang = seg / np.maximum(ds, 1e-12)[:, None]
    nrm2 = np.stack([-tang[:, 1], tang[:, 0]], 1)

    d = cent[:, None, :] - nodes[None, :, :]
    r2 = (d ** 2).sum(-1)
    r2 = np.where(r2 > 1e-18, r2, np.inf)
    k = shed[None, :] / (2.0 * np.pi * r2)
    v_y = (k * -d[:, :, 1]).sum(1)
    v_z = (k * d[:, :, 0]).sum(1)
    w = v_y * nrm2[:, 0] + v_z * nrm2[:, 1]
    return float(-(gam * w * ds).sum() / area)


# -------------------------------------------------------------------- solve


@dataclass
class AeroPoint:
    alpha_deg: float
    CL: float
    CDi: float
    Cm: float               # about x_ref, positive nose-up
    e_oswald: float
    cl_local: np.ndarray    # (S,) section lift coefficient per strip
    y_strip: np.ndarray
    chord_strip: np.ndarray
    x_ref_m: float


class VLM:
    """One lattice, reusable across angles of attack.

    The AIC matrix does not depend on alpha, so it is factorised once and
    every extra angle costs a triangular solve. A trim search that walks
    five angles therefore costs barely more than one."""

    def __init__(self, plan: Planform, ns: int = 24, nc: int = 6, fins=None,
                 nf: int = 12, ncf: int = 6):
        self.plan = plan
        self.fins = fins
        self.lat = build_lattice(plan, ns, nc, fins=fins, nf=nf, ncf=ncf)
        d = np.array([1.0, 0.0, 0.0])          # streamwise trailing legs
        v = _influence(self.lat.cp, self.lat, d)
        self.aic = np.einsum("mnk,mk->mn", v, self.lat.normal)
        self._lu = np.linalg.inv(self.aic)     # small and dense; invert once
        self._chord_strip = np.array(
            [plan.at(abs(y) / plan.half_span_m).chord_m
             for y in self.lat.y_strip])

    def _sheet(self, g_strip: np.ndarray):
        """The wing's wake in the Trefftz plane: strip centres (S,2), nodes
        (S+1,2) and circulations, port to starboard.

        Nodes are midpoints between centres, the two ends stepped out by
        half a strip along the local run of the sheet."""
        lat = self.lat
        order = np.argsort(lat.y_strip)
        pts = np.stack([lat.y_strip[order], lat.z_strip[order]], 1)
        mid = 0.5 * (pts[:-1] + pts[1:])
        first = pts[0] - 0.5 * (pts[1] - pts[0])
        last = pts[-1] + 0.5 * (pts[-1] - pts[-2])
        edge = np.concatenate([first[None, :], mid, last[None, :]], 0)
        return pts, edge, g_strip[order]

    @property
    def has_fins(self) -> bool:
        """The fins are IN this lattice: their side force, and what they do
        to the wing's tip loading and its wake, come out of the solve."""
        return self.lat.n_fin_strips > 0

    def solve(self, alpha_deg: float, x_ref_m: float) -> AeroPoint:
        lat = self.lat
        a = np.radians(alpha_deg)
        vinf = np.array([np.cos(a), 0.0, np.sin(a)])       # |V| = 1
        rhs = -(lat.normal @ vinf)
        gamma = self._lu @ rhs

        # --- strip totals ---
        n_strips = len(lat.y_strip)
        g_all = np.bincount(lat.strip, weights=gamma,
                            minlength=n_strips + lat.n_fin_strips)
        g_strip = g_all[:n_strips]

        # --- lift and moment: Kutta-Joukowski on the bound segments ---
        # dF = rho * V x Gamma*dl ; with |V|=1 and rho=1 the coefficients
        # fall straight out of the sums.
        dl = lat.b - lat.a
        dF = gamma[:, None] * np.cross(np.broadcast_to(vinf, dl.shape), dl)
        L = dF[:, 2] * np.cos(a) - dF[:, 0] * np.sin(a)
        CL = 2.0 * L.sum() / lat.area
        # nose-up positive: lift acting aft of the reference pitches down
        Cm = -2.0 * ((lat.x_qc - x_ref_m) * L).sum() / (lat.area * lat.mac)

        # --- induced drag in the Trefftz plane, in TWO dimensions ---
        # The wake rolls off into the (y, z) crossflow plane, and the
        # sheet there is a curve, not a line: dihedral tilts it and a
        # winglet turns it through most of a right angle. The previous
        # version used y alone, which projects that curve flat -- so a
        # winglet shed its vorticity at exactly the same place as no
        # winglet, and scored exactly the same induced drag. With z
        # carried through, this reduces EXACTLY to the old expression
        # when the wing is planar (z = 0 makes the normal +z and the
        # kernel 1/dy), which is what keeps the elliptic-wing e = 0.99
        # calibration honest.
        pts, edge, g = self._sheet(g_strip)

        seg = edge[1:] - edge[:-1]                      # (S,2) per-strip run
        ds = np.linalg.norm(seg, axis=1)
        tang = seg / np.maximum(ds, 1e-12)[:, None]
        # normal to the sheet, rotated +90 deg: (ty, tz) -> (-tz, ty)
        nrm2 = np.stack([-tang[:, 1], tang[:, 0]], 1)

        g_pad = np.concatenate([[0.0], g, [0.0]])
        gam_shed = g_pad[:-1] - g_pad[1:]               # (S+1,)

        # 2D point-vortex kernel: v = gam/(2 pi r^2) * (-dz, dy)
        d = pts[:, None, :] - edge[None, :, :]          # (S, S+1, 2)
        r2 = (d ** 2).sum(-1)
        r2 = np.where(r2 > 1e-18, r2, np.inf)
        k = gam_shed[None, :] / (2.0 * np.pi * r2)
        v_y = (k * -d[:, :, 1]).sum(1)
        v_z = (k * d[:, :, 0]).sum(1)
        w = v_y * nrm2[:, 0] + v_z * nrm2[:, 1]
        CDi = -(g * w * ds).sum() / lat.area
        if lat.n_fin_strips:
            # The fins make the sheet a branched curve -- a T at each tip --
            # which the chain above cannot hold, so the whole sheet is
            # redone as nodes and strips. The chain's own nodes and
            # centres are kept exactly, so the planar part is the same
            # geometry; test_validation pins that this general form
            # reproduces the chain to round-off on a wing without fins.
            CDi = trefftz_cdi(pts, edge, g, lat.fin_rel, lat.fin_side,
                              g_all[n_strips:], lat.area)
        ar = lat.span**2 / lat.area
        e = (CL**2 / (np.pi * ar * CDi)) if CDi > 1e-12 else 1.0

        cl_local = 2.0 * g_strip / np.maximum(self._chord_strip, 1e-9)
        return AeroPoint(alpha_deg, float(CL), float(CDi), float(Cm),
                         float(e), cl_local, lat.y_strip,
                         self._chord_strip, x_ref_m)

    def sweep(self, alphas, x_ref_m: float) -> list[AeroPoint]:
        return [self.solve(float(al), x_ref_m) for al in alphas]

    def trim_alpha(self, x_ref_m: float,
                   bounds: tuple[float, float] = (-6.0, 12.0),
                   tol: float = 1e-7) -> float | None:
        """Angle where Cm about x_ref is zero, by secant iteration.

        Bisection was costing 40 iterations x 2 solves. The lattice is a
        LINEAR system whose right-hand side depends on alpha only through
        (cos a, sin a), so over a +/-12 degree bracket Cm is very nearly
        straight and a secant lands in three or four solves. The bracket
        is still checked first, so a design with no trim point is still
        reported as having none rather than being extrapolated a root.
        """
        lo, hi = bounds
        f_lo, f_hi = self.solve(lo, x_ref_m).Cm, self.solve(hi, x_ref_m).Cm
        if f_lo * f_hi > 0:
            return None
        a0, a1, f0, f1 = lo, hi, f_lo, f_hi
        for _ in range(12):
            if abs(f1 - f0) < 1e-15:
                break
            a2 = a1 - f1 * (a1 - a0) / (f1 - f0)
            if not (lo <= a2 <= hi):            # secant left the bracket
                a2 = 0.5 * (a0 + a1)            # fall back to bisection
            f2 = self.solve(a2, x_ref_m).Cm
            a0, f0, a1, f1 = a1, f1, a2, f2
            if abs(f2) < tol:
                break
        return float(a1)

    # ------------------------------------------------------ lateral derivatives

    def _midpoint_influence(self) -> np.ndarray:
        """(N, N, 3) velocity each horseshoe induces at each bound-vortex
        midpoint, per unit circulation. Built on first use: trim never
        needs it, and most designs in a search never get far enough to."""
        w = getattr(self, "_w_mid", None)
        if w is None:
            mid = 0.5 * (self.lat.a + self.lat.b)
            w = _influence(mid, self.lat, np.array([1.0, 0.0, 0.0]))
            self._w_mid = w
        return w

    def lateral_derivatives(self, alpha_deg: float, ref_m: np.ndarray,
                            axes: str = "stability", d_beta_deg: float = 1.0,
                            d_rate: float = 0.02) -> np.ndarray:
        """Sideslip and rotary derivatives of the wing, per radian.

            rows  CY, Cl, Cn          cols  beta, p_hat, r_hat
            p_hat = p b / 2V; conventional signs, so Cl_p < 0 damps roll

        Sideslip and body rates are imposed as ONSET flow on the same
        lattice -- the influence matrix does not change, so each case is
        one back-substitution -- and derivatives are central differences.
        Both halves are panelled, so antisymmetric loading needs nothing
        new.

        Forces include the INDUCED velocity at each bound vortex, not just
        the freestream. solve() gets away without it because lift is first
        order in circulation; side force and yawing moment are not, and a
        yawing moment is largely an induced-drag difference between the two
        halves -- exactly the term a freestream-only Kutta-Joukowski sum
        throws away.

        `axes` is "stability" (x along the relative wind) or "body". The
        lateral equations in dynamics.py are stability-axis equations and
        must be fed stability-axis derivatives: the first prototype mixed
        the two, which at the trainer's 7.8 degree trim leaked roll
        stiffness into yaw and made every aircraft look violently unstable.
        The test suite checks the two sets are an exact rotation.

        Geometry is in the lattice frame (x aft, y starboard, z up);
        moments come back in the conventional x-forward, z-down sense."""
        lat = self.lat
        mid = 0.5 * (lat.a + lat.b)
        dl = lat.b - lat.a
        w_mid = self._midpoint_influence()
        s_ref, b = lat.area, lat.span
        a = np.radians(alpha_deg)
        ca, sa = np.cos(a), np.sin(a)
        if axes == "stability":
            roll_ax, yaw_ax = np.array([-ca, 0.0, -sa]), np.array([sa, 0.0, -ca])
        elif axes == "body":
            roll_ax, yaw_ax = np.array([-1.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0])
        else:
            raise ValueError(f"axes must be 'stability' or 'body', got {axes!r}")
        ref = np.asarray(ref_m, dtype=float)

        def coeffs(beta: float, p_hat: float, r_hat: float) -> np.ndarray:
            vinf = np.array([ca * np.cos(beta), -np.sin(beta), sa * np.cos(beta)])
            omega = (2.0 * p_hat / b) * roll_ax + (2.0 * r_hat / b) * yaw_ax

            def onset(pts: np.ndarray) -> np.ndarray:
                return vinf[None, :] - np.cross(np.broadcast_to(omega, pts.shape),
                                                pts - ref)

            gamma = self._lu @ (-np.einsum("nk,nk->n", lat.normal, onset(lat.cp)))
            v_local = onset(mid) + np.einsum("mnk,n->mk", w_mid, gamma)
            dF = gamma[:, None] * np.cross(v_local, dl)
            force = dF.sum(axis=0)
            moment = np.cross(mid - ref, dF).sum(axis=0)
            return np.array([force[1] / (0.5 * s_ref),
                             moment @ roll_ax / (0.5 * s_ref * b),
                             moment @ yaw_ax / (0.5 * s_ref * b)])

        db = np.radians(d_beta_deg)
        out = np.empty((3, 3))
        out[:, 0] = (coeffs(db, 0.0, 0.0) - coeffs(-db, 0.0, 0.0)) / (2.0 * db)
        out[:, 1] = (coeffs(0.0, d_rate, 0.0) - coeffs(0.0, -d_rate, 0.0)) / (2.0 * d_rate)
        out[:, 2] = (coeffs(0.0, 0.0, d_rate) - coeffs(0.0, 0.0, -d_rate)) / (2.0 * d_rate)
        return out
