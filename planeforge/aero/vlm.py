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


def _bound(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Biot-Savart for a finite filament a->b of unit strength.

    p (M,3) field points, a/b (N,3) segment ends -> (M,N,3)."""
    r1 = p[:, None, :] - a[None, :, :]
    r2 = p[:, None, :] - b[None, :, :]
    cr = np.cross(r1, r2)
    den = np.einsum("mnk,mnk->mn", cr, cr)
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


def _semi_infinite(p: np.ndarray, a: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Filament from a to infinity along unit vector d -> (M,N,3)."""
    r = p[:, None, :] - a[None, :, :]
    n = np.linalg.norm(r, axis=2)
    cr = np.cross(d[None, None, :], r)
    den = n * (n - np.einsum("k,mnk->mn", d, r))
    good = den > 1e-12
    coef = np.where(good, 1.0 / np.where(good, den, 1.0), 0.0) / FOURPI
    return coef[..., None] * cr


def horseshoe(p: np.ndarray, a: np.ndarray, b: np.ndarray,
              d: np.ndarray) -> np.ndarray:
    """Bound segment a->b plus the two streamwise trailing legs."""
    return (_bound(p, a, b) + _semi_infinite(p, b, d) - _semi_infinite(p, a, d))


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
    area: float
    mac: float
    span: float

    @property
    def n_panels(self) -> int:
        return len(self.a)


def build_lattice(plan: Planform, ns: int = 24, nc: int = 6) -> Lattice:
    """Cosine-spaced spanwise strips, cosine-spaced chordwise panels.

    Cosine spanwise clustering puts panels where the loading gradient is
    -- the tip, and the blend kink. Uniform spacing on a BWB underloads
    the tip and quietly flatters the induced drag."""
    etas = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, ns + 1)))  # 0..1 edges
    xc_edges = cosine_x(nc + 1)

    def camber_point(eta: float, xc: float) -> np.ndarray:
        st = plan.at(eta)
        yc = float(st.airfoil.camber(np.array([xc]))[0])
        p = np.array([xc - 0.25, yc])
        ang = np.radians(-st.twist_deg)
        ca, sa = np.cos(ang), np.sin(ang)
        rot = np.array([p[0] * ca - p[1] * sa, p[0] * sa + p[1] * ca])
        return np.array([st.x_le_m + (0.25 + rot[0]) * st.chord_m,
                         eta * plan.half_span_m,
                         st.z_le_m + rot[1] * st.chord_m])

    A, B, CP, NRM, DY, XQC, STRIP = [], [], [], [], [], [], []
    y_strip, dy_strip = [], []
    s_index = 0
    for e0, e1 in zip(etas, etas[1:]):
        em = 0.5 * (e0 + e1)
        width = (e1 - e0) * plan.half_span_m
        y_strip.append(em * plan.half_span_m)
        dy_strip.append(width)
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

    # mirror to port. The port half is the starboard half with y negated
    # and the bound vortex ends SWAPPED, so circulation keeps running the
    # same way round the wing.
    flip = np.array([1.0, -1.0, 1.0])
    a_m, b_m = b * flip, a * flip
    cp_m, nrm_m = cp * flip, nrm * flip
    ns_strips = len(y_strip)

    return Lattice(
        a=np.concatenate([a, a_m]), b=np.concatenate([b, b_m]),
        cp=np.concatenate([cp, cp_m]), normal=np.concatenate([nrm, nrm_m]),
        dy=np.concatenate([dy, dy]), x_qc=np.concatenate([xqc, xqc]),
        strip=np.concatenate([strip, strip + ns_strips]),
        y_strip=np.concatenate([y_strip, -y_strip]),
        dy_strip=np.concatenate([dy_strip, dy_strip]),
        area=plan.area_m2, mac=plan.mac_m, span=plan.span_m,
    )


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

    def __init__(self, plan: Planform, ns: int = 24, nc: int = 6):
        self.plan = plan
        self.lat = build_lattice(plan, ns, nc)
        d = np.array([1.0, 0.0, 0.0])          # streamwise trailing legs
        v = horseshoe(self.lat.cp, self.lat.a, self.lat.b, d)
        self.aic = np.einsum("mnk,mk->mn", v, self.lat.normal)
        self._lu = np.linalg.inv(self.aic)     # small and dense; invert once
        self._chord_strip = np.array(
            [plan.at(abs(y) / plan.half_span_m).chord_m
             for y in self.lat.y_strip])

    def solve(self, alpha_deg: float, x_ref_m: float) -> AeroPoint:
        lat = self.lat
        a = np.radians(alpha_deg)
        vinf = np.array([np.cos(a), 0.0, np.sin(a)])       # |V| = 1
        rhs = -(lat.normal @ vinf)
        gamma = self._lu @ rhs

        # --- strip totals ---
        n_strips = len(lat.y_strip)
        g_strip = np.bincount(lat.strip, weights=gamma, minlength=n_strips)

        # --- lift and moment: Kutta-Joukowski on the bound segments ---
        # dF = rho * V x Gamma*dl ; with |V|=1 and rho=1 the coefficients
        # fall straight out of the sums.
        dl = lat.b - lat.a
        dF = gamma[:, None] * np.cross(np.broadcast_to(vinf, dl.shape), dl)
        L = dF[:, 2] * np.cos(a) - dF[:, 0] * np.sin(a)
        CL = 2.0 * L.sum() / lat.area
        # nose-up positive: lift acting aft of the reference pitches down
        Cm = -2.0 * ((lat.x_qc - x_ref_m) * L).sum() / (lat.area * lat.mac)

        # --- induced drag in the Trefftz plane ---
        order = np.argsort(lat.y_strip)
        y = lat.y_strip[order]
        g = g_strip[order]
        dyv = lat.dy_strip[order]
        # trailing sheet strength shed between adjacent strips
        g_pad = np.concatenate([[0.0], g, [0.0]])
        y_edge = np.concatenate([[y[0] - 0.5 * dyv[0]],
                                 0.5 * (y[:-1] + y[1:]),
                                 [y[-1] + 0.5 * dyv[-1]]])
        gam_shed = g_pad[:-1] - g_pad[1:]
        d = y[:, None] - y_edge[None, :]
        w = (gam_shed[None, :] / np.where(np.abs(d) > 1e-9, d, np.inf)).sum(1)
        w /= 2.0 * np.pi
        CDi = -(g * w * dyv).sum() / lat.area
        ar = lat.span**2 / lat.area
        e = (CL**2 / (np.pi * ar * CDi)) if CDi > 1e-12 else 1.0

        cl_local = 2.0 * g_strip / np.maximum(self._chord_strip, 1e-9)
        return AeroPoint(alpha_deg, float(CL), float(CDi), float(Cm),
                         float(e), cl_local, lat.y_strip,
                         self._chord_strip, x_ref_m)

    def sweep(self, alphas, x_ref_m: float) -> list[AeroPoint]:
        return [self.solve(float(al), x_ref_m) for al in alphas]
