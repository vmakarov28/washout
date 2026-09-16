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
    z_strip: np.ndarray    # (S,) strip centre z at the trailing edge
    """Where the strip sheds its wake, vertically. The Trefftz plane is a
    two-dimensional problem in the (y, z) crossflow plane, and dropping z
    -- which this did -- flattens every dihedral and winglet onto the y
    axis. A winglet exists precisely to move shed vorticity OUT of that
    plane, so the planar version scores one at exactly zero benefit."""
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

    return Lattice(
        a=np.concatenate([a, a_m]), b=np.concatenate([b, b_m]),
        cp=np.concatenate([cp, cp_m]), normal=np.concatenate([nrm, nrm_m]),
        dy=np.concatenate([dy, dy]), x_qc=np.concatenate([xqc, xqc]),
        strip=np.concatenate([strip, strip + ns_strips]),
        y_strip=np.concatenate([y_strip, -y_strip]),
        dy_strip=np.concatenate([dy_strip, dy_strip]),
        z_strip=np.concatenate([z_strip, z_strip]),
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
        order = np.argsort(lat.y_strip)
        y = lat.y_strip[order]
        z = lat.z_strip[order]
        g = g_strip[order]
        dyv = lat.dy_strip[order]
        pts = np.stack([y, z], 1)                       # (S,2) strip centres

        # sheet edges: midpoints, with the two ends stepped out by half a
        # strip along the local run of the sheet
        mid = 0.5 * (pts[:-1] + pts[1:])
        first = pts[0] - 0.5 * (pts[1] - pts[0])
        last = pts[-1] + 0.5 * (pts[-1] - pts[-2])
        edge = np.concatenate([first[None, :], mid, last[None, :]], 0)  # (S+1,2)

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
            w = horseshoe(mid, self.lat.a, self.lat.b, np.array([1.0, 0.0, 0.0]))
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
