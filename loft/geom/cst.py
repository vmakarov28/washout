"""CST (Kulfan class/shape transformation) airfoil parametrization.

Why CST and not a spline through points: the optimizer needs a *small*,
*always-valid* design vector. A CST section with n+1 Bernstein
coefficients per surface is smooth by construction, has a round leading
edge and a controllable trailing edge for free, and every point in
coefficient space is a real airfoil. Fitting is linear least squares
(the class function factors out), so any Selig .dat becomes a design
vector exactly, not approximately-by-optimizer.

    y(x) = C(x) * S(x) + x * dz_te
    C(x) = x^N1 * (1-x)^N2          N1=0.5 round LE, N2=1.0 sharp TE
    S(x) = sum_i  A_i * B_{i,n}(x)  Bernstein basis

Reflex -- the aft-upward camber a tailless flying wing needs for
Cm0 > 0 -- lives in the last two or three coefficients of each surface.
It is not a special case here; it is just a region of the same space.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy.special import comb

N1_ROUND_LE = 0.5
N2_SHARP_TE = 1.0


def bernstein(n: int, x: np.ndarray) -> np.ndarray:
    """Bernstein basis of order n evaluated at x -> (n+1, len(x))."""
    i = np.arange(n + 1)[:, None]
    x = np.asarray(x, dtype=float)[None, :]
    return comb(n, i) * x**i * (1.0 - x) ** (n - i)


def cst_y(x: np.ndarray, a: np.ndarray, dz_te: float = 0.0) -> np.ndarray:
    """One CST surface: class function times shape function, plus the
    linear trailing-edge opening."""
    x = np.asarray(x, dtype=float)
    cls = x**N1_ROUND_LE * (1.0 - x) ** N2_SHARP_TE
    shape = (np.asarray(a, dtype=float)[:, None] * bernstein(len(a) - 1, x)).sum(0)
    return cls * shape + x * dz_te


def cosine_x(n: int) -> np.ndarray:
    """Cosine-clustered chord stations: dense at LE and TE, where the
    curvature that matters lives."""
    beta = np.linspace(0.0, np.pi, n)
    return 0.5 * (1.0 - np.cos(beta))


@dataclass(frozen=True)
class Airfoil:
    """A unit-chord section. au/al are CST coefficients; te_gap is the
    TOTAL trailing-edge thickness as a fraction of chord (split evenly
    between the surfaces), which the vase-mode exporter will later force
    up to at least one extrusion width."""

    au: np.ndarray
    al: np.ndarray
    te_gap: float = 0.0
    te_camber: float = 0.0
    name: str = "cst"

    @property
    def order(self) -> int:
        return len(self.au) - 1

    def y_upper(self, x: np.ndarray) -> np.ndarray:
        return cst_y(x, self.au, 0.5 * self.te_gap + self.te_camber)

    def y_lower(self, x: np.ndarray) -> np.ndarray:
        return cst_y(x, self.al, -0.5 * self.te_gap + self.te_camber)

    def coords(self, n: int = 121) -> np.ndarray:
        """Closed Selig-order loop, (2n-1, 2): TE over the UPPER surface
        to the LE, then back along the lower surface to the TE."""
        x = cosine_x(n)
        upper = np.stack([x, self.y_upper(x)], 1)[::-1]   # TE -> LE
        lower = np.stack([x, self.y_lower(x)], 1)[1:]     # LE -> TE
        return np.concatenate([upper, lower], 0)

    # -- geometric readouts the objective and the printability gate need --

    def thickness(self, x: np.ndarray) -> np.ndarray:
        return self.y_upper(x) - self.y_lower(x)

    def camber(self, x: np.ndarray) -> np.ndarray:
        return 0.5 * (self.y_upper(x) + self.y_lower(x))

    @property
    def t_max(self) -> float:
        x = cosine_x(200)
        return float(self.thickness(x).max())

    @property
    def x_t_max(self) -> float:
        x = cosine_x(200)
        return float(x[int(np.argmax(self.thickness(x)))])

    @property
    def camber_max(self) -> float:
        x = cosine_x(200)
        return float(np.abs(self.camber(x)).max())

    @property
    def area(self) -> float:
        """Enclosed area / chord^2 -- the shell's cross-section, which is
        what sets vase-mode wall length and therefore mass."""
        x = cosine_x(400)
        return float(np.trapezoid(self.thickness(x), x))

    @property
    def perimeter(self) -> float:
        """Loop length / chord -- one vase-mode layer's extrusion length."""
        p = self.coords(300)
        return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())

    def is_valid(self) -> bool:
        """Surfaces must not cross: a 'negative thickness' airfoil is a
        self-intersecting print, and the optimizer WILL find one.

        Checked over (0, 0.98] only. The last 2% of chord is the trailing
        edge, where a sharp section legitimately has zero thickness and
        the vase-mode exporter owns the minimum-feature problem (it
        thickens the TE to one extrusion width). Rejecting sharp TEs
        here would reject every real airfoil, MH45 included."""
        x = cosine_x(300)
        x = x[(x > 1e-3) & (x <= 0.98)]
        return bool(np.all(self.thickness(x) > 1e-4))

    @property
    def te_y(self) -> float:
        """Camber-line height at the trailing edge. The single number the
        class function cannot express on its own."""
        return self.te_camber

    def scaled_thickness(self, factor: float) -> "Airfoil":
        """Same camber line, thickness scaled -- how a blended body grows
        thick at the root without changing its aerodynamic camber."""
        cam = 0.5 * (self.au + self.al)
        half = 0.5 * (self.au - self.al) * factor
        return replace(self, au=cam + half, al=cam - half,
                       te_gap=self.te_gap * factor)


def fit(xy: np.ndarray, order: int = 8, name: str = "fit") -> Airfoil:
    """Least-squares fit of a Selig loop to CST coefficients.

    Linear in the coefficients once the class function is divided out,
    so this is exact within the basis' reach -- no optimizer, no seed,
    no convergence question."""
    xy = np.asarray(xy, dtype=float)
    i_le = int(np.argmin(xy[:, 0]))
    upper = xy[: i_le + 1][::-1]          # LE -> TE
    lower = xy[i_le:]                     # LE -> TE
    te_gap = float(upper[-1, 1] - lower[-1, 1])
    # The trailing-edge CAMBER offset, and why it has to be here: the
    # class function C(x) = x^N1 (1-x)^N2 is ZERO at x = 1, so C(x)S(x)
    # pins the surface to the chord line at the trailing edge no matter
    # what the shape coefficients say. Without an explicit x*z_te term
    # carrying the same sign on both surfaces, a REFLEXED section cannot
    # be represented at all -- the fit hooks violently down over the last
    # 1% of chord, and that hook behaves like a large down-flap. It cost
    # a wrong-signed Cm0 and an afternoon to find. te_gap (opposite signs)
    # is thickness; te_camber (same sign) is camber. Both are needed.
    te_camber = float(0.5 * (upper[-1, 1] + lower[-1, 1]))

    def solve(surf: np.ndarray, sign: float) -> np.ndarray:
        x, y = surf[:, 0], surf[:, 1]
        keep = (x > 1e-6) & (x < 1.0 - 1e-9)
        x, y = x[keep], y[keep]
        cls = x**N1_ROUND_LE * (1.0 - x) ** N2_SHARP_TE
        basis = (bernstein(order, x) * cls[None, :]).T
        rhs = y - x * (sign * 0.5 * te_gap + te_camber)
        return np.linalg.lstsq(basis, rhs, rcond=None)[0]

    return Airfoil(au=solve(upper, +1.0), al=solve(lower, -1.0),
                   te_gap=te_gap, te_camber=te_camber, name=name)


def load_selig(path: str | Path, order: int = 8) -> Airfoil:
    """Read a Selig .dat and return it as a design vector."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    pts = []
    for ln in lines[1:]:
        parts = ln.split()
        if len(parts) >= 2:
            try:
                pts.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    if len(pts) < 10:
        raise ValueError(f"{path}: only {len(pts)} coordinate pairs")
    return fit(np.array(pts), order=order, name=lines[0].strip() or path.stem)


def naca4(code: str = "2412", order: int = 8, n: int = 200) -> Airfoil:
    """Analytic NACA 4-digit, fitted -- a reference section for tests."""
    m, p, t = int(code[0]) / 100, int(code[1]) / 10, int(code[2:]) / 100
    x = cosine_x(n)
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2
                  + 0.2843 * x**3 - 0.1036 * x**4)
    if p > 0:
        yc = np.where(x < p, m / p**2 * (2 * p * x - x**2),
                      m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x**2))
    else:
        yc = np.zeros_like(x)
    upper = np.stack([x, yc + yt], 1)
    lower = np.stack([x, yc - yt], 1)
    loop = np.concatenate([upper[::-1], lower[1:]], 0)
    return fit(loop, order=order, name=f"NACA {code}")


def blend(a: Airfoil, b: Airfoil, t: float) -> Airfoil:
    """Linear interpolation in coefficient space -- this is the whole
    reason for a CST basis. Blending two Selig point clouds is
    ill-defined; blending two coefficient vectors is just a lerp, and
    the result is still a valid smooth airfoil."""
    if len(a.au) != len(b.au):
        raise ValueError("blend needs equal CST order")
    return Airfoil(au=(1 - t) * a.au + t * b.au,
                   al=(1 - t) * a.al + t * b.al,
                   te_gap=(1 - t) * a.te_gap + t * b.te_gap,
                   te_camber=(1 - t) * a.te_camber + t * b.te_camber,
                   name=f"{a.name}~{b.name}@{t:.2f}")


def deflect_te(af: Airfoil, deg: float, hinge: float = 0.70) -> Airfoil:
    """Deflect the camber line aft of `hinge`; positive deg = TRAILING
    EDGE UP, i.e. reflex. Thickness is carried along unchanged.

    This exists because nudging raw CST coefficients is NOT a reflex
    knob, which cost a debugging session worth writing down. The last
    Bernstein coefficient multiplied by the class function peaks near
    x = 0.85 and returns to zero at x = 1, so raising it builds a bump
    whose tail slopes DOWN into the trailing edge -- aerodynamically a
    positive flap deflection. Cm0 duly went the wrong way.

    Reflex is a property of the trailing edge itself, so it is
    parametrized as one: a quadratic ramp that leaves the hinge with zero
    slope (no crease to print around) and reaches tan(deg) of rise at the
    trailing edge. Monotone in Cm0 by construction, and it reads as what
    it is -- degrees of up-elevon moulded into the section.
    """
    x = cosine_x(240)
    cam = af.camber(x)
    t = af.thickness(x)
    s = np.clip((x - hinge) / (1.0 - hinge), 0.0, 1.0)
    cam = cam + np.tan(np.radians(deg)) * (1.0 - hinge) * s**2
    upper = np.stack([x, cam + 0.5 * t], 1)
    lower = np.stack([x, cam - 0.5 * t], 1)
    loop = np.concatenate([upper[::-1], lower[1:]], 0)
    return fit(loop, order=af.order, name=f"{af.name}+{deg:.1f}deg_reflex")


def scale_camber(af: Airfoil, factor: float) -> Airfoil:
    """Scale the camber line, keep the thickness distribution."""
    x = cosine_x(240)
    cam = af.camber(x) * factor
    t = af.thickness(x)
    loop = np.concatenate([np.stack([x, cam + 0.5 * t], 1)[::-1],
                           np.stack([x, cam - 0.5 * t], 1)[1:]], 0)
    return fit(loop, order=af.order, name=f"{af.name}*cam{factor:.2f}")


def set_thickness_peak(af: Airfoil, x_target: float) -> Airfoil:
    """Move the point of maximum thickness, keeping camber and t/c.

    A power-law remap of the thickness distribution: t_new(x) = t(x^p).
    The old peak at x0 lands at x0^(1/p), so p = ln(x0)/ln(x_target) puts
    it exactly where asked. Monotone for any positive p, so the section
    stays valid.

    Worth having as a design variable because it is the knob that trades
    the two things a printed wing cares about: forward peak gives a
    fuller nose and gentler stall, aft peak gives lower drag and more
    internal depth further back -- which is where the battery goes."""
    x = cosine_x(300)
    cam = af.camber(x)
    t = af.thickness(x)
    x0 = float(x[int(np.argmax(t))])
    x_target = float(np.clip(x_target, 0.15, 0.55))
    if not (1e-3 < x0 < 1.0) or abs(x0 - x_target) < 1e-4:
        return af
    p = np.log(x0) / np.log(x_target)
    t_new = np.interp(np.clip(x**p, 0.0, 1.0), x, t)
    loop = np.concatenate([np.stack([x, cam + 0.5 * t_new], 1)[::-1],
                           np.stack([x, cam - 0.5 * t_new], 1)[1:]], 0)
    return fit(loop, order=af.order, name=f"{af.name}@xt{x_target:.2f}")


def scale_thickness_ratio(af: Airfoil, t_over_c: float) -> Airfoil:
    """Set the thickness ratio outright, leaving camber alone."""
    cur = af.t_max
    if cur < 1e-6:
        return af
    return af.scaled_thickness(float(t_over_c) / cur)


def flap_effectiveness(chord_frac: float) -> float:
    """Thin-aerofoil flap effectiveness tau: dcl/ddelta = a0 * tau.

    tau = 1 - (theta - sin theta)/pi,  theta = acos(2 Ef - 1)

    The classical result, and the reason elevons are so powerful on a
    flying wing: a 25% chord elevon already recovers about half the
    section's full lift slope, so a few degrees moves the trim point a
    long way. That is worth checking as a CONSTRAINT rather than
    assuming -- a wing with too much elevon authority is not safe, it is
    twitchy."""
    e = float(np.clip(chord_frac, 0.02, 0.6))
    theta = np.arccos(2.0 * e - 1.0)
    return float(1.0 - (theta - np.sin(theta)) / np.pi)
