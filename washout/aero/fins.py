"""Vertical tip fins: flat plates, printed flat, glued to the wing tips.

Why they are here. The Dutch-roll study on the gen3 trainer found its
weakness was YAW, not simply too much dihedral: a tailless wing's only
yaw stiffness came from its canted winglets, and a canted surface buys
side area only by also buying dihedral effect -- the same geometry that
feeds the roll half of the oscillation. A vertical plate adds side area,
weathercock stiffness and yaw DAMPING with far less roll coupling, and
with none at all if it reaches as far below the chord line as above.

Why flat plates and not part of the loft. Vase mode prints one closed
contour per layer with the span along Z; a surface standing normal to
the span cannot come out of that spiral. A flat part printed on the bed
can, in minutes -- and a fin is exactly the part that should be cheap to
replace after a hard landing.

What is modelled, and on what authority:
  * geometry -- a trapezoid plate, root chord on the tip chord line,
    trailing edge flush with the tip trailing edge, leading edge swept,
    split above and below the chord line by `below_frac`
  * lift slope -- Helmbold's low-aspect-ratio formula on an effective
    aspect ratio 1.5x geometric, because the wing tip acts as a partial
    reflection plane for the fin root. That factor is the calibrated part.
  * derivatives -- side force at the plate centroid, with the local
    sideslip each body rate induces there: the standard tail-surface
    argument, with no further coefficients
  * drag -- laminar flat-plate friction on both faces with a form factor
    for a printed plate; it also damps yaw, since it acts at the tips
  * mass -- plate volume at the filament density plus glue, at the centroid

NOT modelled, deliberately: the fins' end-plate reduction of the wing's
induced drag. The benefit is real and it is left unclaimed, so any fin
the optimizer fits has to earn its place on stability alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

EFFECTIVE_AR_FACTOR = 1.5
DYNAMIC_PRESSURE_RATIO = 0.95
FORM_FACTOR = 1.25
GLUE_KG_PER_FIN = 0.0005
NU_AIR = 1.5e-5


def lift_slope(aspect_ratio: float) -> float:
    """Helmbold, per radian. Valid well below aspect ratio 1, where the
    2*pi*A/(A+2) form used for wings is not."""
    a = max(float(aspect_ratio), 1e-6)
    return 2.0 * np.pi * a / (2.0 + np.sqrt(4.0 + a * a))


@dataclass(frozen=True)
class TipFins:
    """One fin's geometry; the aircraft carries two, mirrored."""

    area_m2: float              # ONE fin
    height_m: float
    root_chord_m: float
    tip_chord_m: float
    below_frac: float
    sweep_deg: float
    thickness_m: float
    x_root_le_m: float
    y_m: float                  # spanwise station, both sides
    z_root_m: float             # height of the tip chord line

    @property
    def mean_chord_m(self) -> float:
        return 0.5 * (self.root_chord_m + self.tip_chord_m)

    def outline(self) -> np.ndarray:
        """(x, z) polygon of one plate, counter-clockwise."""
        t = np.tan(np.radians(self.sweep_deg))
        hu = (1.0 - self.below_frac) * self.height_m
        hd = self.below_frac * self.height_m
        x0, zr = self.x_root_le_m, self.z_root_m
        cr, ct = self.root_chord_m, self.tip_chord_m
        pts = [(x0, zr), (x0 + cr, zr), (x0 + hu * t + ct, zr + hu),
               (x0 + hu * t, zr + hu)]
        if hd > 1e-6:
            pts = [(x0, zr), (x0 + hd * t, zr - hd), (x0 + hd * t + ct, zr - hd),
                   (x0 + cr, zr), (x0 + hu * t + ct, zr + hu), (x0 + hu * t, zr + hu)]
        p = np.array(pts)
        if _signed_area(p) < 0.0:
            p = p[::-1]
        return p

    def centroid(self) -> tuple[float, float]:
        p = self.outline()
        q = np.roll(p, -1, axis=0)
        cross = p[:, 0] * q[:, 1] - q[:, 0] * p[:, 1]
        a = 0.5 * cross.sum()
        return (float(((p[:, 0] + q[:, 0]) * cross).sum() / (6.0 * a)),
                float(((p[:, 1] + q[:, 1]) * cross).sum() / (6.0 * a)))

    def effective_aspect_ratio(self) -> float:
        return EFFECTIVE_AR_FACTOR * self.height_m ** 2 / self.area_m2

    def cd0(self, v_ms: float, s_ref_m2: float) -> float:
        """Both fins, both faces, referred to the wing area."""
        re = max(v_ms * self.mean_chord_m / NU_AIR, 1e3)
        cf = 1.328 / np.sqrt(re)
        return 2.0 * (2.0 * self.area_m2) * cf * FORM_FACTOR / s_ref_m2

    def mass_kg(self, density_kg_m3: float) -> float:
        """Both fins."""
        return 2.0 * (self.area_m2 * self.thickness_m * density_kg_m3
                      + GLUE_KG_PER_FIN)

    def derivatives(self, s_ref_m2: float, span_m: float, x_cg_m: float,
                    z_cg_m: float, alpha_deg: float) -> np.ndarray:
        """Both fins' contribution, STABILITY axes, per radian:

            rows  CY, Cl, Cn
            cols  beta, p_hat, r_hat      (p_hat = p b / 2V)

        The side force acts at the plate centroid. A body rate gives the
        fin a local sideslip of (r x_s - p z_s)/V, where x_s and z_s are
        the centroid's stability-axis coordinates from the CG (x forward,
        z down); everything below follows from that and from the moment
        arms of the same force."""
        xf, zf = self.centroid()
        a = np.radians(alpha_deg)
        ca, sa = np.cos(a), np.sin(a)
        xb, zb = -(xf - x_cg_m), -(zf - z_cg_m)       # body: x fwd, z down
        xs, zs = xb * ca + zb * sa, -xb * sa + zb * ca
        cyb = (-2.0 * lift_slope(self.effective_aspect_ratio())
               * DYNAMIC_PRESSURE_RATIO * self.area_m2 / s_ref_m2)
        b = span_m
        return np.array([
            [cyb, -2.0 * cyb * zs / b, 2.0 * cyb * xs / b],
            [-zs * cyb / b, 2.0 * cyb * zs * zs / b ** 2, -2.0 * cyb * xs * zs / b ** 2],
            [xs * cyb / b, -2.0 * cyb * xs * zs / b ** 2, 2.0 * cyb * xs * xs / b ** 2],
        ])

    def mesh_mm(self) -> tuple[np.ndarray, np.ndarray]:
        """A closed plate lying flat on the bed: outline in X-Y, thickness Z."""
        from ..printing import stl as _stl

        poly = self.outline()
        poly = (poly - poly.min(axis=0)) * 1000.0
        n = len(poly)
        tri = np.asarray(_stl._ear_clip(poly), dtype=int)
        # match the triangulation's winding to the (counter-clockwise)
        # outline, so caps and side walls agree before the global check
        a, b, c = poly[tri[0, 0]], poly[tri[0, 1]], poly[tri[0, 2]]
        if (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) < 0.0:
            tri = tri[:, ::-1]
        t = self.thickness_m * 1000.0
        verts = np.vstack([np.column_stack([poly, np.zeros(n)]),
                           np.column_stack([poly, np.full(n, t)])])
        sides = []
        for i in range(n):
            j = (i + 1) % n
            sides += [(i, j, j + n), (i, j + n, i + n)]
        tris = np.vstack([tri[:, ::-1], tri + n, np.array(sides)])
        return verts, _stl._orient_outward(verts, tris)

    def export_stl(self, path: str | Path) -> dict:
        from ..printing import stl as _stl

        verts, tris = self.mesh_mm()
        rep = _stl.manifold_report(tris)
        _stl.write_stl(path, verts, tris, header="washout tip fin (print 2)")
        return rep

    def report(self) -> str:
        return (f"tip fins: 2x {self.area_m2*1e4:.1f} cm^2, "
                f"{self.height_m*1000:.0f} mm tall "
                f"({100*self.below_frac:.0f}% below the chord line), root "
                f"{self.root_chord_m*1000:.0f} mm / tip "
                f"{self.tip_chord_m*1000:.0f} mm chord, "
                f"{self.thickness_m*1000:.1f} mm flat plate -- print two")


def _signed_area(p: np.ndarray) -> float:
    q = np.roll(p, -1, axis=0)
    return 0.5 * float((p[:, 0] * q[:, 1] - q[:, 0] * p[:, 1]).sum())


def tip_fins(plan, area_frac: float, aspect: float, below_frac: float,
             taper: float = 0.6, sweep_deg: float = 30.0,
             thickness_mm: float = 2.0) -> TipFins | None:
    """Size a pair of fins from design variables; None means no fins.

    `area_frac` is ONE fin's area as a fraction of the wing reference
    area. The fin root chord may not exceed the wing's tip chord -- the
    plate has to sit on something -- so a fin that would overhang is made
    taller instead, keeping its area."""
    if area_frac <= 1e-4:
        return None
    area = area_frac * plan.area_m2
    h = float(np.sqrt(area * aspect))
    cbar = area / h
    cr = 2.0 * cbar / (1.0 + taper)
    tip = plan.at(1.0)
    if cr > tip.chord_m:
        cr = tip.chord_m
        cbar = 0.5 * cr * (1.0 + taper)
        h = area / cbar
    return TipFins(area_m2=area, height_m=h, root_chord_m=cr,
                   tip_chord_m=taper * cr,
                   below_frac=float(np.clip(below_frac, 0.0, 0.5)),
                   sweep_deg=sweep_deg, thickness_m=thickness_mm / 1000.0,
                   x_root_le_m=tip.x_le_m + tip.chord_m - cr,
                   y_m=plan.half_span_m, z_root_m=tip.z_le_m)
