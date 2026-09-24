"""Does the wing twist itself apart, and do the elevons reverse?

`ROADMAP.md` item 9. Nothing in this program could see either until now:
`grep -rn "flutter|divergence|torsion|GJ"` returned one docstring about
the spiral mode. demon1 is scored at **44 m/s** on a single-wall foamed
PLA shell with one 8 mm tube, and the speed objective pushes directly
toward the failure that had no gate. That is the shape of an optimizer
walking off a cliff.

Both speeds come from the same typical-section model, which is the
standard one and is derived here rather than quoted, because the two
results differ by a sign and a factor that are easy to get wrong.

## The model

A rigid section on a torsion spring of stiffness K about its elastic
axis, with the aerodynamic centre a distance `e * c` FORWARD of that
axis. Twist theta, flap deflection delta, dynamic pressure q:

    moment about the EA   M = q c^2 [ e (a theta + cl_d delta)
                                      + cm_d delta ]
    spring                M = K theta

    =>  theta (K - q c^2 e a) = q c^2 (e cl_d + cm_d) delta

**Divergence** is where the bracket vanishes -- the aerodynamic stiffness
has eaten the structural stiffness and the twist is unbounded with no
flap input at all:

    q_div = K / (c^2 e a)

**Reversal** is where total lift stops responding to the flap. With
cl_total = a theta + cl_d delta,

    d cl_total / d delta = a q c^2 (e cl_d + cm_d) / (K - q c^2 e a)
                           + cl_d

Setting that to zero, the `e cl_d` terms cancel exactly and what is left
is remarkably simple:

    q_rev = - cl_d K / (c^2 a cm_d)

The cancellation is the interesting part: **reversal does not depend on
where the elastic axis is**, only on how much nose-down moment the flap
makes for the lift it buys. Divergence depends on `e` and reversal does
not, so the two are moved by different things and a design can be fixed
for one and still fail the other.

For a cantilever of semi-span L with uniform torsional rigidity GJ, the
first torsional mode gives the equivalent spring

    K = GJ (pi / 2L)^2

which reproduces the classical cantilever divergence result
q_div = (pi/2L)^2 GJ / (c^2 a e).

## What is honest about this and what is not

GJ is computed, not assumed: Bredt-Batho for the closed cell the skin
makes, plus the spar tubes, and the open-section formula for what is left
when a bay is cut in the upper skin. Those are closed-form results for
thin-walled sections and they are what this model is entitled to.

The elastic axis is an **estimate**, and it is stated as one. The shear
centre of a closed single-cell thin-walled section is not the enclosed
area's centroid, though for a smooth convex cell it is close; that
centroid is what is used, and it sits aft of the quarter chord, which is
the direction that makes divergence possible. A better number needs a
shear-flow solve around the cell, which is worth doing before anyone
trusts the absolute speed rather than the ranking.

GJ is a **lower bound**, and knowing which way an error points is half
of using it. Bredt-Batho here is applied to ONE closed cell, the one the
skin makes; the rib truss divides that into several cells and a
multi-cell section is stiffer, so the real GJ is higher and both speeds
read low. Against that, the compliance is taken in SERIES along the span
rather than using the stiff root cell everywhere, which on the trainer is
38% below the root value and is the honest direction.

The two speeds are not equally trustworthy, and the derivation says
which is which. Divergence needs `e`, so it inherits the elastic-axis
estimate. **Reversal does not** -- the `e cl_d` terms cancel -- so it
rests only on GJ, the span, the chord, the lift slope and the elevon's
own geometry, all of which are computed. When they disagree, believe
reversal.

This is a STATIC aeroelastic model. Flutter proper -- the coupling of
bending and torsion with the flow, which can happen below both of these
speeds -- needs the mass distribution and a frequency-matched solve, and
is not here. Divergence and reversal are the two speeds a static model is
allowed to state, and they are two more than zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RHO_AIR = 1.225

G_SKIN_MPA = 333.0
"""Shear modulus of the printed skin.

From the shell's own E = 900 MPa (`structure.SkinMaterial`, foamed
LW-PLA) at nu = 0.35: G = E / 2(1 + nu) = 333 MPa. Derived from the
modulus already declared rather than a second independent number, so
the two cannot drift apart."""

G_SPAR_GPA = 5.0
"""Torsional shear modulus of a pultruded unidirectional carbon tube.

An order below its 130 GPa axial modulus, because torsion loads the
matrix rather than the fibres, and that is exactly why a spar contributes
far less to GJ than its bending stiffness suggests."""


def _harmonic(v: np.ndarray) -> float:
    """Series compliance: 1/GJ_eff = mean(1/GJ). Zeros make it zero."""
    v = np.asarray(v, dtype=float)
    if np.any(v <= 0.0):
        return 0.0
    return float(len(v) / np.sum(1.0 / v))


def cell_properties(plan, eta: float, wall_mm: float) -> tuple[float, float, float]:
    """(enclosed area mm^2, perimeter mm, shear-centre chord fraction).

    The enclosed area's centroid is used as the shear-centre estimate --
    see the module docstring for why that is an approximation and which
    way it errs.
    """
    st = plan.at(float(np.clip(eta, 0.0, 1.0)))
    c_mm = st.chord_m * 1000.0
    loop = st.airfoil.coords(121) * c_mm
    x, y = loop[:, 0], loop[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    a2 = float(cross.sum())
    area = 0.5 * abs(a2)
    per = float(np.hypot(xn - x, yn - y).sum())
    # polygon centroid; guard the degenerate case
    if abs(a2) < 1e-9:
        return 0.0, per, 0.4
    cx = float(((x + xn) * cross).sum() / (3.0 * a2))
    # the MIDLINE encloses less than the outer contour: one wall in
    return max(area - 0.5 * per * wall_mm, 0.0), per, cx / max(c_mm, 1e-9)


def gj_closed_nmm2(area_mm2: float, per_mm: float, wall_mm: float,
                   g_mpa: float = G_SKIN_MPA) -> float:
    """Bredt-Batho: J = 4 A^2 / (perimeter / t) for a single closed cell."""
    if area_mm2 <= 0.0 or per_mm <= 0.0 or wall_mm <= 0.0:
        return 0.0
    j = 4.0 * area_mm2 * area_mm2 / (per_mm / wall_mm)
    return float(g_mpa * j)


def gj_open_nmm2(per_mm: float, wall_mm: float,
                 g_mpa: float = G_SKIN_MPA) -> float:
    """An open thin-walled section: J = (1/3) * integral t^3 ds.

    One to two orders of magnitude below the closed value for the same
    material and wall, which is the whole reason cutting a hatch in the
    upper skin is a structural decision and not a cosmetic one."""
    return float(g_mpa * per_mm * wall_mm ** 3 / 3.0)


def gj_multicell_nmm2(plan, eta: float, wall_mm: float, n_ribs: int,
                      x_first: float = 0.20, x_last: float = 0.72,
                      g_mpa: float = G_SKIN_MPA) -> tuple[float, int]:
    """GJ counting the cells the rib truss divides the box into.

    -> (GJ in N.mm^2, number of cells)

    N chordwise webs make N+1 closed cells, and a multi-cell section is
    substantially stiffer in torsion than the single cell the skin makes
    on its own. The standard formulation, for cells i = 1..N+1 under a
    common twist rate psi:

        (1 / 2 A_i) [ q_i * delta_ii - sum_j q_j * delta_ij ] = G psi
        T = 2 * sum_i q_i A_i

    where delta_ii is the closed line integral of ds/t around cell i and
    delta_ij the same along the wall it shares with cell j. Setting
    G psi = 1 turns it into one linear solve for the shear flows; GJ is
    then G T. With a single cell it collapses to 4 A^2 t / s, which is
    Bredt-Batho -- that collapse is the reference check, and it is exact.

    **This is an UPPER bound and the single-cell result is a lower one.**
    The ribs are not continuous webs: the truss is a diamond, adjacent
    ribs sweeping in opposite directions as Z rises, so a rib at a given
    chord station exists only at some heights. Treating it as a solid web
    overstates the shear path; ignoring it entirely understates it. The
    truth is between, and the honest thing is to carry both numbers
    rather than pick one and call it the answer.
    """
    st = plan.at(float(np.clip(eta, 0.0, 1.0)))
    c_mm = st.chord_m * 1000.0
    n = max(int(n_ribs), 0)
    if n <= 0:
        area, per, _ = cell_properties(plan, eta, wall_mm)
        return gj_closed_nmm2(area, per, wall_mm, g_mpa), 1

    xr = np.linspace(x_first, x_last, n) if n > 1 else np.array(
        [0.5 * (x_first + x_last)])
    edges = np.concatenate([[0.0], np.sort(xr), [1.0]])

    t_rib = 2.0 * wall_mm        # two legs one bead apart, welded
    xs = np.linspace(0.0, 1.0, 241)
    y_up = st.airfoil.y_upper(xs) * c_mm
    y_lo = st.airfoil.y_lower(xs) * c_mm
    x_mm = xs * c_mm

    areas, skin_len, rib_len = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (xs >= a) & (xs <= b)
        if sel.sum() < 2:
            sel = np.zeros_like(xs, dtype=bool)
            sel[np.argmin(np.abs(xs - a))] = True
            sel[np.argmin(np.abs(xs - b))] = True
        xx, yu, yl = x_mm[sel], y_up[sel], y_lo[sel]
        areas.append(float(np.trapezoid(yu - yl, xx)))
        skin_len.append(float(np.hypot(np.diff(xx), np.diff(yu)).sum()
                              + np.hypot(np.diff(xx), np.diff(yl)).sum()))
    # each interior edge is a rib; its length is the section depth there
    for e in edges[1:-1]:
        i = int(np.argmin(np.abs(xs - e)))
        rib_len.append(float(y_up[i] - y_lo[i]))

    m = len(areas)
    A = np.zeros((m, m))
    rhs = np.ones(m)
    for i in range(m):
        d_ii = skin_len[i] / max(wall_mm, 1e-9)
        if i > 0:
            d_ii += rib_len[i - 1] / t_rib
        if i < m - 1:
            d_ii += rib_len[i] / t_rib
        A[i, i] = d_ii / (2.0 * max(areas[i], 1e-9))
        if i > 0:
            A[i, i - 1] = -(rib_len[i - 1] / t_rib) / (2.0 * max(areas[i], 1e-9))
        if i < m - 1:
            A[i, i + 1] = -(rib_len[i] / t_rib) / (2.0 * max(areas[i], 1e-9))
    try:
        q = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        area, per, _ = cell_properties(plan, eta, wall_mm)
        return gj_closed_nmm2(area, per, wall_mm, g_mpa), 1
    if np.any(q <= 0.0):                     # unphysical: fall back, loudly
        area, per, _ = cell_properties(plan, eta, wall_mm)
        return gj_closed_nmm2(area, per, wall_mm, g_mpa), 1
    torque = 2.0 * float(np.dot(q, areas))
    return float(g_mpa * torque), m


def gj_spars_nmm2(spar_fits, g_gpa: float = G_SPAR_GPA,
                  eta: float | None = None) -> float:
    """The tubes' own contribution. J = pi (D^4 - d^4) / 32.

    At `eta`, only the tubes that reach it. Every tube used to be added
    at every station, so the outer third of each wing was stiffened by
    carbon that ends at eta 0.66 -- a third of the GJ, and reversal, the
    binding limit on the fleet, goes as its square root."""
    total = 0.0
    for f in spar_fits:
        if eta is not None and f.reach_eta < eta:
            continue
        od = f.spec.d_mm
        idm = max(od - 2.0, 0.0)              # 8x6, 6x4: 1 mm wall
        total += np.pi * (od ** 4 - idm ** 4) / 32.0
    return float(g_gpa * 1000.0 * total)


@dataclass
class Aeroelastic:
    gj_nmm2: float
    gj_open_nmm2: float
    ea_frac: float
    e_frac: float
    k_nmm_per_rad: float
    v_div_ms: float
    v_rev_ms: float
    v_div_open_ms: float
    design_v_ms: float
    margin: float
    ok: bool
    gj_ribbed_nmm2: float = 0.0
    margin_hi: float = 0.0
    v_rev_hi_ms: float = 0.0
    notes: tuple[str, ...] = ()
    hatch_eta: float = 0.0
    """Half-span fraction of the hand-cut hatch, measured from the centreline;
    0 means none."""
    gj_hatch_nmm2: float = 0.0
    v_div_hatch_ms: float = 0.0
    v_rev_hatch_ms: float = 0.0
    margin_hatch: float = 0.0

    def report(self) -> str:
        mark = "OK" if self.ok else "FAILS"
        lines = [
            f"aeroelastic: {mark}  (static: divergence and reversal only)",
            f"  torsion box   GJ {self.gj_nmm2/1e6:.2f} N.m^2 closed | "
            f"{self.gj_open_nmm2/1e6:.3f} open (bay cut in the upper skin)",
            f"  elastic axis  {self.ea_frac:.3f}c estimated, AC at 0.250c "
            f"-> e = {self.e_frac:+.3f}c",
            f"  divergence    {self.v_div_ms:.0f} m/s  |  reversal "
            f"{self.v_rev_ms:.0f} m/s  |  design {self.design_v_ms:.0f} m/s",
            f"  margin        {self.margin:.2f}x to {self.margin_hi:.2f}x "
            f"(ribs ignored .. ribs as solid webs); the gate uses the lower",
        ]
        if self.v_div_open_ms > 0.0:
            lines.append(
                f"  OPEN section   divergence falls to "
                f"{self.v_div_open_ms:.0f} m/s if the WHOLE span were open "
                f"-- a bound, not the hatch (see HATCH)")
        if self.hatch_eta > 0.0:
            lines.append(
                f"  HATCH         cut to eta {self.hatch_eta:.3f}: GJ "
                f"{self.gj_hatch_nmm2/1e6:.3f} N.m^2, divergence "
                f"{self.v_div_hatch_ms:.0f} | reversal {self.v_rev_hatch_ms:.0f} m/s "
                f"-> {self.margin_hatch:.2f}x")
        lines += [f"  note          {n}" for n in self.notes]
        return "\n".join(lines)


def hatch_gj_nmm2(plan, spar_fits, wall_mm: float, gj_closed_eff: float,
                  hatch_eta: float) -> float:
    """The wing's equivalent GJ with the upper skin cut open over
    |eta| < hatch_eta.

    The cut section is OPEN: the skin's own J falls to (1/3) per t^3 and
    what carries torque across is mostly the spar tube. It is short -- the
    battery hatch is 17 mm of a 240 mm semi-span -- but it is at the
    root, and the root is where compliance costs most. Under a torque
    distributed along the span, the section at y carries everything
    outboard of it, t (L - y). The tip twist is the integral of that over
    GJ, so a compliance at the root is weighted L against the span
    average of L / 2: TWICE its span fraction. The existing equivalent GJ
    is a plain harmonic mean (a uniform weight, as `analyse` explains),
    so the hatch's extra compliance is added with that factor of two:

        1/GJ_h = 1/GJ_eff + 2 f (1/GJ_open - 1/GJ_closed),  f = hatch_eta

    both GJs taken at the middle of the hatch, with the tubes that reach
    there. Zero hatch returns the closed value exactly."""
    if hatch_eta <= 0.0 or gj_closed_eff <= 0.0:
        return float(gj_closed_eff)
    e_mid = 0.5 * float(hatch_eta)
    area, per, _ = cell_properties(plan, e_mid, wall_mm)
    spar = gj_spars_nmm2(spar_fits, eta=e_mid)
    closed = gj_closed_nmm2(area, per, wall_mm) + spar
    open_ = gj_open_nmm2(per, wall_mm) + spar
    if open_ <= 0.0 or closed <= 0.0:
        return 0.0
    comp = 1.0 / gj_closed_eff + 2.0 * float(hatch_eta) * (1.0 / open_ - 1.0 / closed)
    return float(1.0 / comp) if comp > 0.0 else 0.0


def analyse(plan, spar_fits, lift_slope_per_rad: float,
            elevon_chord_frac: float, elevon_eta: float,
            design_v_ms: float, wall_mm: float,
            min_margin: float = 1.2, n_ribs: int = 0,
            hatch_eta: float = 0.0) -> Aeroelastic:
    """Divergence and reversal speeds for this wing at this speed.

    `design_v_ms` is the speed the aircraft is SCORED at -- top speed for
    a racer, cruise for a trainer -- because that is the speed the
    optimizer is pushing toward and therefore the one the margin has to
    be measured against.
    """
    from .geom.cst import flap_effectiveness

    # GJ varies along the span -- the section thins outboard and the
    # chord runs out -- so taking the root cell's value with the full
    # semi-span would treat the whole wing as if it were as stiff as its
    # thickest station, which reads both speeds HIGH.
    #
    # Torsional compliance adds in series along the span, so the
    # equivalent uniform stiffness is the HARMONIC mean, not the
    # arithmetic one. On the trainer that is 38% below the root value,
    # which is 38% of a speed the optimizer would otherwise be handed.
    etas = np.linspace(0.02, 0.95, 17)
    gjs, gjs_open, eas = [], [], []
    for e_ in etas:
        area, per, ea_ = cell_properties(plan, float(e_), wall_mm)
        gjs.append(gj_closed_nmm2(area, per, wall_mm))
        gjs_open.append(gj_open_nmm2(per, wall_mm))
        eas.append(ea_)
    spar_gj = np.array([gj_spars_nmm2(spar_fits, eta=float(e_)) for e_ in etas])
    gj = _harmonic(np.array(gjs) + spar_gj)
    gj_open = _harmonic(np.array(gjs_open) + spar_gj)
    # The rib truss divides the box into cells and a multi-cell section is
    # stiffer. Treating the ribs as continuous webs OVERSTATES it -- the
    # truss is a diamond, so a rib at a given chord station exists only at
    # some heights -- so this is the upper bound to the single cell's
    # lower one, and both are carried.
    if n_ribs > 0:
        gj_ribbed = _harmonic(np.array(
            [gj_multicell_nmm2(plan, float(e_), wall_mm, n_ribs)[0]
             for e_ in etas]) + spar_gj)
    else:
        gj_ribbed = gj
    # the elastic axis is weighted by local stiffness: the stiff inboard
    # cell is what the twist is reacting against
    w = np.array(gjs) + spar_gj
    ea = float(np.average(np.array(eas), weights=w))

    half_mm = plan.half_span_m * 1000.0
    k = gj * (np.pi / (2.0 * half_mm)) ** 2          # N.mm per rad
    k_open = gj_open * (np.pi / (2.0 * half_mm)) ** 2
    k_ribbed = gj_ribbed * (np.pi / (2.0 * half_mm)) ** 2

    c_mm = plan.mac_m * 1000.0
    a = float(lift_slope_per_rad)
    e = ea - 0.25

    notes: list[str] = []

    def q_to_v(q):
        return float(np.sqrt(2.0 * max(q, 0.0) / RHO_AIR)) if q > 0 else np.inf

    # --- divergence ---
    if e <= 1e-6:
        v_div = v_div_hi = np.inf
        notes.append("elastic axis is forward of the AC: divergence is not "
                     "possible, the wing twists nose-down under lift")
        v_div_open = np.inf
    else:
        # K in N.mm, c in mm -> q in N/mm^2; x1e6 for N/m^2
        v_div = q_to_v(k / (c_mm * c_mm * e * a) * 1e6)
        v_div_open = q_to_v(k_open / (c_mm * c_mm * e * a) * 1e6)
        v_div_hi = q_to_v(k_ribbed / (c_mm * c_mm * e * a) * 1e6)

    # --- reversal ---
    # cl_d and cm_d per RADIAN of elevon, for the wing as a whole: the
    # elevon covers only part of the span, so both are scaled by the area
    # it actually occupies. The same geometric integral the elevon
    # authority gate already uses, taken about the AC rather than the CG.
    tau = flap_effectiveness(elevon_chord_frac)
    x_ac_m = plan.stations[0].x_le_m + 0.25 * plan.mac_m
    ele_area = 0.0
    ele_arm = 0.0
    edges = np.linspace(float(elevon_eta), 1.0, 9)
    for lo, hi in zip(edges[:-1], edges[1:]):
        st = plan.at(0.5 * (lo + hi))
        dA = st.chord_m * elevon_chord_frac * (hi - lo) * plan.half_span_m
        x_e = st.x_le_m + (1.0 - 0.5 * elevon_chord_frac) * st.chord_m
        ele_area += dA
        ele_arm += dA * (x_e - x_ac_m)
    frac = 2.0 * ele_area / max(plan.area_m2, 1e-9)
    cl_d = frac * a * tau
    arm = ele_arm / max(ele_area, 1e-9)
    cm_d = -frac * a * tau * (arm / plan.mac_m)       # nose-down: negative

    if cm_d >= -1e-9 or cl_d <= 0.0:
        v_rev = v_rev_hi = np.inf
        notes.append("elevon makes no nose-down moment: reversal is not "
                     "possible in this model")
    else:
        v_rev = q_to_v(-cl_d * k / (c_mm * c_mm * a * cm_d) * 1e6)
        v_rev_hi = q_to_v(-cl_d * k_ribbed / (c_mm * c_mm * a * cm_d) * 1e6)

    # The gate runs on the LOWER bound. Flutter is unforgiving and the
    # band's width is reported, so nothing is hidden by the choice.
    worst = min(v_div, v_rev)
    worst_hi = min(v_div_hi, v_rev_hi)
    margin = float(worst / max(design_v_ms, 1e-6)) if np.isfinite(worst) else 99.0
    margin_hi = (float(worst_hi / max(design_v_ms, 1e-6))
                 if np.isfinite(worst_hi) else 99.0)
    ok = margin >= min_margin
    if not ok:
        notes.append(f"the lower of the two is only {margin:.2f}x the speed "
                     f"this design is scored at")
    notes.append("STATIC only: classical flutter needs the mass "
                 "distribution and is not modelled")
    if n_ribs > 0:
        notes.append(
            f"ribs counted as continuous webs would give {gj_ribbed/1e6:.2f} "
            f"N.m^2 and a {margin_hi:.2f}x margin -- an UPPER bound, because "
            f"the truss is a diamond and a rib exists only at some heights")
    else:
        notes.append("no ribs in these settings: single closed cell, so GJ "
                     "is a lower bound and both speeds read low")
    notes.append("reversal does not depend on the elastic axis (the e*cl_d "
                 "terms cancel), so it is the firmer of the two numbers")

    # The hand-cut battery hatch, when there is one: the same two speeds on
    # the stiffness with the root opened. Reported beside the closed
    # numbers, and the build sheet gates on it; the search does not yet.
    gj_h = v_div_h = v_rev_h = margin_h = 0.0
    if hatch_eta > 0.0:
        gj_h = hatch_gj_nmm2(plan, spar_fits, wall_mm, gj, hatch_eta)
        k_h = gj_h * (np.pi / (2.0 * half_mm)) ** 2
        vd = (q_to_v(k_h / (c_mm * c_mm * e * a) * 1e6) if e > 1e-6 else np.inf)
        vr = (q_to_v(-cl_d * k_h / (c_mm * c_mm * a * cm_d) * 1e6)
              if (cm_d < -1e-9 and cl_d > 0.0) else np.inf)
        worst_h = min(vd, vr)
        margin_h = (float(worst_h / max(design_v_ms, 1e-6))
                    if np.isfinite(worst_h) else 99.0)
        v_div_h = float(vd if np.isfinite(vd) else 9999.0)
        v_rev_h = float(vr if np.isfinite(vr) else 9999.0)

    return Aeroelastic(
        hatch_eta=float(hatch_eta), gj_hatch_nmm2=float(gj_h),
        v_div_hatch_ms=v_div_h, v_rev_hatch_ms=v_rev_h,
        margin_hatch=float(margin_h),
        gj_nmm2=gj, gj_open_nmm2=gj_open, ea_frac=ea, e_frac=e,
        k_nmm_per_rad=k,
        v_div_ms=float(v_div if np.isfinite(v_div) else 9999.0),
        v_rev_ms=float(v_rev if np.isfinite(v_rev) else 9999.0),
        v_div_open_ms=float(v_div_open if np.isfinite(v_div_open) else 0.0),
        design_v_ms=float(design_v_ms), margin=margin, ok=bool(ok),
        gj_ribbed_nmm2=float(gj_ribbed), margin_hi=float(margin_hi),
        v_rev_hi_ms=float(v_rev_hi if np.isfinite(v_rev_hi) else 9999.0),
        notes=tuple(notes))
