"""Does the wing survive being flown? Spar sizing, skin buckling, deflection.

Until now this program could design an aeroplane that flies and prints,
and had nothing whatever to say about whether it holds together. That is
the gap this closes, and it is the gap that decides the print settings:
skin thickness, rib pitch and spar tube are not taste, they are the
lightest combination that passes a load case.

The load case is a hand launch and a beginner's recovery, not a race:

    n_limit  3.5 g   what a trainer actually sees pulling out of a dive
    n_ult    5.25 g  1.5x limit, the classic factor -- nothing may break
    landing         arrives on grass, nose-high, repeatedly

Everything is computed from the VLM's own span loading rather than an
assumed elliptical distribution, because a blended wing body with a fat
centre body is emphatically not elliptical, and the root bending moment
is what sizes the spar.

Allowables are deliberately conservative. Pultruded carbon tube runs
1500 MPa in tension but fails long before that in compression-buckling at
these diameters, and a 3D-printed wing's real failure mode is the skin
letting go around the spar, which no beam formula sees. 400 MPa design
stress and a 3x margin on skin buckling is the honest way to write down
"I cannot model the joint".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

G = 9.80665
RHO_AIR = 1.225


# ------------------------------------------------------------- materials


@dataclass(frozen=True)
class SparTube:
    """A carbon tube off the shelf. od/id in mm, density g/cc."""

    od_mm: float
    id_mm: float
    name: str = ""
    E_gpa: float = 130.0            # pultruded unidirectional CF
    sigma_design_mpa: float = 400.0  # conservative: buckling, not tension
    density_gcc: float = 1.55

    @property
    def I_mm4(self) -> float:
        return np.pi / 64.0 * (self.od_mm**4 - self.id_mm**4)

    @property
    def area_mm2(self) -> float:
        return np.pi / 4.0 * (self.od_mm**2 - self.id_mm**2)

    def mass_g(self, length_mm: float) -> float:
        return self.area_mm2 * length_mm * 1e-3 * self.density_gcc

    def stress_mpa(self, moment_nmm: float) -> float:
        return moment_nmm * (0.5 * self.od_mm) / max(self.I_mm4, 1e-9)


# The tubes a hobbyist actually has. Sizing is a CHOICE FROM THIS LIST,
# not a continuous optimum, because you cannot buy a 7.3 mm tube.
STOCK_TUBES = (
    SparTube(4.0, 2.0, "4x2"),
    SparTube(5.0, 3.0, "5x3"),
    SparTube(6.0, 4.0, "6x4"),
    SparTube(8.0, 6.0, "8x6"),
    SparTube(10.0, 8.0, "10x8"),
    SparTube(12.0, 10.0, "12x10"),
)


@dataclass(frozen=True)
class SkinMaterial:
    """The printed shell. LW-PLA foams on extrusion, so its modulus is a
    fraction of solid PLA and depends on how hard it is foamed."""

    name: str = "LW-PLA"
    density_gcc: float = 0.60
    E_mpa: float = 900.0            # foamed; solid PLA is ~3500
    sigma_mpa: float = 12.0


# ---------------------------------------------------------- span loading


@dataclass
class SpanLoads:
    y_mm: np.ndarray            # 0 at centreline -> tip
    lift_n_per_mm: np.ndarray
    shear_n: np.ndarray
    moment_nmm: np.ndarray

    @property
    def root_moment_nmm(self) -> float:
        return float(self.moment_nmm[0])


def span_loads(plan, aero_point, mass_kg: float, n_g: float) -> SpanLoads:
    """Shear and bending moment along the half span at n g.

    The VLM's own strip loading is used and then SCALED so the total lift
    equals n * weight. Scaling rather than recomputing is deliberate: the
    shape of the distribution is what the lattice knows well, while its
    absolute level at some assumed speed is not what we want anyway --
    the load case is a load factor, not an airspeed."""
    order = np.argsort(np.abs(aero_point.y_strip))
    y = np.abs(aero_point.y_strip)[order] * 1000.0
    cl = aero_point.cl_local[order]
    chord = aero_point.chord_strip[order] * 1000.0

    keep = y >= 0.0
    y, cl, chord = y[keep], cl[keep], chord[keep]
    shape = np.maximum(cl * chord, 0.0)          # lift per unit span, arb units
    half_span_mm = y.max()

    total = np.trapezoid(shape, y)               # one half wing
    target = n_g * mass_kg * G / 2.0             # newtons carried by one half
    l = shape * (target / max(total, 1e-12))     # N/mm

    # integrate inboard from the tip: shear then moment
    shear = np.array([np.trapezoid(l[i:], y[i:]) for i in range(len(y))])
    moment = np.array([np.trapezoid(shear[i:], y[i:]) for i in range(len(y))])
    return SpanLoads(y, l, shear, moment)


def tip_deflection_mm(loads: SpanLoads, spar: SparTube) -> float:
    """Double integration of M/EI from root to tip.

    A stiff wing is not a luxury on a trainer: a floppy panel washes out
    under load, which is stabilising until it flutters."""
    ei = spar.E_gpa * 1000.0 * spar.I_mm4       # MPa * mm^4 = N*mm^2
    curvature = loads.moment_nmm / max(ei, 1e-9)
    slope = np.concatenate([[0.0], np.cumsum(
        0.5 * (curvature[:-1] + curvature[1:]) * np.diff(loads.y_mm))])
    defl = np.concatenate([[0.0], np.cumsum(
        0.5 * (slope[:-1] + slope[1:]) * np.diff(loads.y_mm))])
    return float(defl[-1])


# -------------------------------------------------------- skin buckling


def max_rib_pitch_mm(panel_width_mm: float, skin_t_mm: float,
                     skin: SkinMaterial, load_mpa: float,
                     margin: float = 3.0) -> float:
    """Rib spacing that keeps the compression skin from buckling.

    Classical simply-supported plate buckling, k = 4 for a long panel:

        sigma_cr = k pi^2 E / (12 (1-nu^2)) * (t/b)^2

    Solved for b, the unsupported length, then divided by a margin of 3.
    A single-bead vase-mode skin is 0.45 mm of foamed PLA -- it buckles
    at trivial loads, which is exactly why the ribs are not decorative."""
    nu = 0.35
    k = 4.0
    if load_mpa <= 1e-6:
        return 1e6
    b = skin_t_mm * np.sqrt(k * np.pi**2 * skin.E_mpa
                            / (12.0 * (1 - nu**2) * load_mpa * margin))
    return float(min(b, panel_width_mm))


def skin_compression_mpa(loads: SpanLoads, plan, spar: SparTube,
                         skin_t_mm: float) -> float:
    """Bending stress carried by the skin at the root, roughly.

    The spar takes most of the moment; the skin takes what its own second
    moment claims. Treating the section as two areas sharing curvature is
    crude but the right order, and it is honest about being crude."""
    st = plan.at(0.0)
    chord_mm = st.chord_m * 1000.0
    h_mm = st.airfoil.t_max * chord_mm            # section depth at the root
    # thin-walled box: two caps of width ~0.6c at +/- h/2
    cap_w = 0.6 * chord_mm
    i_skin = 2.0 * cap_w * skin_t_mm * (0.5 * h_mm) ** 2
    ei_spar = spar.E_gpa * 1000.0 * spar.I_mm4
    ei_skin = 900.0 * i_skin
    share = ei_skin / max(ei_spar + ei_skin, 1e-9)
    m_skin = loads.root_moment_nmm * share
    return float(m_skin * (0.5 * h_mm) / max(i_skin, 1e-9))


# ------------------------------------------------------------ selection


@dataclass
class Structure:
    spar: SparTube
    spar_length_mm: float
    spar_mass_g: float
    rib_pitch_mm: float
    skin_t_mm: float
    n_limit: float
    n_ult: float
    stress_limit_mpa: float
    stress_ult_mpa: float
    tip_defl_mm: float
    tip_defl_pct: float
    skin_stress_mpa: float
    ok: bool
    notes: tuple[str, ...] = ()

    def report(self) -> str:
        mark = "OK" if self.ok else "FAILS"
        lines = [
            f"structure: {mark}",
            f"  spar          {self.spar.name} carbon tube, "
            f"{self.spar_length_mm:.0f} mm, {self.spar_mass_g:.1f} g",
            f"  stress        {self.stress_limit_mpa:.0f} MPa at "
            f"{self.n_limit:.1f} g   |   {self.stress_ult_mpa:.0f} MPa at "
            f"{self.n_ult:.2f} g ult (design {self.spar.sigma_design_mpa:.0f})",
            f"  tip deflection {self.tip_defl_mm:.1f} mm "
            f"({self.tip_defl_pct:.1f}% of semi-span) at limit load",
            f"  skin          {self.skin_t_mm:.2f} mm, "
            f"{self.skin_stress_mpa:.2f} MPa compression at the root",
            f"  ribs          every {self.rib_pitch_mm:.0f} mm "
            f"(buckling-driven)",
        ]
        lines += [f"  note          {n}" for n in self.notes]
        return "\n".join(lines)


def select(plan, aero_point, mass_kg: float, skin_t_mm: float,
           skin: SkinMaterial | None = None,
           n_limit: float = 3.5, ult_factor: float = 1.5,
           max_tip_defl_pct: float = 6.0,
           tubes=STOCK_TUBES) -> Structure:
    """The lightest stock tube that survives the load case, plus the rib
    pitch its skin needs. This is the 'pick the best settings' step, and
    it is a search over a catalogue, not a continuous optimum."""
    skin = skin or SkinMaterial()
    n_ult = n_limit * ult_factor
    loads_lim = span_loads(plan, aero_point, mass_kg, n_limit)
    loads_ult = span_loads(plan, aero_point, mass_kg, n_ult)
    half_span_mm = plan.half_span_m * 1000.0
    # the spar runs the full span, through both panels, plus a joiner
    spar_len = 2.0 * half_span_mm * 0.92

    chosen = None
    for tube in tubes:                         # ordered light -> heavy
        s_lim = tube.stress_mpa(loads_lim.root_moment_nmm)
        s_ult = tube.stress_mpa(loads_ult.root_moment_nmm)
        defl = tip_deflection_mm(loads_lim, tube)
        if (s_ult <= tube.sigma_design_mpa
                and defl / half_span_mm * 100.0 <= max_tip_defl_pct):
            chosen = (tube, s_lim, s_ult, defl)
            break
    if chosen is None:
        tube = tubes[-1]
        chosen = (tube, tube.stress_mpa(loads_lim.root_moment_nmm),
                  tube.stress_mpa(loads_ult.root_moment_nmm),
                  tip_deflection_mm(loads_lim, tube))

    tube, s_lim, s_ult, defl = chosen
    skin_sigma = skin_compression_mpa(loads_ult, plan, tube, skin_t_mm)
    root_chord_mm = plan.stations[0].chord_m * 1000.0
    pitch = max_rib_pitch_mm(root_chord_mm, skin_t_mm, skin, skin_sigma)

    notes = []
    ok = s_ult <= tube.sigma_design_mpa and defl / half_span_mm * 100 <= max_tip_defl_pct
    if not ok:
        notes.append("no stock tube passes -- reduce span, mass, or load factor")
    if pitch < 12.0:
        notes.append(f"rib pitch {pitch:.0f} mm is very tight; consider a "
                     f"thicker skin instead of more ribs")
    return Structure(
        spar=tube, spar_length_mm=spar_len, spar_mass_g=tube.mass_g(spar_len),
        rib_pitch_mm=float(np.clip(pitch, 8.0, 60.0)), skin_t_mm=skin_t_mm,
        n_limit=n_limit, n_ult=n_ult, stress_limit_mpa=s_lim,
        stress_ult_mpa=s_ult, tip_defl_mm=defl,
        tip_defl_pct=defl / half_span_mm * 100.0,
        skin_stress_mpa=skin_sigma, ok=ok, notes=tuple(notes),
    )
