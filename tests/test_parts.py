"""The parts one spiral cannot be, and the gates that size them.

A vase-mode panel is a single closed contour per layer, and some of what
an aeroplane needs is not that: a hatch lid has two faces, a control horn
has a hole through it, a motor mount has five. They are separate printed
parts, and the rule for them is the rule for everything else here --
every dimension comes from the design that was scored, and every claim
about fit is a gate with a number.

The bugs pinned here all have the same shape: a companion part built to a
NOMINAL wing rather than to the one in hand.
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from washout import linkage as lkg
from washout.geom import cst
from washout.printing import parts, stl, vase
from washout.search.design import MISSIONS, Mission, build, evaluate

ASSETS = Path(__file__).resolve().parent.parent / "assets"
RESULTS = Path(__file__).resolve().parent.parent / "results" / "fleet"
WALL = 0.45


@lru_cache(maxsize=None)
def _built(name):
    index = json.loads((RESULTS / "index.json").read_text(encoding="utf-8"))
    d = json.loads((RESULTS / index[name] / "design.json").read_text(encoding="utf-8"))
    base = cst.load_selig(ASSETS / "mh45.dat")
    mission = getattr(Mission, name)()
    s = vase.PrintSettings(filament_density_gcc=0.55, spar_d_mm=mission.spar_d_mm,
                           bed_z_mm=250.0,
                           max_overhang_deg=mission.max_overhang_deg or 50.0)
    return evaluate(np.array(d["u"]), mission, base, s, want_panels=True), mission


# ------------------------------------------------------------ the mount

def test_the_motor_mount_builds_on_every_aircraft():
    """The flange's root ring is whatever the plate's triangulator made.

    `extrude_plate` triangulates the web with the flange roots as holes,
    and its boundary recovery SPLITS any hole edge the triangulation
    missed -- so the ring it hands back has more vertices than the
    polygon passed in, in an order only it knows. The sweep rebuilt its
    sections from the original samples instead of carrying that ring's
    own points forward, which nothing noticed on a synthetic wedge or on
    the trainer, and which refused to wall a 20-point ring against an
    18-point section on micro, whose thinner root needed the extra
    points. The mount was skipped with a printed excuse."""
    for name in MISSIONS:
        ev, mission = _built(name)
        part, gates = parts.mount_for(ev.plan, ev.print_settings,
                                      mission.powertrain, f"{name}_mount")
        rep = stl.manifold_report(part.tris)
        assert rep["watertight"], f"{name}: {rep}"
        assert part.mass_g() > 0.5, f"{name}: {part.mass_g():.2f} g is not a mount"
        assert gates, "a mount with no gates is a mount nobody checked"


def test_the_prop_clearance_is_measured_where_the_disc_reaches():
    """A pusher on a SWEPT wing loses clearance outboard.

    The trailing edge runs aft as the disc runs out, so it is the blade
    tips that are in danger and not the root -- measuring at the
    centreline says everything is fine while the tips are inside the
    wing. Two of the three aircraft turned out to be exactly that: micro
    17.6 mm INSIDE its own trailing edge, demon1 2.6 mm."""
    for name in MISSIONS:
        ev, mission = _built(name)
        d_in = mission.powertrain.prop.diameter_in
        at_tip = parts.prop_clearance_mm(ev.plan, d_in)
        at_root = (parts.disc_plane_mm(ev.plan)
                   - parts.te_station(ev.plan)(0.0))
        assert at_tip <= at_root + 1e-9, (
            f"{name}: clearance at the disc's edge ({at_tip:.1f} mm) must "
            f"never beat the centreline ({at_root:.1f} mm) on a swept wing")
        # and the gate the score uses agrees with the mount's own
        part, gates = parts.mount_for(ev.plan, ev.print_settings,
                                      mission.powertrain, f"{name}_mount")
        g = next(x for x in gates if x.name == "prop to trailing edge")
        assert g.value == pytest.approx(at_tip, abs=0.5), (
            f"{name}: the mount says {g.value:.1f} mm and the score says "
            f"{at_tip:.1f} -- two places measuring one trailing edge")


# ------------------------------------------------------------- the horn

def test_the_horn_is_built_to_the_surface_it_sits_on():
    """Its tongue follows the elevon's own curve, not a flat line.

    A foot laid flat on a curved skin rocks on the crown and the blade
    stands out of square, which on an 8 mm horn is a degree or two of
    trim nobody can find afterwards."""
    ev, mission = _built("trainer_v3")
    link = lkg.Linkage(hinge_x_mm=96.0, hinge_z_mm=6.0, servo_x_mm=70.0,
                       servo_z_mm=5.0, servo_arm_mm=11.0, horn_arm_mm=8.0,
                       side=+1.0, horn_dx_mm=15.5)
    eta = max(0.55, ev.print_settings.elevon_eta + 0.02)
    x_mid, length, depth, above = parts.horn_geometry(ev.plan, link, eta, WALL)
    assert length > parts.HORN_TONGUE_MM, "the socket holds the tongue plus a fit"
    assert above == pytest.approx(link.horn_arm_mm)

    # a station with real depth: the blade follows the skin, and the
    # tongue's top edge is not a straight line
    curve = lambda x: -0.004 * x * x
    part = parts.control_horn("horn", 3.0, above, parts._lkg.HORN_STRAP_MM,
                              curve, applied_nmm=100.0)
    assert stl.manifold_report(part.tris)["watertight"]
    assert part.quantity == 2, "one per side"
    g = next(x for x in part.gates if x.name == "horn socket holds")
    assert g.limit == parts.HORN_SAFETY
    ys = [curve(x) for x in np.linspace(0.0, parts.HORN_TONGUE_MM, 5)]
    assert max(ys) - min(ys) > 0.1, "the test's own surface must be curved"
    lo = part.verts[:, 1].min()
    assert lo == pytest.approx(min(ys) - 3.0, abs=1e-6), (
        "the tongue's floor follows the surface it sits under")

    # and a station with none refuses, rather than handing the
    # triangulator an outline that folds through itself
    with pytest.raises(ValueError):
        parts.control_horn("horn", 0.0, above, parts._lkg.HORN_STRAP_MM, curve)


def test_a_socket_is_gated_on_the_load_not_on_a_ratio():
    """The tongue is loaded as a COUPLE, so its LENGTH is most of the
    strength, and a rule in thicknesses cannot say that: it has no length
    in it. The servo's stall torque over its arm is the load case,
    because a servo that meets a jammed surface delivers it and the horn
    is what gives."""
    short = parts.socket_moment_capacity_nmm(5.0, 3.0)
    long_ = parts.socket_moment_capacity_nmm(10.0, 3.0)
    assert long_ > 3.5 * short, (
        "doubling the tongue must do far more than double the capacity")
    deep = parts.socket_moment_capacity_nmm(10.0, 6.0)
    assert deep == pytest.approx(2.0 * long_), "depth is linear, length is not"
    assert parts.socket_moment_capacity_nmm(10.0, 0.0) == 0.0


# -------------------------------------------------------------- the lid

def test_the_hatch_lid_starts_at_the_centreline():
    """A lid mirrored across both halves begins at an OUTBOARD end.

    That is where the bay has narrowed to almost nothing, so the first
    layer was 32 mm2 against the 40 mm2 a small part needs to stay on the
    bed -- and the part was 259 mm tall against a 250 mm envelope. One
    per half, begun at the centreline, starts on the widest and thickest
    section the lid has and is half as tall."""
    ev, mission = _built("trainer_v3")
    s = ev.print_settings
    lids = [parts.hatch_lid(ev.plan, s, c, f"lid_{i}")
            for i, c in enumerate(ev.bay_cuts)]
    lids = [l for l in lids if l is not None]
    assert lids, "the trainer's root bays are lidded and must produce lids"
    for lid in lids:
        assert lid.height_mm <= s.bed_z_mm, (
            f"{lid.name}: {lid.height_mm:.0f} mm will not fit the envelope")
        assert stl.manifold_report(stl.skin(lid)[1])["watertight"]
        # it starts at the CENTRELINE and runs outboard, so its first
        # layer is on the widest, thickest section it has
        assert lid.eta[0] < 0.02, (
            f"{lid.name}: begins at eta {lid.eta[0]:.3f}, not the centreline")
        assert lid.eta[-1] > lid.eta[0]
        assert lid.role == "lid"


def test_the_lid_is_as_thick_as_the_ledge_it_sits_in():
    """A declared 1.35 mm ledge under a printed 0.95 mm lid leaves the lid
    0.4 mm below the skin, which is a step the airflow trips on. The
    thickness is DERIVED -- two beads a clearance apart, so they weld into
    one curved plate, the rule the rib slits already use -- and the ledge
    is cut to whatever that comes to."""
    s = vase.PrintSettings()
    assert parts.lid_thickness_mm(s) == pytest.approx(s.lid_mm)
    assert s.lid_mm > s.extrusion_width_mm, "one bead is not a plate"
