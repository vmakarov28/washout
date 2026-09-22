"""Every fast path in tier 0 gives the answer its slow path gave.

Tier 0 is contracted at about 0.4 s an evaluation (CLAUDE.md) and had
drifted to 12 s, almost none of it physics: 160 000 calls to
`Planform.at` for 11 000 distinct stations, interior depths asked one
chord station at a time, and two exhaustive geometric searches -- the
overhang's vertex-to-segment distances and the bore's inscribed circles
-- run over every layer of every part. gen6's searches were cut to
popsize 4 to fit in a night because of it.

The fixes are memoisation, arrays and PRUNING, and pruning is where an
optimisation quietly changes an answer: a bound that is not really a
bound, or a tie broken differently. So each one is checked here against
the loop it replaced, copied verbatim as the reference, on the fleet's
own geometry and on geometry deliberately made rougher than any wing.
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from washout import spars as sp
from washout.geom import cst
from washout.geom import interior as it
from washout.printing import vase
from washout.search.design import MISSIONS, Mission, build, unit_to_physical

ASSETS = Path(__file__).resolve().parent.parent / "assets"
RESULTS = Path(__file__).resolve().parent.parent / "results" / "fleet"
WALL = 0.45


@lru_cache(maxsize=None)
def _plan(name):
    index = json.loads((RESULTS / "index.json").read_text(encoding="utf-8"))
    d = json.loads((RESULTS / index[name] / "design.json").read_text(encoding="utf-8"))
    base = cst.load_selig(ASSETS / "mh45.dat")
    mission = getattr(Mission, name)()
    u = np.array(d["u"])
    return u, mission, build(u, mission, base)


@lru_cache(maxsize=None)
def _panels(name):
    u, mission, plan = _plan(name)
    p = unit_to_physical(u)
    s = vase.PrintSettings(filament_density_gcc=0.55, spar_d_mm=8.0,
                           bed_z_mm=250.0, ribs=True, rib_pitch_mm=25.0,
                           elevon_chord=p["elevon_chord"],
                           elevon_eta=p["elevon_eta"],
                           spar_avoid=((0.25, 5.0),))
    return vase.build_panels(plan, s, z_step_mm=1.0)


# ------------------------------------------------------ the references


def _overhang_reference(stack, stride=4):
    """`overhang_deg` as it was before pruning, verbatim."""
    worst, worst_z = 0.0, 0.0
    c = stack.contours
    for k in range(0, len(c) - stride, stride):
        d = vase._dist_to_segments(c[k + stride], c[k])
        rise = float(stack.z_mm[k + stride] - stack.z_mm[k])
        ang = np.degrees(np.arctan2(d.max(), max(rise, 1e-9)))
        if ang > worst:
            worst, worst_z = float(ang), float(stack.z_mm[k])
    return worst, worst_z


def _bore_reference(stack):
    """`spar_fit` as it was before pruning, verbatim."""
    s = stack.settings
    fracs = stack.spar_x_local or (s.spar_x_frac,)
    best, pinch_z = np.inf, 0.0
    for k, layer in enumerate(stack.contours):
        lo_x, hi_x = layer[:, 0].min(), layer[:, 0].max()
        chord = hi_x - lo_x
        for f in fracs:
            gap = (vase._inscribed_gap(layer, lo_x + float(f) * chord)
                   - s.extrusion_width_mm - s.spar_clearance_mm)
            if gap < best:
                best, pinch_z = float(gap), float(stack.z_mm[k])
    return best, pinch_z


def _roughened(stack, seed, scale_mm):
    """The same part with every vertex jittered. A wing moves smoothly
    from layer to layer, which is exactly the property the overhang
    bound leans on -- so the bound is also checked where it is weakest."""
    rng = np.random.default_rng(seed)
    c = stack.contours + rng.normal(0.0, scale_mm, stack.contours.shape)
    return vase.LayerStack(z_mm=stack.z_mm, eta=stack.eta, contours=c,
                           settings=stack.settings, name=stack.name,
                           z_step_mm=stack.z_step_mm, has_ribs=stack.has_ribs,
                           spar_x_local=stack.spar_x_local)


# ------------------------------------------------------------ the tests


def test_the_station_memo_is_the_loft():
    """`Planform.at` is memoised because one evaluation asks for the same
    6 000 stations about 25 times each. A memo must return what the loft
    returns, and since it hands back the SAME object each time, the
    coefficient arrays inside it must be read-only -- otherwise one caller
    editing a section in place edits every later answer."""
    u, mission, plan = _plan("trainer_v3")
    base = cst.load_selig(ASSETS / "mh45.dat")
    fresh = build(u, mission, base)
    for e in np.linspace(0.0, 1.0, 37):
        plan.at(float(e))                       # warm
    for e in np.linspace(0.0, 1.0, 37):
        a, b = plan.at(float(e)), fresh.at(float(e))
        assert a is plan.at(float(e))
        assert (a.chord_m, a.x_le_m, a.z_le_m, a.twist_deg) == \
               (b.chord_m, b.x_le_m, b.z_le_m, b.twist_deg)
        assert np.array_equal(a.airfoil.au, b.airfoil.au)
        assert np.array_equal(a.airfoil.al, b.airfoil.al)
        with pytest.raises(ValueError):
            a.airfoil.au[0] = 0.0


@pytest.mark.parametrize("name", MISSIONS)
def test_overhang_pruning_is_exact(name):
    """The pruned overhang returns the unpruned loop's angle AND its Z.

    It skips work using an upper bound: the distance to the 7 segments of
    the layer below around the same index can only be LARGER than the
    distance to all of them. Checked on every fleet panel, and on the same
    panels jittered by up to half a millimetre per vertex, where layers
    no longer follow each other smoothly and the bound is at its loosest."""
    for stack in _panels(name):
        assert vase.overhang_deg(stack) == _overhang_reference(stack)
        for seed, scale in ((1, 0.05), (2, 0.5)):
            rough = _roughened(stack, seed, scale)
            assert vase.overhang_deg(rough) == _overhang_reference(rough)


@pytest.mark.parametrize("name", MISSIONS)
def test_bore_pruning_is_exact(name):
    """The pruned bore returns the full scan's diameter AND its pinch Z.

    It prunes with a LOWER bound: the middle one of the fifteen centres
    the scan tries. Checked on every fleet panel with a spar station, and
    on rough copies whose pinch is somewhere a smooth wing would never
    put it."""
    for stack in _panels(name):
        if stack.role != "wing":
            continue
        for fr in ((0.25,), (0.18, 0.55)):
            st = vase.LayerStack(z_mm=stack.z_mm, eta=stack.eta,
                                 contours=stack.contours,
                                 settings=stack.settings, name=stack.name,
                                 z_step_mm=stack.z_step_mm,
                                 has_ribs=stack.has_ribs, spar_x_local=fr)
            assert vase.spar_fit(st) == _bore_reference(st)
            rough = _roughened(st, 3, 0.3)
            assert vase.spar_fit(rough) == _bore_reference(rough)


def test_interior_depths_over_arrays_match_one_at_a_time():
    """`depth_mm`, `z_interval` and `overlap_mm` take arrays of chord
    stations now. The array answer must be the scalar answer, element by
    element, including the EMPTY interval a too-shallow station returns."""
    _, mission, plan = _plan("trainer_v3")
    xs = np.linspace(0.02, 0.98, 23)
    for e in (0.0, 0.3, 0.7, 0.95):
        d = it.depth_mm(plan, e, xs, WALL)
        assert np.allclose(d, [it.depth_mm(plan, e, float(x), WALL) for x in xs],
                           rtol=0.0, atol=1e-12)
        for anchor in it.ANCHORS:
            v = it.Volume("box", 0.2, 0.6, 0.0, 1.0, 14.0, anchor=anchor,
                          offset_mm=0.5)
            lo, hi = v.z_interval(plan, e, xs, WALL)
            for x, l_, h_ in zip(xs, lo, hi):
                ls, hs = v.z_interval(plan, e, float(x), WALL)
                assert abs(l_ - ls) < 1e-12 and abs(h_ - hs) < 1e-12


def _overlap_reference(a, b, plan, wall_mm, n_eta=9, n_x=5):
    """`overlap_mm` as it was before it took arrays, verbatim."""
    e0, e1 = max(a.eta0, b.eta0), min(a.eta1, b.eta1)
    if e1 < e0:
        return 0.0
    worst = 0.0
    for eta in np.linspace(e0, e1, n_eta):
        ax, bx = a.band(plan, float(eta)), b.band(plan, float(eta))
        x0, x1 = max(ax[0], bx[0]), min(ax[1], bx[1])
        if x1 < x0:
            continue
        for x in np.linspace(x0, x1, n_x):
            alo, ahi = a.z_interval(plan, float(eta), float(x), wall_mm)
            blo, bhi = b.z_interval(plan, float(eta), float(x), wall_mm)
            if ahi < alo or bhi < blo:
                continue
            dz = min(ahi, bhi) - max(alo, blo)
            if dz <= 0.0:
                continue
            dx_mm = (x1 - x0) * plan.at(float(eta)).chord_m * 1000.0
            de_mm = (min(a.eta1, b.eta1) - max(a.eta0, b.eta0)) \
                * plan.half_span_m * 1000.0
            worst = max(worst, min(dz, dx_mm, de_mm))
    return float(worst)


def test_overlap_over_arrays_matches_the_loop():
    """Every pair of the fleet's payload boxes and spar tubes, and a grid
    of synthetic boxes stacked, graze-deep and clear, against the loop
    `overlap_mm` replaced."""
    for name in MISSIONS:
        _, mission, plan = _plan(name)
        vols = [it.bay_volume(b.name, b.x_frac, b.box_mm, plan, offset_mm=WALL)
                for b in mission.bays]
        for f in sp.fit_all(plan, mission.spars, WALL):
            vols.append(it.spar_volume(f, WALL, plan))
        for h in (4.0, 12.0, 30.0):
            for anchor in it.ANCHORS:
                vols.append(it.Volume(f"s{h}{anchor}", 0.2, 0.5, 0.0, 0.4, h,
                                      anchor=anchor))
        for i, a in enumerate(vols):
            for b in vols[i + 1:]:
                got = it.overlap_mm(a, b, plan, WALL)
                want = _overlap_reference(a, b, plan, WALL)
                assert abs(got - want) < 1e-12, (name, a.name, b.name, got, want)


def test_the_spar_reach_scan_over_arrays_matches_the_loop():
    """`reach_many` scans every candidate chord station together; each
    must stop at the station the one-at-a-time scan stopped at."""
    for name in MISSIONS:
        _, mission, plan = _plan(name)
        for spec in mission.spars:
            xs = np.linspace(spec.x_lo, spec.x_hi, 25)
            got = sp.reach_many(plan, xs, spec, WALL)
            need = spec.needed_mm(WALL)
            for x, r in zip(xs, got):
                want = 1.0
                for e in np.linspace(0.0, 1.0, 160):
                    if sp.depth_at(plan, float(e), float(x), WALL) < need:
                        want = float(max(e - 1.0 / 159, 0.0))
                        break
                assert r == want, (name, spec.name, x, r, want)
