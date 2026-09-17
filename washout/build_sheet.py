"""The build, generated from the design that was scored.

Everything here already exists inside an `Evaluation`. Until now it left
the program as prose in a `print()` call at the end of an export, or not
at all -- and the single most load-bearing sentence in the project,
"print with zero bottom layers", lived in one person's memory. A part
whose root face does not open is a part with an 8 mm tube that cannot go
in.

So this module writes it down, from the same object the design sheet is
drawn from, and every number in it is computed rather than typed:

  * the CG station in mm from the root leading edge, and how far the pack
    may drift before the static margin leaves its band
  * the spar cut list, from the corridors that were actually fitted
  * elevon throw in MILLIMETRES at the trailing edge, from the four-bar
    the linkage gate solved
  * which panel is the fuselage, and which face has to stay open
  * which gates were close, so the builder knows what not to add weight to

The one thing it will not do is invent. Where a quantity is a declared
allowance rather than a measurement it says so, and where something is
unmodelled -- bonding mass, the hand launch -- it is listed as unknown
rather than filled in.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .printing import elevons as elv
from .printing import vase


def cg_window_mm(ev) -> tuple[float, float]:
    """CG stations that keep the static margin inside its band.

    SM = (x_np - x_cg) / mac, so the band on SM is a band on x_cg --
    exactly, with no search. This is the number a builder needs and the
    one the program has always been able to compute and never printed:
    not "balance here" but "balance here, and here is how far out you may
    be before it stops flying the way it was designed to."
    """
    t = ev.trim
    mac = ev.plan.mac_m * 1000.0
    x_np = t.x_np_m * 1000.0
    lo = x_np - mac * ev.sm_band[1]
    hi = x_np - mac * ev.sm_band[0]
    return float(min(lo, hi)), float(max(lo, hi))


def pack_travel_mm(ev) -> float:
    """How far the battery may shift before the CG leaves that window.

    The pack is the heaviest single item and the one most likely to move,
    so the window in CG becomes a window in pack position scaled by the
    mass ratio: moving m_pack by d moves the CG by d * m_pack / m_total.
    """
    pack = next((i for i in ev.mass.items if i.name == "battery"), None)
    if pack is None or pack.mass_kg <= 0.0:
        return 0.0
    lo, hi = cg_window_mm(ev)
    cg = ev.trim.x_cg_m * 1000.0
    slack = min(cg - lo, hi - cg)
    return float(max(slack, 0.0) * ev.mass.total_kg / pack.mass_kg)


def throw_mm(ev) -> tuple[float, float] | None:
    """Elevon throw at the trailing edge, in mm, down and up.

    A transmitter is set up by measuring the trailing edge with a ruler,
    not by knowing the angle, so the angle the linkage gate solved is
    converted at the elevon's own mid-span chord."""
    if ev.linkage is None or ev.print_settings is None:
        return None
    from . import linkage as lkg

    down, up, _ = lkg.sweep(ev.linkage)
    eta_mid = 0.5 * (ev.print_settings.elevon_eta + 1.0)
    c_mm = ev.plan.at(eta_mid).chord_m * 1000.0
    arm = ev.print_settings.elevon_chord * c_mm
    return (float(arm * np.sin(np.radians(abs(down)))),
            float(arm * np.sin(np.radians(abs(up)))))


def tight_gates(ev, within: float = 0.15) -> list[str]:
    """Gates passing by less than `within` of their limit.

    A builder who knows the wing loading passed by 2% knows not to add a
    heavier battery; one who only sees "OK" does not."""
    out = []
    t = ev.trim
    if t is None:
        return out
    band = ev.cruise_band
    span = band[1] - band[0]
    if span > 0:
        margin = min(t.v_ms - band[0], band[1] - t.v_ms) / span
        if 0.0 <= margin < within:
            out.append(f"cruise speed {t.v_ms:.1f} m/s, {margin*100:.0f}% "
                       f"into the {band[0]:.0f}-{band[1]:.0f} m/s band")
    if ev.max_loading_gdm2 < 1e8:
        loading = ev.mass.total_kg * 1000.0 / (ev.plan.area_m2 * 100.0)
        margin = (ev.max_loading_gdm2 - loading) / ev.max_loading_gdm2
        if 0.0 <= margin < within:
            out.append(f"wing loading {loading:.1f} of "
                       f"{ev.max_loading_gdm2:.0f} g/dm2 "
                       f"({margin*100:.0f}% spare)")
    if ev.structure is not None:
        s = ev.structure
        margin = (s.spar.sigma_design_mpa - s.stress_ult_mpa) / s.spar.sigma_design_mpa
        if 0.0 <= margin < within:
            out.append(f"spar stress {s.stress_ult_mpa:.0f} of "
                       f"{s.spar.sigma_design_mpa:.0f} MPa at ultimate")
    return out


def render(ev, parts, settings: vase.PrintSettings) -> str:
    """The build sheet, as Markdown."""
    plan, t, m = ev.plan, ev.trim, ev.mass
    root_c = plan.stations[0].chord_m * 1000.0
    L: list[str] = []
    A = L.append

    A(f"# {plan.name}: build sheet")
    A("")
    A("Generated by `washout` from the design that was scored. Every number")
    A("here is computed from that design; nothing is typed in.")
    A("")
    A("**Absolute L/D is not quotable from this program.** Tier-0 profile")
    A("drag is an estimate and the measured polars are of a different")
    A("section; rankings are trustworthy, levels are not.")
    A("")

    # ---------------------------------------------------------- balance
    A("## Balance this first")
    A("")
    lo, hi = cg_window_mm(ev)
    A(f"- **CG: {t.x_cg_m*1000:.1f} mm aft of the root leading edge.**")
    A(f"- Acceptable window **{lo:.1f} to {hi:.1f} mm** — that is the static")
    A(f"  margin band {ev.sm_band[0]:.2f} to {ev.sm_band[1]:.2f} expressed as a")
    A(f"  position, with the neutral point at {t.x_np_m*1000:.1f} mm.")
    travel = pack_travel_mm(ev)
    if travel > 0.0:
        A(f"- The pack may sit **{travel:.0f} mm** either side of its drawn")
        A(f"  station before the CG leaves that window. Tape it, do not trust it.")
    A(f"- Design mass **{m.total_kg*1000:.0f} g** for the whole aircraft.")
    A("")
    A("Bonding mass is **not** in that figure. Adhesive follows from bonded")
    A("area, which this program does not yet compute, so expect a few grams")
    A("more and check the CG after assembly rather than before.")
    A("")

    # ------------------------------------------------------------- parts
    A("## Parts")
    A("")
    A("| part | what it is | height | mass | print |")
    A("|---|---|---|---|---|")
    for p in parts:
        kind = ("**fuselage** — the centre body" if p.name.endswith("p0")
                else "control surface" if p.role == "elevon"
                else "wing panel")
        A(f"| `{p.name}.stl` | {kind} | {p.height_mm:.0f} mm | "
          f"{p.mass_g():.1f} g | vase |")
    if getattr(ev, "fins", None) is not None:
        A(f"| `{plan.name}_tip_fin.stl` | tip fin, print TWO | — | — | solid |")
    A("")
    A(f"Everything above is **one half wing**. Print two of each and mirror.")
    A("")

    # ------------------------------------------------------------ slicer
    A("## Slicer")
    A("")
    A("| setting | value |")
    A("|---|---|")
    A("| mode | **spiral vase / spiralize outer contour, ON** |")
    A("| top layers | **0** |")
    A(f"| bottom layers | **0** — see below |")
    A(f"| extrusion width | {settings.extrusion_width_mm} mm |")
    A(f"| layer height | {settings.layer_h_mm} mm |")
    A(f"| nozzle | {settings.nozzle_mm} mm |")
    A(f"| material | {settings.filament_density_gcc:.2f} g/cc foamed PLA |")
    A("")
    A("**Zero bottom layers, and it is not optional.** The mesh is closed at")
    A("both ends of every panel; the root face is opened by the slicer. The")
    A("spar passes through the centreline and the electronics load before the")
    A("halves are joined, so a sealed root face is a panel with a carbon tube")
    A("that cannot go in. The tip fins and any solid parts print normally.")
    A("")

    # -------------------------------------------------------------- spars
    if ev.spar_fits:
        A("## Spar cut list")
        A("")
        A("| spar | tube | cut length | chord station | seat | reaches |")
        A("|---|---|---|---|---|---|")
        tube = ev.structure.spar.name if ev.structure else "?"
        for f in ev.spar_fits:
            A(f"| {f.spec.name} | {tube} | **{2*f.reach_mm:.0f} mm** | "
              f"{f.x_frac:.2f}c ({f.x_frac*root_c:.0f} mm aft of root LE) | "
              f"{f.anchor} skin | eta {f.reach_eta:.2f} |")
        A("")
        A("Each tube runs **tip to tip through the centre body** — one length,")
        A("not two meeting at the centreline. The bore is the shell's own")
        A("cavity: there is no hole to drill through the wall, but the tube")
        A("has to be seated against the skin the table names, because that is")
        A("where the clearance was solved for.")
        A("")

    # ----------------------------------------------------------- controls
    if ev.print_settings is not None and elv.has_elevon(ev.print_settings):
        ps = ev.print_settings
        A("## Control surfaces")
        A("")
        A(f"- Elevons run from **eta {ps.elevon_eta:.2f} to the tip**, "
          f"{ps.elevon_chord:.2f}c deep, and print as their own parts.")
        A(f"- Hinge on the **upper** surface. Gap {ps.hinge_gap_mm:.1f} mm; the")
        A(f"  elevon nose is bevelled {ev.max_elevon_deflect_deg + ps.hinge_margin_deg:.0f}°")
        A(f"  so it clears the cut face through the full {ev.max_elevon_deflect_deg:.0f}° of travel.")
        thr = throw_mm(ev)
        if thr is not None:
            A(f"- **Set the throws with a ruler at the trailing edge:** "
              f"{thr[0]:.1f} mm down, {thr[1]:.1f} mm up is the full")
            A(f"  mechanical range. The score assumes "
              f"{ev.max_elevon_deflect_deg:.0f}° is available and the linkage")
            A(f"  delivers more, so set the transmitter endpoints, not the horn.")
        if ev.linkage is not None:
            k = ev.linkage
            A(f"- Servo arm **{k.servo_arm_mm:.0f} mm**, horn hole "
              f"**{k.horn_arm_mm:.0f} mm** below the hinge axis, pushrod "
              f"**{k.rod_mm:.0f} mm** between centres.")
        A("")

    # ------------------------------------------------------- aeroelastic
    if getattr(ev, "aeroelastic", None) is not None:
        r = ev.aeroelastic
        A("## Speed limits")
        A("")
        A(f"- **Control reversal: {r.v_rev_ms:.0f} m/s.** Past this the")
        A(f"  elevons work backwards. It is the lower of the two and the")
        A(f"  firmer number, because the derivation shows it does not depend")
        A(f"  on where the elastic axis is.")
        A(f"- Torsional divergence: {r.v_div_ms:.0f} m/s.")
        A(f"- This design is scored at **{r.design_v_ms:.0f} m/s**, a margin")
        A(f"  of {r.margin:.2f}x.")
        A("")
        A("Both are computed from a single closed torsion cell, so they are")
        A("**lower bounds**: the rib truss makes the section multi-cell and")
        A("stiffer. Classical flutter is not modelled at all.")
        A("")

    # ------------------------------------------------------------- order
    A("## Assembly order")
    A("")
    A("The order matters because two of these steps cannot be undone once")
    A("the next one is done.")
    A("")
    A("1. Print everything. Check each panel's root face is open.")
    A("2. **Load the electronics and route the wiring while the halves are")
    A("   apart.** After the centre joint there is no access.")
    A("3. Slide the spars in and bond them, seated against the skin the cut")
    A("   list names.")
    A("4. Join the panels outboard, then the two halves at the centreline.")
    A("5. Hinge the elevons, fit the horns and the pushrods.")
    A("6. Glue the tip fins on.")
    A("7. Balance to the CG window above. Then set the throws.")
    A("")

    # --------------------------------------------------------- tightness
    tight = tight_gates(ev)
    if tight:
        A("## What is tight")
        A("")
        A("These gates passed by a small margin. Do not spend the margin.")
        A("")
        for g in tight:
            A(f"- {g}")
        A("")
    if ev.reasons:
        A("## This design does NOT pass its mission")
        A("")
        for r in ev.reasons:
            A(f"- {r}")
        A("")
        A("It is printable, and it is not the aeroplane the mission asked for.")
        A("")

    A("## Not modelled")
    A("")
    A("Stated so it is not mistaken for a clean bill of health:")
    A("")
    A("- **Classical flutter.** Divergence and control reversal ARE now")
    A("  computed (see above); flutter proper couples bending and torsion")
    A("  with the flow, needs the mass distribution, and is not modelled.")
    A("- **The hand launch.** Static thrust/weight is a proxy for it.")
    A("- **Bonding mass and joint strength.** The spar is sized; the bonded")
    A("  joints between panels are not.")
    A("- **Openings.** No hatch, firewall, servo pocket or wiring channel is")
    A("  cut in the geometry yet; the bays are reserved volume that is")
    A("  checked, not opened.")
    return "\n".join(L) + "\n"


def bom(ev, parts) -> str:
    """Every bought part, with a size and a count.

    Built from the masses the mission DECLARED and the geometry that was
    solved, so it cannot list a spar the design does not have or a servo
    count that disagrees with the linkage. Quantities are for the whole
    aircraft, not the half wing the STLs describe, because nobody buys
    half a servo."""
    L = [f"# {ev.plan.name}: bill of materials", "",
         "Quantities are for the **whole aircraft**. Sizes are what the",
         "design was solved around: substituting a different one invalidates",
         "the gate that cleared it.", "",
         "| qty | item | size | mass (all of them) | note |",
         "|---|---|---|---|---|"]
    tube = ev.structure.spar.name if ev.structure is not None else "?"
    for f in ev.spar_fits:
        m = next((i.mass_kg for i in ev.mass.items
                  if i.name == f"spar {f.spec.name}"), 0.0)
        L.append(f"| 1 | carbon tube — {f.spec.name} | {tube}, "
                 f"**{2*f.reach_mm:.0f} mm** | {m*1000:.0f} g | "
                 f"seats on the {f.anchor} skin at {f.x_frac:.2f}c |")
    known = {f"spar {f.spec.name}" for f in ev.spar_fits}
    for i in ev.mass.items:
        if i.name in known:
            continue
        # The mass budget's items are already per-AIRCRAFT totals, so the
        # mass column is not multiplied by the quantity -- "2 servos, 18 g"
        # means eighteen grams of servo in the aeroplane, not thirty-six.
        # The name carries the count for the pairs, so it is stripped out
        # of the name and put in the column where it belongs.
        qty = 2 if ("x2" in i.name or i.name == "tip fins") else 1
        name = i.name.replace(" x2", "").rstrip("s") if qty == 2 else i.name
        L.append(f"| {qty} | {name} | — | {i.mass_kg*1000:.0f} g | "
                 f"at {i.x_m*1000:.0f} mm aft of the root LE |")
    if ev.linkage is not None:
        k = ev.linkage
        L.append(f"| 2 | control horn | hole {k.horn_arm_mm:.0f} mm below the "
                 f"hinge axis | — | bonded to the elevon's lower surface |")
        L.append(f"| 2 | pushrod | {k.rod_mm:.0f} mm between centres | — | "
                 f"1 mm wire with a clevis, or a Z-bend |")
    if ev.print_settings is not None and elv.has_elevon(ev.print_settings):
        ps = ev.print_settings
        span_mm = (1.0 - ps.elevon_eta) * ev.plan.half_span_m * 1000.0
        L.append(f"| — | hinge tape | 2 x {span_mm:.0f} mm | — | "
                 f"upper surface, {ps.hinge_gap_mm:.1f} mm gap |")
    L += ["", f"Filament: about **{ev.mass.shell_kg*1000:.0f} g** of shell at "
          f"the declared {ev.print_settings.filament_density_gcc:.2f} g/cc, "
          f"plus adhesive.", ""]
    return "\n".join(L) + "\n"


def manifest(ev, parts, settings: vase.PrintSettings) -> dict:
    """Machine-readable: part -> profile, orientation, first-layer area.

    The bed rotation is already solved by `best_bed_rotation` and was
    only ever printed to a terminal. A slicer script can read this."""
    import numpy as _np

    out = {"design": ev.plan.name, "mass_g": round(ev.mass.total_kg * 1000, 1),
           "half_wing_parts": [], "mirror": True,
           "profiles": {
               "vase": {"spiralize": True, "top_layers": 0,
                        "bottom_layers": 0,
                        "extrusion_width_mm": settings.extrusion_width_mm,
                        "layer_height_mm": settings.layer_h_mm,
                        "nozzle_mm": settings.nozzle_mm},
               "solid": {"spiralize": False, "top_layers": 4,
                         "bottom_layers": 4,
                         "extrusion_width_mm": settings.extrusion_width_mm,
                         "layer_height_mm": settings.layer_h_mm,
                         "nozzle_mm": settings.nozzle_mm}}}
    for p in parts:
        deg, bx, by = p.best_bed_rotation()
        c0 = p.contours[0]
        area = 0.5 * abs(_np.dot(c0[:, 0], _np.roll(c0[:, 1], -1))
                         - _np.dot(c0[:, 1], _np.roll(c0[:, 0], -1)))
        out["half_wing_parts"].append({
            "file": f"{p.name}.stl", "profile": "vase", "role": p.role,
            "height_mm": round(p.height_mm, 1),
            "mass_g": round(p.mass_g(), 2),
            "bed_rotation_deg": round(deg, 1),
            "footprint_mm": [round(bx, 1), round(by, 1)],
            "first_layer_mm2": round(float(area), 1),
            "eta": [round(float(p.eta[0]), 4), round(float(p.eta[-1]), 4)]})
    if getattr(ev, "fins", None) is not None:
        out["half_wing_parts"].append(
            {"file": f"{ev.plan.name}_tip_fin.stl", "profile": "solid",
             "role": "fin", "quantity_per_aircraft": 2})
    return out


def write(ev, parts, settings: vase.PrintSettings, path) -> Path:
    import json as _json

    path = Path(path)
    path.write_text(render(ev, parts, settings), encoding="utf-8")
    (path.parent / "BOM.md").write_text(bom(ev, parts), encoding="utf-8")
    (path.parent / "MANIFEST.json").write_text(
        _json.dumps(manifest(ev, parts, settings), indent=2), encoding="utf-8")
    return path
