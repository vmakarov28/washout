# From an STL to an aeroplane

The companion to [`ROADMAP.md`](ROADMAP.md). That document is about
whether the numbers are true. This one is about whether the parts are
buildable: everything that has to exist before `run.py export` emits a
set of files you print, assemble and fly, with **no CAD step in
between**.

Today the exporter emits a shell. It reserves *volume* for the battery,
receiver, ESC and servos and checks each fits where its mass sits. It
models **no openings, no mounts, no fasteners, no hinges and no
linkages at all**. Every one of those is currently cut, drilled and
glued by hand, and the tribal knowledge that makes it work leaves the
program only as prose.

---

# 0. The constraint that decides every answer

Read this first. Every requirement below is shaped by it, and most of
the obvious solutions are impossible because of it.

**Spiralize mode prints exactly one closed contour per layer, and print
Z is the span.** So each layer is one aerofoil section in the
chord-thickness plane, and the part is built outboard from the root.

Three consequences, and they are not intuitive:

**A surface that is horizontal in flight is vertical in the print.** The
chord plane contains chord (bed X) and span (print Z). A bay floor, a
shelf, a servo tray — anything you would think of as a floor — is a
vertical wall to the printer, at zero overhang. *Floors are the cheapest
feature in this geometry, not the hardest.*

**A surface that is vertical in flight, normal to the span, is
horizontal in the print.** The end wall of a pocket, a firewall normal
to the span, a closing rib — these are roofs. Vase mode cannot print a
roof. It has no top layers and cannot bridge.

**Therefore every internal feature is a detour of the skin loop whose
depth is a function of Z**, and a feature cannot start or stop abruptly.
It fades in and out, and the fade rate is an overhang.

## The four classes

Every requirement in this document is exactly one of these. Getting the
class right is most of the design work.

| class | what it is | cost | precedent in the code |
|---|---|---|---|
| **detour** | the skin loop dips into the interior and comes back | geometry only, inside vase mode | `printing/ribs.py` |
| **companion** | a separate part printed flat in normal mode | its own STL, its own gates, a mating interface | `aero/fins.py` |
| **insert** | bought hardware slid or screwed in | a fit gate and a line in the BOM | the spar tube |
| **gate** | a number that must hold, checked and reported | ~free | `printing/vase.py` |

A fifth class — **post-print operation** — is what we are trying to
drive to zero. Anything that cannot be moved into the first four is
listed in §6 with the reason.

## The ramp law

A feature that ramps in the thickness direction adds its own `dy` to
whatever `dx` that contour point already has from the wing's taper,
sweep and twist. `overhang_deg` measures the distance from each point of
layer *k+1* to the nearest point anywhere on layer *k*, so the binding
rule is a **quadrature**, not a sum:

```
    hypot(dx, dy) / dz  <=  tan(theta_max)

    =>  dy/dz  <=  sqrt( tan(theta_max)^2  -  (dx/dz)^2 )
```

This distinction is not pedantic — it changes the answer qualitatively.
Subtracting the wing's chordwise motion *linearly* (the obvious first
guess, and the way the rib truss budget is written) says the trainer's
root panel has 0.36 mm/mm of ramp available and a 20 mm feature needs
56 mm of span. In quadrature it has **0.80 mm/mm** and needs **25 mm**.
The difference is whether a battery bay is possible at all.

Measured on the fleet, over the chord band 0.12c–0.70c on the upper
surface, which is where a bay lives:

| panel | height | ramp rate | 20 mm feature needs | max depth, one ramp over half the panel |
|---|---|---|---|---|
| trainer p0 (the centre body) | 111 mm | 0.796 mm/mm | 25 mm | 44 mm |
| trainer p1 | 141 mm | 0.809 | 25 mm | 57 mm |
| demon1 p0 | 93 mm | 1.222 | 16 mm | 57 mm |
| micro p0 | **24.5 mm** | 0.597 | **34 mm — does not fit** | **7 mm** |
| micro p1 | 52 mm | 0.806 | 25 mm | 21 mm |

So: the trainer's centre body can carry a 24 mm deep bay with ramps at
both ends (35 mm of bay + 2 × 30 mm of ramp = 95 mm of a 111 mm panel).
**The micro's centre body cannot ramp anything useful** — 7 mm is the
most it can bury with one ramp — so its bay must run clear through the
panel and be closed by the joint at the far end, or move outboard into
p1. That is a per-aircraft architectural difference that falls straight
out of the print constraint, and no gate sees it today.

These figures are *indicative*: they take the worst point in the band
over the whole panel, not the point where a given feature actually sits.
The first deliverable in Phase 1 is to compute this per feature, at its
own station, as a gate.

## Two corollaries worth stating plainly

**An opening in vase mode is always a recess, never a hole.** The loop
has to go somewhere. To open the top of a bay, the contour runs along
the upper skin, dives at the front of the bay, crosses to one bead above
the *lower* skin, and climbs back at the rear. That is the rib detour
widened from a slit to a chord band — and the "floor" you get is the
lower skin, doubled. A true through-hole needs a companion part or a
knife.

**Bays should open at the root face and close outboard.** The root face
is already open (the spar has to get in, and the halves join there), so
a bay that starts at the centreline needs only *one* ramp, at its
outboard end. That halves the span it costs and it is how a real BWB is
laid out anyway.

---

# 1. Where the fleet actually stands

Measured on the three tracked designs, today. These are not
hypotheticals; they are what the current exporter would hand you.

## Every aircraft has a spar running through its battery

| | pack occupies | spar at | verdict |
|---|---|---|---|
| trainer | 0.291c – 0.586c | TE spar **0.540c** | inside the pack |
| demon1 | 0.197c – 0.570c | LE spar **0.210c** | inside the pack |
| micro | 0.239c – 0.582c | main spar **0.363c** | inside the pack, and through the receiver too |

`bay_fits` asks "is the section deep enough here". `spars.fit_all` asks
"can a tube reach". `panel_etas` asks "does the panel fit the bed".
**None of the three knows the other two exist**, so nothing has ever
asked whether the tube and the pack want the same space. They do, on
every aircraft in the fleet.

It is not necessarily fatal, and that is the interesting part: the
trainer's bay is 34.5 mm deep internally, the pack is 24 mm, and the
tube needs about 8.8 mm. They can stack — with **under 2 mm to spare**,
and only if the tube lies hard against one skin and the pack against the
other. Which raises the real problem:

## The spar has no vertical coordinate

`spars.place` chooses a chordwise station `x_frac` and nothing else.
`depth_at` returns the section thickness there. The tube is "somewhere
in the cavity". There is no Z, no seat, no retention, and therefore no
way to ask the stacking question above, or to place a floor above the
tube, or to know whether a rib detour lands on it.

Giving the spar a vertical seat is the single unblocking change in this
document. Four other requirements are waiting on it.

## Micro's battery bay straddles a print joint

The pack reaches η 0.170. Panel p0 ends at η **0.140**. The bay crosses
the split between two separately printed parts, and nothing notices.

## The root opening is a slicer setting, not geometry

`stl.skin` caps **both** ends of every panel — `bottom` at layer 0,
`top` at layer L−1. The mesh is closed. The root face is opened by
telling the slicer "zero bottom layers", which lives in a print
statement at the end of `do_export` and in a person's memory. Nothing
in the file says it. Hand these STLs to someone who slices them with
default settings and they get a sealed shell with an 8 mm tube that
cannot go anywhere.

## Minor, flagged rather than asserted

`spars.depth_at` subtracts both skin walls, and `SparSpec.needed_mm`
then adds `2 * wall` back on top of the tube diameter. Either that is a
deliberate extra margin or the wall is counted twice, costing about
0.9 mm of spar reach. Worth confirming before anyone tunes it.

---

# 2. The requirements

Each entry gives its **class** (§0), the geometry in *print*
coordinates, what has to be gated, and what it depends on. Ordered
within each group by what blocks what.

## A. Payload, access and routing

### A1. Give the spar a vertical seat — **gate + detour**, unblocks 4 others

Add a `z_frac` to `SparFit`, solved rather than assumed: the tube seats
against the upper or lower inner skin (whichever leaves the larger clear
volume for payload), following the camber line so it does not wander
across the section as the aerofoil changes outboard.

- **Gates:** tube clears both skins by ≥ 1 bead at every station it
  occupies; tube does not intersect any reserved bay volume; tube does
  not intersect a rib detour's floor.
- **Depends on:** nothing. Do this first.
- **Note:** the tube seat is also the natural place to put a *retention
  shoulder* — a short spanwise detour that stops the tube sliding out,
  which is otherwise a blob of epoxy applied by hand.

### A2. Reserved volume becomes real geometry — **detour**

Today a `Bay` is a number checked against section depth. It should
*cut the shell*: a bay floor (a spanwise detour at constant depth, zero
overhang), two chordwise end walls where the geometry allows, and a
ramped outboard closure at the rate §0 permits.

- **Geometry:** the detour's depth profile `d(z)` is trapezoidal — ramp
  in, constant, ramp out — with ramp slopes set by the measured local
  budget, not by a constant.
- **Gates:** the box fits the *cut* volume, not the section depth (this
  supersedes `bay_fits`); ramps within the per-station overhang budget;
  bay does not straddle a panel joint (**micro fails this today**);
  bay does not intersect a spar or a rib.
- **Depends on:** A1.

### A3. Hatch and lid — **detour + companion**

The bay is open-topped by construction (§0 corollary), so the "hatch"
is the companion lid and the rebate it sits in.

- **Geometry:** a shallow spanwise rebate around the bay mouth, one lid
  thickness deep, printed as part of the shell. The lid is a flat
  companion part following the local upper-surface curve, printed in
  normal mode like the tip fins already are.
- **Retention:** magnets in generated pockets, or a tongue at the
  forward edge and one screw aft. Screws into a printed boss in
  foamed PLA are weak; prefer the tongue-and-magnet pattern and gate
  the pocket depth.
- **Gates:** lid area vs. upper-skin buckling (see C4 — this is not
  cosmetic); rebate does not cross a spar corridor or a rib; lid
  printable within bed limits; magnet pocket ≥ 2 beads from the outer
  surface.
- **Depends on:** A2.

### A4. Battery retention — **detour**

A pack that moves is a CG that moves, and CG is the strongest lever on
trim in the whole aircraft. A hook-and-loop strap needs two slots
through the bay floor; a foam-pinch fit needs a dimensioned
interference.

- **Gates:** strap slots do not breach the lower skin's outer surface;
  retained CG travel ≤ the static-margin band's tolerance — i.e.
  compute how far the pack may shift before the margin leaves
  `[min_static_margin, max_static_margin]`, and report it as a number
  in the build sheet.
- **Depends on:** A2.

### A5. Wiring conduits — **detour**, cheap

A channel from the centre bay to each servo bay and to the motor. It
runs spanwise, so it is *constant in Z*: zero overhang, the cheapest
feature in the geometry.

- **Gates:** ≥ 4 mm bore for a servo lead with its connector; keeps one
  bead from skins, spars and rib detours; does not pass through a bay
  wall without a grommet detour; crosses each panel joint at a station
  where the joint is deep enough to pass the connector.
- **Depends on:** A1 (corridor deconfliction).

## B. Control surfaces

This group is the largest single block of manual work today, and all of
it is generatable.

### B1. The elevon becomes its own part — **companion**

The hinge line runs spanwise, which is print Z. **An elevon is
therefore a vase-mode part in exactly the same orientation as the wing
panels** — a small tapered wedge, printed root-down. This is the key
realisation that makes control surfaces tractable.

- **Geometry:** the wing panel's trailing edge is truncated at the hinge
  line (a spanwise cut, constant in Z, zero overhang). The elevon is
  built from the same `Planform` stations, aft of `1 - elevon_chord`,
  with a rounded leading edge to clear the wing's hinge pocket through
  the deflection range.
- **Measured today:** the trainer's elevon runs η 0.488 → tip, 0.278c
  deep: 37.2 mm chord at its root falling to 22.2 mm at the tip, in a
  section 10.8 mm thick falling to 5.6 mm.
- **Gates:** elevon root chord ≥ some minimum for a horn to anchor;
  hinge pocket clears the elevon nose at the full `max_elevon_deflect_deg`
  (12° trainer, and it must clear both ways); elevon printable as its own
  stack (bed footprint, first-layer area — a 22 mm × 5.6 mm tip section
  is well under the current 300 mm² adhesion gate, so the elevon prints
  root-down and tapers, which is the right way round).
- **Depends on:** nothing. This can start immediately.

### B2. The hinge — **insert + detour**

Three candidates, in order of preference:

1. **Pinned**: a 1–1.5 mm carbon rod through generated eyes on both
   parts. Eyes are spanwise detours, zero overhang. Strongest, fully
   generatable, needs the eyes to interdigitate — which costs
   point-count and a clearance gate.
2. **Taped**: a generated chamfer on both parts giving a clean
   V for the tape and the full deflection range. Trivially
   generatable, universally used, and it is a manual step.
3. **Live hinge**: a thinned spanwise web. Free geometrically, and a
   bad idea in foamed PLA — it is the one material property that
   makes a printed hinge fail in fatigue. Do not model it as viable
   without a test.

- **Gates:** hinge line ≥ 1 bead from the outer surface at its thinnest
  station (5.6 mm of section at the trainer's tip elevon, so this is
  not automatic); gap at full deflection ≥ 0.5 mm both directions.
- **Depends on:** B1.

### B3. Servo bays — **detour + companion**

A servo bay is the same trapezoidal detour as A2, but out in the wing
where the section is thin and the ramp budget is *better* (trainer p1:
0.809 mm/mm) while the depth available is *worse*.

- **Geometry:** pocket sized to the servo body with its mounting flange
  seated on a generated ledge; the flange ledge is chordwise, so it is a
  roof — it must be a ramped shelf, not a step, or a companion tray that
  drops in.
- **Gates:** section depth at the chosen station ≥ servo height +
  2 beads + lead clearance; bay does not straddle a panel joint; bay
  clear of spar corridors and rib detours; **the servo's output shaft
  lines up with the horn** (B4) within the linkage geometry's reach.
- **Depends on:** A1, B1, B4. This is the most constrained feature in
  the document and it should be *placed by a solver*, the way spars
  already are, not seated by hand.
- **Note:** servo position is currently a mass at a fixed 0.72c and
  nothing else. Making its station a solved output changes the CG, so it
  has to be inside the evaluation loop, not an afterthought — the same
  lesson `choose_structure` already learned.

### B4. Control horns — **companion**, with a generated socket

A horn projecting from the elevon surface is a bump on the contour at
one spanwise station: it must ramp in and out, ~10 mm tall at
~0.8 mm/mm ≈ 13 mm of span each side. That is affordable but it wastes
elevon span and prints a cantilever in foamed PLA.

Prefer a **companion horn** printed flat in normal PLA (or cut from
2 mm G10) dropping into a generated socket in the elevon — the same
pattern as the tip fins.

- **Gates:** socket depth ≥ 3 × horn thickness; socket ≥ 2 beads from
  the outer surface; horn hole pattern matches the linkage (B5).
- **Depends on:** B1.

### B5. The linkage actually delivers the throw — **gate**

The search computes `dcm_ddeg` and scores top speed assuming
`max_elevon_deflect_deg` is available. Nothing checks the mechanism can
deliver it. Servo arm radius, horn radius, pushrod length and the offset
between the servo output axis and the hinge axis together determine the
achievable deflection and its linearity.

- **Gates:** achievable deflection ≥ `max_elevon_deflect_deg` in both
  directions at ≤ 90% of servo travel; differential (up vs. down) within
  tolerance; pushrod does not foul the bay wall or the skin anywhere in
  the sweep.
- **Depends on:** B3, B4.
- **Why it matters:** this gate closes a real loop. demon1's entire
  speed objective is computed from an elevon deflection the aircraft has
  never been shown to be able to make.

## C. Structure and joints

### C1. Panel joints — **detour + insert**

Panels butt together on the spar and nothing else. There is no
alignment feature, no glue land, and no check that the joint carries the
bending moment the spar is sized for.

- **Geometry:** a shallow spigot/socket lip at each joint face (a
  chordwise feature = a roof, so it must be a ramped lip), plus a glue
  land of known area.
- **Gates:** bonded area × a conservative shear allowable ≥ the ultimate
  bending couple at that station, from `structure.span_loads` — which
  already computes the moment distribution and currently uses only its
  root value.
- **Depends on:** A1.

### C2. The centre joint — **detour + insert**

Two half-wings meet on the centreline. On a BWB with a fat body this is
the most loaded joint in the aircraft and it does not exist in the
model. The root faces are open (they must be), so this is a butt joint
on the spar plus bonded area.

- **Gates:** as C1, at the root moment; plus the joint must not obstruct
  the bay, the conduits, or the spar's insertion path.
- **Depends on:** A1, C1.

### C3. The open bay versus the torsion box — **gate**, and it is serious

Cutting the upper skin over a battery bay removes the closed torsion
box exactly where the torsional moment is highest. Two consequences the
program cannot currently see:

1. **Skin buckling.** `structure.max_rib_pitch_mm` sizes rib pitch from
   plate buckling of the compression skin. A bay mouth is a plate with a
   free edge, which buckles at a fraction of the closed value. Rib pitch
   around the bay must be resized, and that is mass.
2. **Torsional stiffness**, which is the input to the divergence and
   flutter work in [`ROADMAP.md` item 9](ROADMAP.md). An open section
   has a torsion constant one to two *orders of magnitude* below a
   closed one. Opening the centre bay and then asking whether demon1
   diverges at 44 m/s are the same question.

- **Gates:** closed-cell `GJ` retained ≥ some fraction over the bay
  span; buckling-driven rib pitch recomputed with the free edge.
- **Depends on:** A2, A3. **Blocks** the flutter work.

## D. Propulsion

### D1. Motor mount and firewall — **companion**

The motor is a 36 g point mass at 0.97c and nothing else: no firewall,
no bolt pattern, no thrust line. A 2205 needs an M3 16 × 16 mm pattern
on a face normal to the thrust line. That face is roughly normal to the
chord, i.e. *normal to bed X* — a vertical plane in the print, so it
**can** be a detour, but a bolt boss in foamed PLA will pull out.

Prefer a companion mount plate in solid PLA or ply, in a generated
pocket at the trailing edge of the centre body, bonded over a gated
area.

- **Gates:** bolt pattern clears the skin; bonded area × allowable ≥
  static thrust × a factor, and ≥ the gyroscopic and torque loads of the
  prop; mount face perpendicular to the thrust line within a stated
  tolerance (thrust-line error is a trim change the lattice never sees).
- **Depends on:** C2.

### D2. Prop clearance — **gate**

A pusher at 0.97c on a swept wing: the disc must clear the trailing
edge, the elevons through their full deflection, and the ground on a
belly landing.

- **Gates:** disc-to-TE clearance at `prop_diam_in` (a *design
  variable*, 4–7 in, so this is not a constant); clearance to the
  elevon at full down; the prop must not strike the wing under the tip
  deflection `structure.tip_deflection_mm` already computes.
- **Depends on:** B1, D1.

### D3. ESC and battery cooling — **detour**

A 30 A ESC at 380 W burst inside a sealed foam shell will cook. An inlet
at a high-pressure station and an outlet at a low-pressure one are both
detours; the lattice already knows the pressure distribution.

- **Gates:** inlet and outlet areas, and a stated (not simulated) mass
  flow; drag charged for it. Do not claim a cooling *rate* — the model
  cannot support one. Claim the area and say so.
- **Depends on:** A2.

## E. FPV — an optional payload, gated like any other

Treat FPV as a `Bay` family with its own mass, position and gates, so
that a mission either declares it or does not, and the CG consequences
are carried from the start rather than discovered on the bench.

### E1. Camera mount — **detour + companion**
A nose recess with a generated tilt (typically 15–30° up for cruise
attitude; the *right* number is the trim alpha the search already
computes, plus the pilot's preference). Lens must clear the skin
without a bulge that the drag model cannot see — or, if it bulges,
charge it.

### E2. VTX bay — **detour**
Same as A2, with D3's cooling requirement made mandatory rather than
optional: a VTX is a continuous 1–2 W in a foam box.

### E3. Antenna exits — **detour**
A generated exit tube for the VTX antenna and two for the AR630's
receiver antennas, which must sit at 90° to each other and away from
carbon. **The spar is carbon and runs the full span**, so the antenna
routing has to be deconflicted with A1 — this is a real constraint, not
a detail, and it is why the receiver bay's position matters.

- **Gates:** antenna exits ≥ 30 mm from any carbon member; the two rx
  antennas mutually perpendicular; exit tubes do not cross a spar
  corridor.

## F. Finishing and handoff

### F1. CG mark — **detour**, trivial, highest value per line of code
A scribed groove on the *outer* lower surface at the computed CG
station, so balancing is a fingertip check rather than a measurement
from a datum nobody marked. One line of geometry from `MassBudget`.

### F2. Part identity — **detour**
Panel name and orientation embossed on each part's root face. Four
near-identical tapered panels are easy to confuse and impossible to
reorder once printed.

### F3. Bed adhesion and orientation — **gate + output**
`best_bed_rotation` already computes the placement. Emit it, per part,
with the first-layer area the gate already measures.

---

# 3. New gates, in one table

Every one of these is a number with a limit, reported pass/fail, in the
project's existing style.

| gate | quantity | source |
|---|---|---|
| spar vs. bay | tube volume ∩ bay volume = 0 | A1, A2 |
| spar vertical fit | clearance to both skins ≥ 1 bead | A1 |
| bay straddle | bay η extent inside one panel | A2 |
| bay ramp | `dy/dz` ≤ `sqrt(tan²θ − (dx/dz)²)` per station | §0 |
| bay volume fit | box fits the *cut* cavity | A2 |
| lid rebate | ≥ 2 beads from outer surface | A3 |
| CG travel | pack shift before SM leaves its band | A4 |
| conduit bore | ≥ 4 mm, clear of everything | A5 |
| hinge pocket | clears elevon nose at full deflection, both ways | B2 |
| hinge web | ≥ 1 bead at the thinnest station | B2 |
| servo bay depth | servo height + 2 beads + leads | B3 |
| **linkage throw** | achievable deflection ≥ `max_elevon_deflect_deg` | B5 |
| pushrod sweep | no fouling anywhere in travel | B5 |
| joint shear | bonded area × allowable ≥ ultimate couple | C1, C2 |
| open-section GJ | retained fraction over the bay | C3 |
| free-edge buckling | rib pitch recomputed at the bay mouth | C3 |
| motor mount bond | area × allowable ≥ thrust × factor | D1 |
| prop clearance | to TE, to elevon at full down, at tip deflection | D2 |
| antenna isolation | ≥ 30 mm from carbon, rx pair ⊥ | E3 |

---

# 4. The output contract

What a complete `run.py export` must write. This is the definition of
done for the whole document.

```
out/<design>/
  design.json            the vector, as now
  design.png             the sheet, as now
  parts/
    <name>_p0..p3.stl        wing panels          vase mode
    <name>_elevon_l/r.stl    control surfaces     vase mode
    <name>_lid.stl           hatch lid            normal mode
    <name>_horn.stl          control horns x2     normal mode
    <name>_motor_mount.stl   firewall plate       normal mode
    <name>_tip_fin.stl       as now               normal mode
  profiles/
    vase.json              spiralize, 0 top, 1 bottom, width, layer h
    solid.json             the companion parts
  BUILD.md               generated: CG station and ballast, cut list,
                         throws in mm, assembly order, what was tight
  BOM.md                 every insert with a size and a count
  MANIFEST.json          part -> profile -> orientation -> first-layer area
```

Two things in that tree matter more than the rest.

**`BUILD.md` must be generated, not written.** Every number in it
already exists in the `Evaluation`: the CG station in mm from the root
LE, the ballast to reach it, spar lengths and diameters, elevon throws
in mm at the trailing edge from `dcm_ddeg`, which panel is the fuselage,
and which gates were close. Prose in a print statement is how that
knowledge got lost the first time.

**The slicer profiles must be files.** "Print with zero bottom layers"
is currently the single most load-bearing sentence in the project and it
exists only in a `print()` call and in someone's memory. A part whose
root face does not open is a part with an 8 mm tube that cannot go in.

---

# 5. Phases

Each phase ends with something testable. No phase depends on a later
one.

## Phase 1 — make the inside of the wing knowable *(unblocks everything)*

1. `SparFit.z_frac`: give the spar a vertical seat, solved. **(A1)**
2. A single `Volume` type and one occupancy check, so spars, bays, ribs
   and conduits are reserved against each other instead of each being
   checked alone. **(A1, A2)**
3. The three collision gates: spar∩bay, bay straddle, per-station ramp
   budget. **(§0, A2)**

*Acceptance:* the three tracked designs each report their real
collisions — the trainer's TE spar through its pack, demon1's LE spar
through its pack, micro's spar through both and its bay across the p0
joint — and a re-search returns designs that have none. Every finding in
§1 becomes a test with its physical reason in the docstring.

**Expect the fleet to get worse before it gets better.** Deconflicting
the spar and the pack will cost chord, depth or span. That is the
constraint arriving, not a regression.

## Phase 2 — the control surfaces

4. Elevon as its own vase part; hinge pocket and chamfer. **(B1, B2)**
5. Companion horn plus its socket. **(B4)**
6. Servo bay placed by a solver, inside the evaluation loop so its mass
   moves the CG. **(B3)**
7. The linkage throw gate. **(B5)**

*Acceptance:* the exported set includes elevons and horns, and the
linkage gate proves the deflection the speed objective has been assuming
all along.

## Phase 3 — the bay, the lid, and what they cost

8. Bay cut as real geometry with ramped closure. **(A2)**
9. Lid, rebate, retention; conduits. **(A3, A4, A5)**
10. Free-edge buckling and open-section `GJ`. **(C3)**

*Acceptance:* the pack loads without a knife, and the aircraft is
*charged* for the hole in its torsion box. This is the phase that feeds
the flutter work in `ROADMAP.md`.

## Phase 4 — joints and propulsion

11. Panel and centre joints with a shear gate off the existing moment
    distribution. **(C1, C2)**
12. Motor mount, prop clearance, cooling. **(D1, D2, D3)**

## Phase 5 — the handoff

13. Generated `BUILD.md`, `BOM.md`, `MANIFEST.json`. **(§4)**
14. Slicer profiles as files. **(§4)**
15. CG mark, part labels, bed placement. **(F1, F2, F3)**

*Acceptance:* someone who has never seen this repository prints the
folder and flies the result.

## Phase 6 — FPV, once the frame is done

16. FPV bays as a declared payload family, with antenna deconfliction
    against the carbon. **(E1, E2, E3)**

---

# 6. What still needs a human, and why

Honest list. Some of these are removable later; some are not.

| operation | why it survives | removable? |
|---|---|---|
| sliding the spar in and bonding it | insertion is an assembly motion | no — but the retention shoulder makes it foolproof |
| bonding the panel joints | adhesive is not printable | no — but the glue land and area are gated |
| hinging the elevons | tape or rod, applied by hand | partly: the pinned option (B2.1) reduces it to a rod |
| soldering the ESC and motor | electrical | no |
| binding the receiver | electrical | no |
| final balance check | the mark makes it 10 seconds | reduced, not removed |
| control throw setup | transmitter endpoints | reduced: the build sheet states the throw in mm |

What should **not** survive: cutting the hatch, cutting the elevons,
carving servo pockets, drilling horn holes, improvising a motor mount,
guessing the CG. Every one of those is in §2 with a class and a gate.

---

# 7. What not to do

- **Do not add a "print flat" mode.** A horizontal slice of a flat wing
  is multiple contours and spiralize cannot express it. This is already
  a project rule and every feature above respects it.
- **Do not model a feature the gates cannot see.** A cooling duct with a
  claimed mass flow, a vortex generator, a sealed hinge gap — if the
  lattice and the strip-theory drag model cannot see the benefit, the
  optimizer will pay real weight for an invented one. State the area;
  do not claim the rate.
- **Do not let a bay be a soft constraint.** A pack that does not fit is
  not a design that scores slightly worse. Make the collision
  unrepresentable if it can be, and infeasible if it cannot.
- **Do not size the servo bay outside the loop.** `choose_structure`
  already taught this lesson at a cost of four generations: ribs added
  after the verdict were 8–22 g aft of the CG and turned feasible
  designs infeasible. A servo is 9–18 g at 0.72c. Same shape of mistake,
  waiting.
- **Do not trust a live hinge in foamed PLA** without a fatigue test.

---

# 8. The one line summary

Give the inside of the wing a coordinate system and an occupancy check,
and most of this document becomes bookkeeping. Without it — and today
there is no vertical coordinate for the spar at all — every feature
added is another thing that might be sharing space with the battery.
