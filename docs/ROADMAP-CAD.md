# The third dimension

The third roadmap. [`ROADMAP.md`](ROADMAP.md) is whether the numbers are
true; [`ROADMAP-BUILD.md`](ROADMAP-BUILD.md) is whether the parts are
buildable without a CAD step. This one is about the object neither of
them has: **a three-dimensional model of the aircraft as it will be
built**, which a geometry kernel can interrogate, and which leaves the
program as a STEP file a person can open, measure and modify.

It starts with two defects, because they are the argument for everything
after them.

---

# 0. What a 2D pipeline cannot see

Every gate in this program looks at one aerofoil section at a time. The
printer does too -- one contour per layer -- which is why the pipeline was
built that way, and it has served well: every printability claim, the
interior coordinate system, the spar seat, the rib corridors, the bore.
But some facts exist only between sections, and on 2026-09-22 two of them
turned out to be wrong on every aircraft in the fleet.

## 0.1 The spar was a noodle

`spars.reach_of` asks whether the section is deep enough at `(eta,
x_frac)`, station by station. That lets the tube follow its corridor
wherever the loft takes it -- aft with the sweep, up with the dihedral.
A carbon tube is straight.

Measured on the loft before 1.1 landed (flight frame, tube radius plus
clearance, one bead of wall each side, root seat free; lines swept over
0-60 deg of sweep and -4 to 24 deg of dihedral):

| | solver claimed | one straight tube tip to tip | best straight tube per half | V joiner it needs |
|---|---|---|---|---|
| trainer LE, 8 mm | eta 1.00 | 0.25 | 0.68 | 68 deg in plan, 12 in front |
| trainer TE, 8 mm | 0.76 | 0.43 | 0.67 | 60 / 12 |
| demon1 LE, 8 mm | 1.00 | 0.27 | 0.66 | 60 / 16 |
| demon1 TE, 8 mm | 0.88 | 0.42 | 0.68 | 24 / 16 |
| micro_fpv, 6 mm | 1.00 | 0.26 | 0.69 | 100 / 24 |

`BUILD.md` told the builder to run each tube "tip to tip through the
centre body -- one length, not two meeting at the centreline". On every
aircraft that tube leaves the skin between eta 0.25 and 0.43. And
everything downstream was computed on tube that is not there outboard of
about 0.68: the spar mass (charged at twice the claimed reach), the tip
deflection (integrated to the tip with the tube's EI), and the torsional
stiffness -- the tubes are 33-35% of GJ, `gj_spars_nmm2` added them at
every station regardless of reach, and control reversal is the binding
aeroelastic limit on all three aircraft.

## 0.2 The printed panel was straight

`build_stack` places each layer with the station's `x_le`, chord and
twist, and never its `z_le`. Every panel therefore prints straight, and
the dihedral the lattice scored as a smooth curve arrives as a polyline
with a kink at every joint. Nothing emitted the kink angle.

| | every control station a break (HEAD) | minimal split (uncommitted) |
|---|---|---|
| | panels / worst deviation / worst kink | panels / worst deviation / worst kink |
| trainer | 5 / 2.7 mm / 12.3 deg | 3 / 3.8 mm / 14.4 deg |
| demon1 | 5 / 1.8 mm / 6.6 deg | 3 / 3.1 mm / 8.8 deg |
| micro | 4 / 1.3 mm / 13.6 deg | 2 / 4.3 mm / 20.8 deg |
| micro_fpv | 4 / 4.1 mm / 12.1 deg | 2 / **9.0 mm** / 17.2 deg |

A second-order effect of the same cause: each layer is the loft's
`y = const` section stood perpendicular to a tilted panel axis, so the
assembled wing's vertical sections are `1/cos(phi)` thicker than the
scored ones -- **+13%** on the trainer's 28 deg outer panel, +7% on
micro_fpv's p1. The strip-theory drag, the spar depth and the lattice all
used the thinner section.

## 0.3 Why both were invisible

Neither is a subtle model error. They are the same blind spot: a tube is
straight and a panel is straight, and both facts live along the span,
between the sections every gate inspects. A B-rep of the as-built
aircraft -- tubes as cylinders, panels as solids in the flight frame --
makes both impossible to miss, because a kernel asked "is this cylinder
inside that solid" answers in 3D.

**That is the case for the STEP work.** The file is the by-product. The
product is a second, independent geometric model of the built aircraft
that can referee the first.

## 0.4 Smaller things of the same shape

- **The contour is documented as the bead centreline** (`thicken_for_nozzle`)
  but a slicer puts the *outer edge* of the bead on the STL surface.
  That is 0.225 mm on every surface either way -- 6% of a 7 mm tip
  section. The OML STEP has to say which surface it is. One test print
  and a caliper settle it; do that before the conformance gate is
  trusted.
- **`structure.select` sizes the tube for a full-span spar** (`spar_len =
  2 * half_span * 0.92`, deflection integrated to the tip on the tube's
  EI). Outboard of the tube's real reach the shell carries the load.
- **Tier 0 is 15-30x off its own contract.** CLAUDE.md says a
  tier-0 evaluation is about 0.4 s. One trainer evaluation measured
  11.9 s, micro_fpv 6.2 s, with or without panel geometry, and almost
  none of it is physics: 161 000 calls to `Planform.at`, 4.8 s in
  `vase.check` (brute-force overhang distances), 3.6 s in
  `interior.depth_mm` through scalar `cst_y`, 2.7 s placing spars, 2.6 s
  seating bays. gen6's searches were cut to popsize 4 over 22
  iterations to fit in seven hours because of it.

---

# 1. Make the built aircraft the scored one

Before any file format: close both divergences in the model that the
search scores. Each is a change to what gets printed, so each is gated
and each moves the fleet.

## 1.1 Straight spars -- P0

A spar is a straight tube per half wing, from the centreline to where it
leaves the section. The two halves meet in a V joiner whose angles are
the tube's own sweep and dihedral, doubled. A tube parallel to the span
(no sweep, no dihedral) is the special case that runs tip to tip in one
piece.

- **Fit.** For each `SparSpec`, candidate lines through a root seat
  (chordwise across the declared band, vertically across the root
  cavity) aimed at the same relative seat at a target station. Reach is
  the first station, scanning outward, where the circle of tube radius
  plus clearance no longer fits inside the section with a bead of wall
  on each side. Among lines that meet `min_spar_reach_frac`: clearest
  against the reserved volumes, then furthest, then nearest the middle of
  the band -- the same ordering `place` already uses, now on lines.
- **The tube as an interior volume.** Its chordwise band and its height
  above the lower skin both change along the span. `interior.overlap_mm`
  is duck-typed on `band` and `z_interval`, so a `TubeVolume` with the
  line inside it drops into every existing clash check.
- **Downstream, from the line.** Mass is the 3D tube length, doubled, at
  the tube's own centroid x. GJ adds the tube only at stations it
  reaches. The rib corridors in each panel cover the tube's actual chord
  fraction over that panel. The bore gate measures clearance at the
  tube's actual centre in each layer, and only in layers it reaches.
- **Build sheet and BOM.** Cut length per half, root seat in mm, the
  tube's sweep and dihedral, and the V joiner's two angles. No
  "tip to tip" unless it is.

*Acceptance:* a test that a straight tube along the fitted line is
inside the section at every sampled station up to reach and outside just
beyond it; the fleet re-checked, with the new misses stated rather than
tuned away.

## 1.2 Panels are true slices of the loft -- P0

A printed layer is the loft cut by a plane perpendicular to the panel's
own axis, so the panel follows the dihedral curve inside itself (a lean
in print Y, charged against the overhang budget by the same quadrature
law as sweep) and its sections are the real perpendicular sections.

- **The frame.** Each panel has an axis in the span-height plane. The
  centre body's is horizontal, so the two halves' root faces are both the
  symmetry plane and mate flat. The others run between joint pivots.
- **Joints.** Two adjacent panels' end faces are perpendicular to two
  different axes, so they cannot both be the same plane. They are
  pivoted about the joint section's top point when the dihedral steepens
  outboard (bottom when it flattens), which makes interpenetration
  impossible by construction and leaves an open wedge on the far skin.
  Its angle and its opening in mm are reported per joint on the build
  sheet: that wedge is glue, and the builder needs to know how much.
- **Slicing.** Each point of the reference section is carried along the
  loft's local span direction to the layer plane. The displacement is
  first order in the distance moved (a few mm at most), with the
  section's chord, sweep, dihedral and twist rates included and the
  aerofoil's own shape change left out; the error that leaves is bounded
  in the docstring and measured against an exact root-find in a test.
- **Elevons share their wing panel's frame**, so the hinge faces of the
  two parts are the same surface.

*Acceptance:* printed contours, mapped back to the flight frame, lie on
the loft within 0.1 mm away from joints; every joint's kink and wedge
are on the build sheet; the minimal panel split is then safe to commit
and is.

## 1.3 Tier 0 at its contract again -- P1

No change to any number. Memoise `Planform.at`, evaluate interior depths
over arrays rather than one chord station at a time, prune the overhang
measurement with an upper bound so only candidate worst points pay for
the exact point-to-segment search, and replace the bore scan with the
tube-centre distance once 1.1 lands.

*Acceptance:* a golden snapshot of every tracked design's verdict --
score, reasons, masses, gate values -- taken before the change and
compared after it to 1e-9; each fast path pinned against its slow
reference in a test.

---

# 2. The CAD export

## 2.1 Build the B-rep from design intent, not from the print encoding

The first attempt (`out/fpv/s4/step/`, never committed) lofted the
*printed contour*: 247 points a loop with rib slits one bead wide. Smooth
fits rang through the slits -- one came back with a bounding box 75 m
wide on a 240 mm part -- so it fell back to a ruled loft through 40
polygons. That one was faithful, volume within 0.002%, and useless: 9,635
faces, 30 MB, one B-spline patch per quad strip. A person cannot fillet
it, shell it, section it or edit it, and most CAD packages take minutes
to open it.

The lesson is that the contour is a *manufacturing encoding* of the
design -- one closed loop, ribs as detours of the skin -- and not the
design. The design is the loft. Build the B-rep from the loft, derive
the contours from the same loft, and let each referee the other.

The sections are exact. The class exponents are 0.5 and 1, so with the
substitution `x = t^2` every CST surface is a polynomial in `t`: `y(t) =
t (1 - t^2) S(t^2) + t^2 dz_te`. A B-spline fitted in `t` has nothing to
ring against. Only the nozzle-thickened trailing edge is not polynomial,
and it is smooth.

## 2.2 Topology: what "usable" means, as rules

A STEP file is usable when a person can open it quickly, understand what
each face is, and apply ordinary CAD operations to it -- shell, fillet,
offset, section, boolean -- without the kernel failing. Each rule below
is a gate or a test.

1. **One face per logical surface.** A wing panel is a wrapped skin, a
   trailing-edge face, and two end faces: four faces, not 9,635. An
   elevon adds its chamfer and its nose face. Face count is gated per
   part.
2. **The seam goes at the trailing edge, never the leading edge.** The
   skin wraps from the upper trailing edge round the nose to the lower
   one, so the leading edge -- where curvature is highest and where
   offsets and fillets most often fail -- is the smooth interior of a
   face.
3. **Planes are planes.** End faces, joint faces and the hinge cut on a
   straight hinge are written as PLANE entities, spar tubes as
   CYLINDRICAL_SURFACE. An analytic surface is what makes a feature
   selectable, dimensionable and mateable in CAD; a B-spline that happens
   to be flat is none of those.
4. **Continuity inside faces.** Cubic, C2 across the chord and along the
   span. Knots are chosen from a deviation tolerance, not from the input
   sampling, so a panel is a few dozen control points by a few dozen, not
   one per printed vertex.
5. **One valid closed solid per part.** Sewn, `BRepCheck` valid, one
   shell, positive volume, every edge shared by exactly two faces.
6. **Shared boundaries.** Adjacent panels are built from the same section
   curve at their joint, with the same knot vector, so their joint faces
   coincide edge for edge; the wing panel and its elevon share the hinge
   surface. An assembly then mates by construction rather than within a
   tolerance.
7. **Compatible parametrisation.** Every section of a part is fitted on
   one shared knot vector, so spanwise skinning never has to re-knot, and
   the result has no isoparametric twisting.
8. **Two levels of detail, two files.** The *outer mould line* solid is
   the aerodynamic object: for CFD, rendering, mould-making, and for
   anyone who wants to redesign the interior themselves. The *as-printed
   structural* solid is the skin as a one-bead shell, the rib webs as
   chains of ruled faces between the kinks of their triangle wave, and the
   tubes as cylinders: for FEA and for checking fit. Never merge them.
9. **Two frames.** The assembly is in the flight frame, millimetres, x
   aft, y starboard, z up, both halves built (a mirror transform is a
   negative determinant, which many CAD importers mishandle, so the left
   half is mirrored geometry, not a mirrored instance). Each part is
   also written alone in its print frame, root face on z = 0, next to its
   STL.
10. **Names, colours, layers.** Products named for the part
    (`trainer_v3_p1`, `LE spar`, `elevon0`), faces named for what they are
    (`skin`, `trailing edge`, `root joint`, `hinge chamfer`), colour by
    role (printed vase, printed solid, bought carbon, reserved payload).
    Payload boxes are exported as reference bodies on their own layer, so
    the builder sees where the pack goes.
11. **Provenance.** The design hash, the mission, the scoring verdict and
    the print settings go in the file's header and in a sidecar JSON, so
    a STEP file can always be traced to the `design.json` it came from.
12. **Size.** A panel's OML is kilobytes, an aircraft assembly well under
    5 MB. File size is gated, because a 30 MB part is the symptom of rule
    1 being broken.

## 2.3 Conformance gates

The value of the B-rep is the 3D questions it can answer, reported like
every other gate, with a number and a limit.

| gate | quantity | catches |
|---|---|---|
| loft fidelity | OML surface vs `Planform.at` on held-out `(eta, x)` | a fit that drifted |
| volume | OML volume vs the planform integral | a missing or doubled region |
| print conformance | printed outer vertices, mapped to the flight frame, vs the OML | 0.2 (straight panels) |
| tube inside shell | min clearance of each tube cylinder to the OML shell over its length | 0.1 (bent spars) |
| mass | as-printed solid volume x density vs `LayerStack.mass_g` | a shell that is not the shell weighed |
| validity | one valid closed solid per part, face and size budgets | the 9,635-face file |

## 2.4 Where it lives

`washout/cad/`, with the OpenCASCADE bindings (`cadquery-ocp`) as an
optional install, `pip install -e .[cad]`. The search stays
numpy/scipy, per CLAUDE.md; tests that need the kernel use
`pytest.importorskip("OCP")`; one CI job installs the extra. It runs at
export and in `check --conformance`, never inside the search loop -- the
same rule as tiers 1 and 2.

---

# 3. Run modes

The fidelity ladder is the architecture, and today only its bottom rung
has a command. With tier 0 back at its contract the ladder can be verbs.

| mode | what it does | status |
|---|---|---|
| `screen` | aero plus analytic gates, no panels built: feasibility probes, mission scoping, broad exploration | new -- the 595-draw probe as a command |
| `search` | as now, full build gates, seeded from `screen` | exists |
| `refine --top N` | tier-1 polars on the finalists' actual sections, re-rank, and report how often the order changed | ROADMAP.md item 7 |
| `export [--step]` | STLs, companions, BUILD/BOM/MANIFEST; `--step` adds `cad/` and the conformance gates | extends |
| `verify` | tier 2: 3D LBM on the OML, and FEA of the structural solid (gmsh + CalculiX, shelled out like the tunnel) against the closed-form GJ, EI and buckling | new -- the structural twin of the tunnel rung |
| `check [--conformance]` | every tracked design under today's gates; with the flag, the 3D gates too | extends |
| `fleet` | missions x seeds, resumable, `fleet status`; replaces `scripts/fleet/*.sh` | new |

Optional: `trade`, feasible first and then a Pareto front over two
merits. It is not a weighted sum, so it is inside the rules, and it is
the right tool for choosing a mission -- docile against fast for FPV --
rather than for choosing a design.

---

# 4. Structural topology

With nothing cut through the skin, the interior is one loop, a diamond
truss and tubes loose in the cavity. The truss is worth about 22% of GJ;
the tubes 33-35%, and 0.1 takes tube out of the outer third of the span.
demon1 fails reversal at 0.97x against 1.5x on the *optimistic* bound.
What it needs is torsion box, and within the one-loop rule there are
three ways to buy it:

- **Straight shear webs** at the spar stations: a detour that does not
  sweep, turning the section into a genuine multi-cell box.
  `gj_multicell_nmm2` can price this before a line of geometry is
  written, and should.
- **A sleeve that wraps the tube**: the loop circles the bore, which
  captures the tube positively (no glued seat) and shear-ties it into the
  box. It moves with the straight tube's chord fraction, so it is a lean
  and is gated like one.
- **Spanwise stringers on the compression skin**: zero-overhang detours
  that raise the skin's buckling load and relax the rib pitch it forces.

Then a structure mode: the OML fixed, the detour layout searched against
the closed forms (GJ, EI, buckling, mass) as tier 0, the structural STEP
through FEA as tier 1.

---

# 5. Phases

| phase | contents | status |
|---|---|---|
| A | `micro_fpv` tracked, suite green | **done** |
| B | tier 0 at contract (1.3) | **3.5x, not 30x** -- see below |
| C | true-slice panels, joint pivots and wedges (1.2) | **done** |
| D | straight spars (1.1) | **done** |
| E | OML STEP with the topology rules, loft-fidelity and volume gates | **done** |
| F | print-conformance and tube-inside-shell gates | **done**, one open question (0.4) |
| G | elevon split, spars as cylinders, assembly with names and colours | **done**; companions and payload not in the assembly yet |
| H | as-printed structural solid; `verify` with FEA | |
| I | structural topology (section 4) | |

## What landed, 2026-09-22, and what it found

**B -- tier 0.** Memoised `Planform.at`, a plan shared between the two
passes, interior depths over arrays, the overhang and bore searches
pruned against exact bounds. Zero mismatches against a golden snapshot of
sixteen designs at 1e-9, and every fast path pinned to its slow reference
in `tests/test_fast_paths.py`. A trainer evaluation went from 11.9 s to
about 3 s; the suite from 626 s to 175 s. That is not the 0.4 s CLAUDE.md
contracts -- what is left is `build_stack` and the elevons, lofted layer
by layer at 0.25 mm, which a search does not need: it is the next item.

**C -- panels are slices.** Every printed skin point, put back where it
flies, is within 0.045 mm of the loft. Panel masses outboard fell 3-7%:
the sections square to a tilted panel are thinner than the vertical ones
they used to be printed as. The joints table is on the build sheet, and
the minimal panel split is safe.

**D -- straight spars.** The fitter seats a tube at the root and aims it
at a seat further out, 900-odd candidate lines scanned together in about
0.03 s. Two things came out of making it honest:

- *A cut tube's end face is square to its axis*, so on a 46 degree tube
  it reaches 3.3 mm further out than the axis does, into section no
  station had checked. The bore gate found it (7.91 mm for an 8 mm tube);
  tubes now end short by exactly that much.
- *Twelve points round an ellipse are not the ellipse*: the polygon's
  chords sit 0.2 mm inside it. An independent check at 24 points found a
  tube poking through micro's skin; the polygon now circumscribes it.

The fleet, re-checked: **no aircraft is feasible**, and every miss is the
aircraft changing, not the model drifting. No straight tube reaches the
trainer's or demon1's outer joint. demon1's reversal margin falls from
0.92x to 0.70x now its tubes' stiffness stops where the tubes do.
micro_fpv's static margin falls from 0.132 to 0.091, outside its band,
because a tube swept 47 degrees weighs half its length aft of its seat.
micro's tracked design no longer trims at all. A re-search is next, and
it is searching against the aircraft that would actually be built.

**E-G -- the CAD export.** `run.py export --step` writes `cad/`. On
micro_fpv:

| | the first attempt | now |
|---|---|---|
| centre body faces | 9,635 | 4: skin, trailing edge, root plane, tip plane |
| centre body file | 30.9 MB | 0.30 MB |
| elevon | 9,600 faces | 7, each named |
| whole aircraft | -- | 3.7 MB: both halves, both tubes, 8 valid solids |
| surface vs print | -- | 0.02-0.045 mm on held-out layers |

Three things were wrong with the PRINT and had to be fixed first, because
a clean B-rep of a defective part is a clean record of the defect:

- *Every section had a 1 mm flat for a nose.* `thicken_for_nozzle`
  applied the trailing edge's floor along the whole chord, and at the
  nose, where the vertical thickness is zero by construction, it pushed
  the leading-edge vertex up half a millimetre and the next point down
  half. It now acts aft of 0.35c only. The wall gates had been passing the
  nose BECAUSE of the flat, pairing two sides of one bead's turn as two
  walls; they now skip neighbours by distance along the loop. One gate got
  harder and is recorded rather than tuned: micro's centre body overhangs
  51.4 degrees at its swept root nose, not 48.8.
- *The elevon's chamfer corner fell between two samples*, so the contour
  cut it with a chord and the turn hopped between vertices along the
  span -- twisted triangles in the STL, and 3.8 mm of wander in a surface
  skinned through it. It is a vertex now, at one index on every layer.
- *A design that did not trim lost everything it knew.* Rejected at trim,
  it came back without its parts, settings, tubes or linkage, so it could
  not be exported to see why, and the build sheet crashed on it. It keeps
  them now, the sheet says plainly that it does not trim, and the gates
  that are geometry -- spar reach, hinge line, propeller, four-bar -- run
  before trim so an untrimmed design's report is complete.

Still open: companions (horn, mount, fins) and payload boxes are not in
the assembly -- the parts exist as meshes in frames of their own; the
export takes 60-90 s, almost all of it fitting; and 0.4, which side of
the bead the surface is, is still a caliper away from settled.

---

# 6. What not to do

- **Do not make the STEP the source of truth.** The loft is. A B-rep
  derived from it and checked against it is a referee; a B-rep the search
  read back would be a second copy of the geometry, and two copies is
  how every serious bug in this project started.
- **Do not import OCC in the search.** It is an export-time and a
  check-time dependency, like the tunnel.
- **Do not export geometry that was not scored.** The STEP is built from
  the same `Evaluation` the STLs are, or not at all.
- **Do not add a "print flat" mode because the STEP makes one easy.** A
  STEP of the OML invites conventional printing with infill. That is a
  different aircraft with a different mass and a different structure, and
  nothing here would have scored it.
- **Do not loft the printed contour again.** Section 2.1.
