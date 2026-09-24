# The fleet, as searched

Three aircraft, one motor (2205 2300 kv), one receiver (Spektrum AR630),
one printer (X1C, Bambu PLA Aero). Each folder holds the design vector,
the design sheet the search drew, and where one exists the log of the
search that found it.

**STLs are not kept here.** They are 2.1 GB and they are a pure function
of `design.json` plus the print settings, so they are rebuilt, not
stored:

```bash
python run.py export --mission trainer_v3 --design results/fleet/gen5_trainer_v3_v101/design.json --polar data/polars/trainer_mid_re60k.csv@60000,data/polars/thin_reflex_re100k.csv@100000 --out out/rebuild_trainer
```

Pass the same `--polar` the search used. Without it the export falls back
to the flat tier-0 drag model and reports a different, flattering L/D for
the same aeroplane — the mistake commit `f3b61a9` exists to prevent.

## gen10, 2026-09-24: the nose-down stall, and most of the speed back

Eight micro_fpv searches ran (`scripts/fleet/gen10.py`), all seeded from
gen9's v144, the one design found inside the nose-down-stall region, for
50 generations. **All eight finished feasible**, between 7.61 and 7.83
m/s. The first launch died 2.2 h in when WSL had to be restarted, and
its best, 7.88 m/s, died with it. Searches now write
`best_so_far.json` every time they improve.

The top three were within 0.003 m/s of each other. The search's own top
score, v155, also has the widest stall margin, so it is the pick:

| | gen8 curl | gen9 v144 | **gen10 v155** |
|---|---|---|---|
| slowest trimmed | 7.38 m/s | 8.04 | **7.61** |
| stall starts | 0.18 MAC **behind** the CG (pitches up) | 1.7 mm ahead | **6.6 mm ahead** (0.044 MAC) |
| hands-off | 12.3 m/s (1.67×) | 11.7 (1.45×) | 12.3 (1.61×) |
| area / mass | 757 cm² / 165 g | 628 / 166 | 693 / 172 |
| SM / ζ Dutch roll | 0.124 / 0.086 | 0.139 / 0.083 | 0.126 / 0.089 |
| prop to TE | passes | 13.2 mm | 12.4 mm |

gen10 gets back most of the 0.7 m/s the nose-down stall cost in gen9: it
is 0.23 m/s behind gen8, and stalls the right way. The export passes
every gate, build sheet and CAD, with no exceptions. The stall margin is
6.6 mm of CG: better than gen9's, and still a reason to balance at or
ahead of the design CG.

The minimum speed and the stall location use a declared section cl_max
of 0.85. Nothing on this machine passed validation as a cl_max predictor
(`data/validation/README.md`): NeuralFoil is 13% high on E387 at Re
100k, and the 2D LBM is 25% low before the stall. So these are rankings
of how evenly the lift is spread and what trim costs, not a measured
stall.

### In the tunnel

The runs are at Re 15,000, against about 122,000 in flight, on a
44.5 Mcell grid.

| run | α | tunnel CL | VLM CL | inner half, tunnel / VLM | outer 20%, tunnel / VLM |
|---|---|---|---|---|---|
| gen10 micro_fpv | 4.01° (trim) | 0.080 | 0.269 | 95% / 60% | −0.2% / 12% |
| gen10 micro_fpv | 9° | 0.222 | 0.537 | 87% / 58% | 4.0% / 13.5% |

This is the same low-Re picture as every earlier run, with the centre
body carrying the lift. At 9° the tunnel CL of 0.222 matches gen8's
curl (0.223) and is well above gen9 (0.147).

```bash
python ../windtunnel/scripts/serve_viewer.py out/tunnel3d/gen10_micro_fpv_a9/tunnel/viewer
```

## gen9, 2026-09-23: micro_fpv for a beginner

Before this generation, the search changed in two ways (commit `18b9f37`):

- **The objective is the slowest TRIMMED speed** (`performance.slow_flight`).
  It is limited by whichever comes first: the first section reaching
  cl_max, or the up-elevon running out. The old stall speed put every
  section at the declared cl_max at once and spent nothing on trim.
  Under the new objective, gen8's curl is 7.38 m/s, not 6.40.
- **Two beginner gates.**
  - Hands-off speed must be at least 1.3× the slowest trimmed speed.
  - The first section to stall must sit ahead of the CG, so the nose
    drops at the stall.

  Every micro_fpv design through gen8 started its stall at mid-span,
  0.18–0.31 MAC behind the CG. That pitches the nose up into the stall.

Eight searches ran (`scripts/fleet/gen9.py`, 4.25 h): five of the curl
family and three of the winglet family, each seeded from its gen8 winner.
**One found feasible designs**: v144, with 11 feasible out of 8,610
evaluations. The other seven stopped early on differential evolution's
tolerance with nothing feasible. The winglet family came closest at
0.08 MAC behind the CG.

| | mass | area | slowest trimmed | hands-off | stall starts | SM | ζ Dutch roll | fins |
|---|---|---|---|---|---|---|---|---|
| `gen9_micro_fpv_v144` | 166 g | 628 cm² | **8.04 m/s** (CL 0.654) | 11.7 m/s, 1.45× | eta 0.38, **0.013 MAC ahead of the CG** | +0.139 | +0.083 | 2 × 11 cm², 41 mm |
| gen8 curl, for comparison | 165 g | 757 cm² | 7.38 m/s | 12.3 m/s | eta 0.48, 0.18 MAC behind | +0.124 | +0.086 | 2 × 5 cm² |

The nose-down stall costs 0.7 m/s. The search paid for it by moving the
CG forward (139 mm against 172) and giving up wing area. L/D is 6.30 at
tier 0, a ranking only.

**The stall margin is thin: 0.013 MAC, about 2 mm.** Balance the
aircraft a few millimetres nose-heavy of the design CG, never aft of it.
A battery a little aft of its bay position is enough to put the stall
back behind the CG.

The export passes every gate except one, on the build sheet: propeller
to trailing edge is 9.83 mm against 10. The horn socket passes at 2.58×,
and the CAD loft fidelity passes on every part (worst 0.04 mm). Both had
failed on gen8. Section cl_max is still the declared 0.85, the same
everywhere. A measured section would move every number above, so the
tier-1 polar at flight Re is the next thing worth doing.

### In the tunnel

The runs are at Re 15,000, against about 105,000 in flight. The grid is
48.5 M cells, 68 cells per MAC.

| run | α | tunnel CL | VLM CL | inner half, tunnel / VLM | outer 20%, tunnel / VLM |
|---|---|---|---|---|---|
| gen9 micro_fpv | 6.71° (trim) | 0.112 | 0.317 | 89% / 63% | −0.7% / 9.9% |
| gen9 micro_fpv | 9° | 0.147 | 0.438 | 90% / 61% | 1.8% / 11% |

The result matches every earlier run. At this Re the centre body carries
the lift and the outer wing carries almost none, and the tunnel CL is a
third of the lattice's. The circulation split recovers 85–86% of the
force total. The low-Re result neither confirms nor contradicts the stall
prediction, which is made at flight Re with a declared cl_max.

The first attempt at this run died at the first time-average, out of GPU
memory under the 11 GB cap. `scripts/tunnel3d.py` now asks for
expandable segments, and these runs used a 12.5 GB cap.

Viewer:

```bash
python ../windtunnel/scripts/serve_viewer.py out/tunnel3d/gen9_micro_fpv_a9/tunnel/viewer
```

## gen8, 2026-09-23: micro_fpv, curled tips against real winglets

**Superseded by gen9.** Neither design passes today's nose-down-stall gate.
The numbers below are as they were searched, under the flat stall speed.

Two families on the same gates and objective (stall speed), four seeds
each (`scripts/fleet/gen8.py`, 5.0 h). `micro_fpv_winglet` is micro_fpv
with the tip rise capped at 12% of the semi-span and the fins free to
grow into winglets. The fins are now panels in the vortex lattice, not
a plate-model add-on (`ROADMAP.md` item 20).

| | mission | mass | area | stall | L/D | trim | SM | fins | tip rise |
|---|---|---|---|---|---|---|---|---|---|
| `gen8_micro_fpv_v132` | micro_fpv | 165 g | 757 cm² | **6.40 m/s** | 5.33 | +4.77° | +0.124 | 2 × 5 cm² | 27% |
| `gen8_micro_fpv_winglet_v130` | micro_fpv_winglet | 165 g | 674 cm² | 6.80 m/s | 5.49 | +7.78° | +0.170 | 2 × 17 cm², 50 mm | 12% |

The curl wins the mission's objective. For the same mass it carries 12%
more wing area. The winglet design has slightly better tier-0 L/D, but
it trims at 7.8° against an 8° limit. L/D is tier 0, so the ranking
holds and the levels are not quotable. One winglet seed (v132) stopped
early on differential evolution's tolerance with nothing feasible.

The joints now have wedge inserts (`printing/inserts.py`): a printed
part that fills the gap the dihedral leaves between panels. On the
winglet winner, the tube cuts through the wedge, so that joint is filled
with glue (0.6 g of microballoon epoxy, declared) instead.

**Open before printing either one:**
- The **horn socket gate fails on both** (0.45× and 0.57× of the couple
  the servo can apply). It is a build-sheet gate, not a search gate.
- micro_fpv's centre body misses **loft fidelity** by 0.4 µm at the nose
  (0.0504 against 0.05 mm, `ROADMAP-CAD.md`).

### In the tunnel (tier 2)

Every run is the whole aircraft in the 3D LBM tunnel
(`scripts/tunnel3d.py`), fins included, at 68 cells per MAC. The Re is
**15k, about ten times below flight** (126k–145k), so the coefficients
are not flight values. What the runs compare is where the lift is
carried.

| run | α | tunnel CL | VLM CL | inner half, tunnel / VLM | outer 20%, tunnel / VLM |
|---|---|---|---|---|---|
| gen7 micro_fpv | 3.16° (trim) | 0.077 | 0.203 | 103% / 67% | −1% / 7% |
| gen7 micro_fpv | 9° | 0.208 | 0.493 | 80% / 61% | 2.4% / 11% |
| gen8 micro_fpv (curl) | 4.77° (trim) | 0.113 | 0.234 | 88% / 67% | 2.8% / 7.9% |
| gen8 micro_fpv (curl) | 9° | 0.223 | 0.443 | 76% / 62% | 6.0% / 10.9% |
| gen8 winglet | 7.78° (trim) | 0.103 | 0.264 | 102% / 71% | 0.1% / 6.0% |
| gen8 winglet | 9° | 0.130 | 0.326 | 94% / 68% | 0.9% / 7.8% |

Tunnel CL is momentum exchange on the body. The spanwise split comes
from Kutta–Joukowski circulation on the time-averaged flow, and it
recovers 89–93% of the force total.

At this Re, both families carry their lift on the centre body, and the
outer wing carries almost none. The winglet aircraft does so more
strongly than the curl, and at a higher α: its outer 20% carries under
1% of the lift. So these runs do not show the winglets doing the job the
lattice credits them with. At Re 15k, though, the outer panels are
thin, laminar sections that separate. That is a Reynolds-number effect
this tunnel cannot remove, and it is the same reason every tunnel CL
here is about half the lattice's. Do not read it as a flight prediction.
It is the strongest argument yet for the tier-1 section polar at flight
Re (`ROADMAP.md`).

Each run folder in `out/tunnel3d/<run>/` holds `smoke.mp4` (three-colour
dye, three cameras), `span_loading.png`, `report.json` and
`tunnel/viewer/`. The viewer is an interactive 3D view: ray-marched
smoke, a speed or vortex volume, streamlines, and the body coloured by
surface Cp. To open it:

```bash
python ../windtunnel/scripts/serve_viewer.py out/tunnel3d/gen8_micro_fpv_winglet_a9/tunnel/viewer
```

then go to http://localhost:8765.

## gen7, 2026-09-22: the fleet re-searched against the aircraft as built

After the straight spars and the sliced panels (below), twelve searches
(`scripts/fleet/gen7.py`, four missions by three seeds, each seeded from
its tracked design, 40 generations, pooled workers) ran for 5.0 h.
Winners were exported with `--step`; every CAD gate passes on all four.

| | mission | mass | span | headline | trim | SM | ζ Dutch roll | verdict |
|---|---|---|---|---|---|---|---|---|
| `gen7_trainer_v3_v121` | trainer_v3 | 374 g | 900 mm | L/D 7.21 at 10.0 m/s | +7.35°, CL 0.416 | +0.165 | +0.094 | feasible, **indexed** |
| `gen7_micro_v120` | micro | 161 g | 588 mm | L/D 6.48 at 11.5 m/s | +5.47°, CL 0.304 | +0.119 | +0.086 | feasible, **indexed** |
| `gen7_micro_fpv_v120` | micro_fpv | 163 g | 480 mm | cruise 13.5 m/s, stall 6.49 m/s | +3.09°, CL 0.196 | +0.135 | +0.100 | feasible, **indexed** |
| `gen7_demon1_v120` | demon1 | 368 g | 800 mm | 16.6 m/s cruise | +3.57°, CL 0.162 | +0.100 | +0.085 | **infeasible**, kept, not indexed |

L/D is tier 0: the rankings hold, the levels are not quotable.

**demon1 found no feasible design.** All three seeds end on the same
gate: the servo shaft at eta 0.28 lands 10 mm inboard of the elevon it
drives (elevon from eta 0.30). `gen5_demon1_v102` stays in the index as
the last design that met the mission under the gates of its day; it does
not meet today's. The gen7 folder is kept as the next search's seed.

**The trainer's build sheet fails a check the search does not make**:
prop to trailing edge 6.76 mm against 10 (disc plane at x = 261 mm, radius
64 mm). Prop clearance is `ROADMAP-BUILD.md` item 12 and is not yet a
search gate, so the search calls the design feasible and the build
sheet does not. Resolve that before printing it.

**The trainer's first print joint was wrong as searched.** Its pivot
rule fell back to the chord line when the joint's turn changed sign, and
that put a 10° turn about the chord line back: p0 and p1 shared 4.3 mm
of material above it. That joint now sits at the height where it does not
turn, so p1 prints level and the faces mate flat. The outer joint takes the whole
21.9° and a 9.4 mm glue wedge. Re-checked under the fixed frames, the
design still passes every search gate (L/D 7.20). The search that found
it scored the old frames.

The export itself found a CAD defect: demon1's elevons came out as
invalid solids. The elevon's nose floor meets its chamfer at a 45° corner
that falls between grid points; the cubic fitted across it folded back
over the nose flat and the end cap crossed itself. Such corners now get
a vertex, straight pieces are fitted as ruled faces, and a
split is kept only where it holds the held-out gate
(`tests/test_validation.py`, `tests/test_cad.py`).

## Re-exported 2026-09-15, under today's gates

| | mission | mass | span | headline | trim | ζ Dutch roll | gates |
|---|---|---|---|---|---|---|---|
| `gen5_trainer_v3_v101` | trainer_v3 | 331 g | 900 mm | L/D 7.32 at 9.6 m/s | +7.80°, CL 0.451 | +0.081 | all pass |
| `gen5_demon1_v102` | demon1 | 317 g | 800 mm | 44.3 m/s top speed | +2.43°, CL 0.157 | +0.106 | all pass |
| `gen4_micro_v92_sized` | micro | 140 g | 352 mm | smallest flyable | +5.37°, CL 0.255 | +0.086 | all pass |

Two of those rows moved since they were logged, and both movements are
worth more than the numbers themselves.

**The trainer reproduces exactly** — 331 g, L/D 7.32, SM +0.161, every
gate OK, identical to its `design.json`.

**The micro does not.** Its `design.json` records 132 g; re-exporting the
same design vector today gives 140 g and ζ +0.086 where the search logged
+0.070. That design came out of gen4, before `evaluate()` sized the spar
and the buckling-driven ribs inside the search loop, and its ribbed parts
were produced by a later hand export. The 8 g is the ribs, arriving after
the verdict. It now passes the damping gate it was recorded as missing.

## A fourth: `fpv_micro_fpv_s4`, tracked 2026-09-22

The `micro_fpv` mission -- sub-250 g, 480 mm, FPV, objective "docile"
(minimum stall speed) -- ran six searches seeded from `gen4_micro_v92_sized`
after every opening was removed from the skin. `s4` and `s2` were the
only designs in `out/` that passed every gate; `s4` is tracked. 163 g,
wing loading 28.2 g/dm² against a 34 limit, cruise 13.5 m/s, SM +0.132,
ζ Dutch roll +0.123.

It is tracked so `run.py check` and the fleet tests cover the mission the
branch added, and it is the aircraft `docs/ROADMAP-CAD.md` measured its
two defects on: its build sheet told the builder to run a straight 6 mm
tube tip to tip through a 48.7° swept wing, and its outer panel printed
straight where the loft rises 66 mm. Both change its verdict once fixed.

## 2026-09-22: straight tubes and sliced panels

Two defects were fixed on this date (`docs/ROADMAP-CAD.md` section 0):
the spars were modelled as tubes that bend with the wing, and the
printed panels as straight where the loft curves. Re-checked with both
fixed, **none of the four tracked designs is feasible**:

| | now misses, among others |
|---|---|
| trainer | neither straight tube reaches the joint at eta 0.74; SM +0.128; wing loading 26.9; 350 g |
| demon1 | neither tube reaches eta 0.63; reversal 0.70x (was 0.92x) -- the tubes' stiffness now stops where they do |
| micro | does not trim: its 46 deg tube weighs 44 mm aft of where it was charged, and SM fell to +0.012 |
| micro_fpv | SM +0.091 against 0.12, for the same reason; trims 10.1 deg |

Every one of those is the aircraft changing. A re-search is what closes
them, and it searches the aircraft that would be built.

## All three are now infeasible, and that is the point

The table above is the state as **searched**. Since then two pieces of
the roadmap have landed and both took feasibility away:

`ROADMAP.md` item 1 — the spars are weighed as fitted rather than charged
a flat allowance. `ROADMAP-BUILD.md` Phase 1 — the inside of the wing has
a coordinate system, the spar has a vertical seat, and bays, tubes and
print joints are reserved against each other.

Re-scoring today with `python run.py check --mission <name>`:

| | now misses |
|---|---|
| trainer | cruise 11.5 m/s (band 7–11); wing loading 27.5 (limit 26) |
| demon1 | electronics bay 11.9 mm deep for 16; pack ∩ TE spar 2.9 mm; pack ∩ electronics 1.7 mm |
| micro | neither bay closes in p0; SM +0.051; trims 11.2°; ζ −0.048 **divergent** |

None of that is a regression. Every one of those gates is measuring
something that was always true and was never checked — an 8 mm tube ran
through the battery of all three aircraft, and the trainer was carrying
24 g of spar it was never charged for. The designs did not get worse; the
program got honest. A re-search is what closes it, and it has to come
after the gates rather than before.

## How the mass went wrong in the first place

`Mission._common(spar_g=...)` charges the trainer a **flat 30 g** for
"spar + joiners". In the same report, `structure.select` sizes a real
8×6 carbon tube, 828 mm long, at 28.2 g — and the mission declares *two*
8 mm corridors, an LE spar reaching η 1.00 and a TE spar reaching η 0.76.
The second tube is never weighed at all. Nothing feeds either number back
into the mass budget.

Weighing the tubes actually fitted puts the trainer near 355 g, and its
wing loading goes from 25.7 g/dm² — inside the 26 g/dm² gate — to about
27.5, outside it. **The headline design does not survive its own spar.**

This is the project's cardinal failure mode wearing new clothes: the
search scored one aircraft and the export builds a different one. It is
item 1 in [`docs/ROADMAP.md`](../docs/ROADMAP.md), and it is why this
page reports what was *scored* and says plainly that it is not what
would be *built*.

The STLs already on disk are fine to print. These numbers are not the
numbers.
