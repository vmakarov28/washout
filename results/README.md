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
