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
