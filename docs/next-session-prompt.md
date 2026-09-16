

---

You are continuing work on **washout**, an automated blended-wing-body design
pipeline. It turns a 35-number design vector into a flying wing, judges it
against measured wind-tunnel data, and exports vase-mode-printable STLs.

**Location:** the folder is `C:\Users\aipla\Desktop\loft` but the project,
Python package, and everything else are named `washout`. The folder rename is
blocked — see item 7.

**Read first:** `README.md`, `docs/how-washout-designs-aircraft.html` (full
narrative of the pipeline, the models, and every bug found so far), and
`tests/test_structure_in_search.py` (the house style for pinning a fix).

## The one discipline

Every serious bug in this project's history has been the same bug wearing
different clothes: **the search scored one aircraft and the export built a
different one.** Lattice resolution, z-sampling, drag model, spar diameter,
ribs, and now spar mass. Before you add anything, ask whether it makes the
scored aircraft and the built aircraft diverge. Fixing a divergence is worth
more than any new feature.

Corollary: prefer making bad designs *unrepresentable* over penalising them.

## Hard constraints

- **Local compute only.** No subagents, no cloud runs, nothing that spends
  usage credits. Long searches run in the background locally and the user is
  happy for them to keep running past the end of a session.
- **Never touch the WSL session running `deepTrader12.01.00.py` or
  `cache/watchdog_loop.sh`.** It is the user's live trading bot. It is the
  child of a `cmd.exe` whose working directory is the repo folder, so an
  innocuous-looking terminal cleanup will kill it. Verify before killing
  anything: `wsl -e bash -lc "ps -eo pid,ppid,args | grep -i deeptrader"`.
- **`loft` survives in the code as a verb** — `lofted()`, `_lofter`, "the
  faired loft". That is the real term for fairing a surface through stations.
  Never rename it.
- **Do not print anything from `out/gen1`–`out/gen4` search exports.** Those
  shells have no ribs. The only ribbed micro is `out/gen4/micro_v92_sized`.
- The user rejects geometry that looks incoherent even when every gate passes.
  Look at the rendered `design.png`, don't just read the numbers.
- Never make up information, everything must always come from simulation and pure physics results (Check to make sure this rule has not be broken).
- Keep the tests, never delete a non passing test case.

## Current fleet, as actually built

| | mass | headline | ζ | misses |
|---|---|---|---|---|
| `out/gen5/trainer_v3_v101` | 355 g | L/D 7.02 | +0.101 | wing loading 27.5 > 26 |
| `out/gen5/demon1_v102` | 334 g | 44.2 m/s | +0.114 | trims at CL 0.136 < 0.15 |
| `out/gen4/micro_v92_sized` | 137 g | 352 mm span | +0.070 | ζ < 0.08 |

These masses are the corrected ones. The searches logged 331 / 317 / 140 g
because of item 1. The user chose to print the parts as-is rather than
re-search, so **the STLs on disk are fine to print but the logged numbers are
not the numbers.**

## Work items, in value order

### 1. Fix the spar mass accounting — P0, invalidates everything else

`Mission._common(spar_g=...)` in `washout/search/design.py` sets a *fixed*
"spar + joiners" payload item (trainer 30 g, demon1 38 g, micro 14 g).
`choose_structure` sizes a real tube and reports `structure.spar_mass_g`, but
nothing feeds that back into the budget.

Charge the real tubes: for each spec in `mission.spars`, take that spar's own
`SparFit.reach_mm` from `spars.fit_all`, double it (one tube runs tip to tip
through the centre body), and price it with `structure.spar.mass_g`. Real
totals today: trainer 54.0 g, demon1 54.5 g, micro 11.2 g.

Acceptance: a test in `tests/test_structure_in_search.py` style asserting the
budgeted spar mass equals the sized tubes' mass, and that re-scoring
`out/gen5/trainer_v3_v101/design.json` gives 355 g rather than 331 g.

### 2. Size the aft spar independently — P1, recovers what item 1 costs

`structure.select` sizes **one** spar against the bending load, but the trainer
and demon1 missions each declare **two** 8 mm corridors, so both get charged a
full 8 mm tube. The aft spar is a torsion and anti-flutter member, not the
primary bending member — it almost certainly does not need to be 8 mm or
full span.

This is the difference between the trainer passing and failing its wing
loading gate, so it is worth doing properly. **Ask the user before changing
it:** the 8 mm spar is declared fixed hardware, and whether they want a 6 mm
or part-span aft tube is their build decision, not yours to assume.

### 3. Run gen6 — P2

Only after 1 and 2. All three missions, seeded from the gen5 winners with
`--seed-design`, local, in parallel, one core each, ~6–10 h per search. Follow
`scripts/run_fleet_gen5.sh`. Put results in `out/gen6`, continuing the v-number
sequence from v102.

### 4. Close the gap between "an STL" and "an aircraft you can build" — P3

The geometry reserves *volume* for the battery, receiver, ESC and servos and
checks it fits, but models **no openings at all**: no battery hatch, no motor
firewall or mount, no servo pockets, no elevon hinge line. The user is
currently cutting all of that by hand.

Cheapest real win: gate that a hatch the size of the battery's footprint fits
in the upper surface without crossing a spar corridor or a rib. Modelling the
hatch itself is a bigger job — vase mode allows one closed contour per layer,
so an opening is not free.

Related, already known: `p0` is the fuselage (the panel split sits exactly at
the body/wing junction), the exported mesh is capped at both ends, and it must
be printed with zero bottom layers so the spar can pass through the centreline
and the electronics can be loaded before the halves are joined. None of that is
in the exporter's output — it is tribal knowledge. Consider emitting it.

### 5. Find the micro's true minimum span — P4

`BOUNDS` has `span_m` at `[0.35, 1.10]` and the micro chose 0.3519 — it is
sitting on the floor, so the real minimum is smaller and unknown. Lower the
bound and find what actually binds. Strong prior: the 8 mm spar bore against
root thickness, which is the gate that has historically rejected small designs.

### 6. Ask why every aircraft needs maximum fins — P4

All three winners sit near the top of the tip-fin area bound to reach ζ ≥ 0.08.
When the optimizer pins a variable at its limit on every mission, the design
space is starved in that direction. Worth understanding before adding more fin
area: is there a cheaper lateral lever (sweep, dihedral distribution, aft spar
position as mass) the search cannot currently reach?

### 7. Housekeeping — P5

`scripts/tunnel3d.py` and `scripts/tunnel_sweep.py` now point at
`/mnt/c/Users/aipla/Desktop/washout`, which does not exist — the folder is
still `loft` because the process holding it runs the trading bot (see
constraints). Either rename the folder at a moment the user is willing to
restart deepTrader, or revert those two paths. Ask; do not decide alone.

Also unfinished: the 3D LBM tunnel run, the top rung of the fidelity ladder,
has never completed — the previous attempt hung the WSL VM. If you attempt it,
run it detached with a resource cap, and treat hanging that VM as a
bot-killing failure, not an inconvenience.

## What not to do

- Do not model wing fences. Neither the vortex lattice nor the strip-theory
  drag model can see spanwise boundary-layer drift, so any benefit would be
  invented, and the optimizer would pay real weight and drag for it.
- Do not give the design credit for the AR630's AS3X gyro. It genuinely helps
  in the air and it is genuinely unmodelled; keep the airframe honest.
- Do not add a weighted-sum score. Feasibility is strictly ordered — fair beats
  unfair, feasible beats infeasible, then merit — specifically so the optimizer
  cannot buy its way past a rule.
