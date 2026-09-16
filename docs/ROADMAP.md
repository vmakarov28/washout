# What is left between here and a flying wing nobody touched

> Two roadmaps, two axes. **This one is whether the numbers are true.**
> [`ROADMAP-BUILD.md`](ROADMAP-BUILD.md) is whether the parts are
> buildable without a CAD step. They meet in two places: opening a
> battery bay destroys the closed torsion box that item 9's flutter
> gate depends on, and the linkage gate there is what finally proves
> the elevon deflection demon1's speed objective assumes.

washout already goes from a 35-number design vector to STLs a slicer will
print in spiral-vase mode, and judges the whole way with a lattice, a
structural sizing, a stability analysis and a stack of printability and
hardware gates. What follows is what still stands between that and the
claim in the title: **an aeroplane designed, checked and made printable
end to end, every number of it from physics or from measurement, with no
hand in the loop.**

Items are in value order. The ordering is not negotiable at the top:
items 1-4 mean the numbers currently reported are not the numbers that
would be built, and every result downstream of them inherits the error.

## The one discipline

Every serious bug in this project's history has been the same bug in
different clothes: **the search scored one aircraft and the export built
a different one.** Lattice resolution, z-sampling, drag model, spar
diameter, ribs, spar mass. Before adding anything, ask whether it makes
the scored aircraft and the built aircraft diverge. Closing a divergence
is worth more than any new feature.

Corollary, and it has paid every time it was applied: prefer making bad
designs *unrepresentable* over penalising them.

---

# A. The numbers are not the numbers

## 1. Charge the spars that were actually fitted - P0

`Mission._common(spar_g=...)` charges a **flat** "spar + joiners" item:
30 g trainer, 38 g demon1, 14 g micro. In the same run,
`structure.select` sizes a real tube and reports `spar_mass_g`, and
`spars.fit_all` reports how far each corridor reaches. Nothing connects
them. `grep -rn spar_mass_g` finds three hits, all inside `structure.py`.

Measured on the tracked trainer today: charged 30 g; the sized tube is an
8x6 carbon, 828 mm, **28.2 g** - and the mission declares *two* 8 mm
corridors, an LE spar reaching eta 1.00 and a TE spar reaching eta 0.76.
The second tube is never weighed at all.

Weigh what was fitted - for each `SparSpec`, that spar's own
`SparFit.reach_mm`, doubled (one tube runs tip to tip through the centre
body), priced with `structure.spar.mass_g` - and the trainer lands near
355 g. Its wing loading goes from 25.7 g/dm2, inside the 26 g/dm2 gate,
to about 27.5, outside it. **The fleet's headline design does not survive
its own spar**, and neither does any conclusion drawn from the gen5
searches.

*Acceptance:* a test in `tests/test_structure_in_search.py` style
asserting the budgeted spar mass equals the sum over fitted corridors of
the sized tube's mass, and that re-scoring
`results/fleet/gen5_trainer_v3_v101/design.json` reports about 355 g,
not 331.

## 2. Size the aft spar against its own load case - P1

`structure.select` sizes **one** tube against root bending. The trainer
and demon1 each declare two corridors, so item 1 will charge both at the
bending member's diameter. That is not what the aft tube is: it is a
torsion and anti-flutter member, and it almost certainly needs neither
8 mm nor full span.

This is the difference between the trainer passing and failing its wing
loading gate, so it is worth doing properly rather than guessing
downward.

**Ask before changing it.** The 8 mm tube is declared hardware. Whether a
6 mm or part-span aft tube is acceptable is a build decision.

## 3. A sweep that measures nothing must not write a file - P1

The Re 150 000 sweep failed at every single alpha: the WSL VM was in a
broken state (`getpwnam(master) failed`, `CreateProcessCommon:742`).
`tunnel_sweep.py` then wrote a header-only `polar.csv` and printed
`polar -> out\tunnel_re150\polar.csv`, which reads exactly like success.
That file is still on disk.

It has not yet poisoned a result - `MeasuredDrag.from_csv` would raise on
an empty array rather than return zero drag - but a silent zero here
would look like a miraculously low-drag aerofoil and win the
optimisation, which is the failure mode the force parser is explicitly
strict about. The writer should hold the same line: refuse to write a
polar with fewer rows than alphas requested, and exit non-zero.

*Related, unverified:* `trainer_mid_re60k.csv` has 9 rows starting at 0
degrees, while its own log reports 10 starting at -2. One measured point
did not reach the file. The sweep runs three alphas concurrently. Pin the
writer with a test before trusting any future sweep.

## 4. `run.py seed` cannot work - P2

`SEEDS` is `{}` by deliberate design ("an empty dict means start from
scratch, which is one less thing to keep true"), so `run.py seed` prints
"no seed recorded for this mission" and exits 1 for every mission - while
`README.md` advertises it as the first command to run. Either delete the
subcommand or redefine it as "evaluate the tracked winner for this
mission" from `results/fleet/`, which is a genuinely useful smoke test
and is what the new CI job does by hand.

---

# B. Put measured physics inside the loop

This section is the actual answer to "completely out of real physics and
simulation", and item 5 is the single highest-value piece of work in the
document.

## 5. A section-polar surrogate over shape, not just over Re - P0 of the physics

**The evidence.** On the tracked fleet, re-exported today, profile drag is
almost the entire drag budget:

| | CD0 | CDi | CD0 share |
|---|---|---|---|
| trainer | 0.0514 | 0.0101 | 84% |
| demon1 | 0.0354 | 0.0013 | **96%** |

Every one of those CD0 numbers comes from `MultiRePolar` interpolating
between exactly two measured curves. And **those two curves are of two
different aerofoils**: `trainer_mid_re60k.dat` is t/c 0.1266, camber
0.0171; `thin_reflex_re100k.dat` is t/c 0.0985, camber 0.0188 - 28%
apart in thickness. What the model calls a Reynolds-number trend is
partly a thickness difference, and *the search moves thickness*:
`t_over_c` ranges 0.075-0.165, `camber_scale` 0.30-1.70, `reflex_deg`
-1 to 9, `x_tmax` 0.20-0.44. The optimizer can reshape the section and
the drag model will not notice, because it was never measured on the
shape it is being asked about.

**The fix.** A surrogate for `cd(cl; t/c, camber, reflex, x_tmax, Re)`
and `cl_max(t/c, camber, reflex, x_tmax, Re)`, fitted to a designed batch
of 2D LBM runs, evaluated at tier-0 cost inside the search.

- **Design of experiments.** Latin hypercube over the four section knobs
  across exactly the bounds `SECTION_BOUNDS` declares, crossed with three
  to four Reynolds numbers spanning the fleet's strips. The trainer's tip
  chord is 79 mm against a 244 mm root, so the tip sits at **a third** of
  the root's Reynolds number, not half.
- **Cost, and how to afford it.** A 9-alpha sweep at 340 cells/chord is
  about 1.6 h wall with three runs sharing the card. Sixty DOE points that
  way is four days. Drop to about 220 cells/chord - cost goes as roughly
  cells cubed once the step count scales with cells - and it is about
  26 min a point, six at a time on 16 GB, so a 60-point DOE is an
  overnight job. **The resolution reduction must be justified, not
  assumed:** run the grid convergence against the 340-cell sweeps already
  in hand and report the cd error it costs. That check is itself a
  deliverable.
- **Sweep to stall.** The existing sweeps stop before cl_max - the Re 60k
  polar is still climbing at 16 degrees. The surrogate needs the peak,
  because item 6 depends on it.
- **It must refuse to extrapolate.** `MeasuredDrag` already clamps rather
  than extrapolating, and `tunnel.MeasuredDrag` raises on a strip it has
  no polar for. The surrogate must do the same: a design whose section
  falls outside the fitted box is *infeasible*, not guessed at. Report
  held-out error on a reserved fraction of the DOE and publish it in the
  module docstring.

**What it buys.** It retires the standing caveat that absolute L/D from
tier 0 is not quotable, within the fitted box and to the stated held-out
error. That caveat is currently attached to every number this program
produces.

## 6. Stall margin from the polar, at each strip's own Reynolds number - P1

`cl_max_section` is a hand-set constant per mission: 0.85 trainer, 1.00
demon1, 0.90 micro. Nothing measured it. It gates the peak section cl and
it underwrites `tip_stall_margin`, which the README correctly calls the
most important safety property in the file - a flying wing that drops a
tip on launch is how a beginner loses it.

With item 5's data, each strip can be charged the cl_max measured for
*its own* section at *its own* Reynolds number. The tip is thinner, more
washed out, and at a third of the root's Re, so its real margin is worse
than any single constant can say. This is the gate most likely to be
quietly optimistic today.

## 7. A `refine` command: make the ladder climb itself - P1

The fidelity ladder is documented as the architecture, and its top two
rungs are driven by hand. `aero/tunnel.py` already has `measure()`,
content-hash caching keyed on the CST coefficients and Re, and its own
strip-theory `MeasuredDrag` - and **nothing in the CLI calls any of it.**

Add `run.py refine --out out/gen6 --top 5`: read the search log, take the
top N feasible designs, extract each one's actual root / mid / tip
sections, dispatch tier-1 sweeps (cache hits are free, misses are a
deliberate spend), re-score, and print the re-ranking with the tier-0
order beside it. The interesting output is not the winner, it is **how
often tier 1 changes the order** - that number says whether the search's
rankings can be trusted at all, which is currently an assumption.

## 8. Finish a tier-2 run - P2

The 3D whole-aircraft LBM has never completed; the previous attempt hung
the WSL VM. It has one job: check whether the VLM's induced drag plus
strip-theory profile drag add up on a real blended wing body with a fat
centre body, where the strip assumption is weakest. Run it detached, with
a memory cap, and treat hanging that VM as a bot-killing failure rather
than an inconvenience (see constraints).

---

# C. Physics the score still cannot see

## 9. Torsional divergence, control reversal, flutter - P1 for demon1

Not modelled anywhere: `grep -rn "flutter|divergence|torsion|GJ"` returns
only the spiral-mode docstring. demon1 is scored at **44.3 m/s** on a
foamed-PLA single-wall shell with one 8 mm tube. Divergence and reversal
speeds are closed-form from the spar's GJ, the shell's contribution and
the offset between the elastic axis and the section's aerodynamic centre
- precisely the closed-form, published-reference style the project
already validates against.

The speed objective pushes directly toward the thing that has no gate.
That is the classic shape of an optimizer walking off a cliff.

## 10. The propeller constants are hand-set, and they set demon1's score - P1

`Propeller.ct0 = 0.105` and `Motor.load_factor = 0.82` are honest,
documented simplifications - "the single number to calibrate if you
bench-test yours". But demon1's *entire objective* is top speed, and top
speed is thrust-bounded, so **its headline number has a hand-chosen
constant inside it.** Under the project's own rule that everything comes
from simulation or published physics, this is the loudest remaining
exception.

Cheapest honest fix: APC's published measured performance data for the
diameters and pitches `prop_diam_in` / `prop_pitch_in` can reach,
interpolated - published measurement, the same standing as the NACA 4412
Cm the VLM is checked against. A bench test of the actual prop is better
still.

## 11. The hand launch - P3

Nothing models the launch: low speed, high alpha, full power, one hand,
asymmetric release. `min_thrust_weight` is a proxy for it. For a trainer
this is the flight phase most likely to end the aeroplane.

---

# D. From "an STL" to "an aircraft you can build"

The geometry reserves *volume* for the battery, receiver, ESC and servos
and gates that each fits where its mass actually sits. It models **no
openings whatsoever**: no hatch, no firewall, no servo pockets, no hinge
line. All of that is currently cut by hand, and it is the largest single
piece of manual work left in the pipeline.

## 12. Gate the battery hatch - P1, cheapest real win

Before modelling an opening, gate that one is *possible*: a hatch the
size of the pack's footprint must fit in the upper surface without
crossing a spar corridor or a rib. Everything needed is already computed
- `Bay.box_mm`, the solved `SparFit.x_frac`, the rib stations.

Modelling the hatch itself is a bigger job, and the rib trick shows the
shape of the answer: in vase mode a rib is not a separate loop, it is a
*detour of the skin loop*. An opening is the same idea - the loop detours
to trace the hatch rim - with the lid printed flat as a separate part.

## 13. Mark the elevon hinge line - P2

`elevon_chord` and `elevon_eta` are design variables that survive the
whole search and then vanish at export: nothing in the STL or the printed
report says where to cut. Scribing a shallow groove along the hinge line
is a contour-level operation, which is the one thing vase mode does
cheaply. A cut template as SVG is the fallback.

## 14. Emit the build, not just the parts - P2

This is tribal knowledge today, and every item of it is already in the
data structures at export time:

- CG station in mm from the root leading edge, and the ballast needed to
  reach it. This is the single most important build number and it is one
  line from `MassBudget`.
- Spar cut list: length, diameter, and how far along the span each runs.
- Elevon throws in mm at the trailing edge, from the `dcm_ddeg` the
  search already computes.
- `p0` is the fuselage - the panel split lands exactly on the body/wing
  junction.
- Print with **zero bottom layers** so the spar passes through the
  centreline and the electronics load before the halves are joined.
- Which gates were tight, so the builder knows what not to add weight to.

Write it as a generated `BUILD.md` beside the STLs, from the same
`Evaluation` the report is drawn from.

## 15. Emit the slicer profile - P3

The vase-mode settings are printed as prose at the end of every export.
Emitting an Orca/Bambu process profile with the exact extrusion width,
layer height, zero top layers, one bottom layer and spiralize on makes
the handoff mechanical instead of a paragraph to retype.

## 16. The motor mount - P3

The motor is a 36 g point mass at 0.97c and nothing else. There is no
firewall, no mount, no thrust line, and no check that the pusher prop
clears the trailing edge at the deflections item 9 would compute.

---

# E. Repo and process

Done in this pass: the `loft` to `washout` rename committed; the fleet
winners and the two measured polars tracked in `results/` and `data/`
with the sections they were measured on; `scripts/fleet/` holding the
generation runners; CI running the validation suite plus a full
end-to-end export on every push.

Still worth doing:

## 17. A determinism test - P2

The README's central claim is that `design.json` plus the print settings
reproduce the exact STLs. Nothing tests it. Export the tracked trainer
twice and assert the bytes match; export it again from a re-parsed
`design.json` and assert the same. It is cheap, and it guards the
property everything in `results/` depends on.

## 18. Fold `out/` into generations - P4

`out/` is 2.5 GB with roughly forty loose run directories beside
`gen2`...`gen5`. It is git-ignored, so this is housekeeping rather than
correctness - but the loose `v50`/`v60` directories no longer belong to
any generation anyone can name.

---

# F. Questions the search is asking you

When the optimizer pins a variable at its limit, it is reporting that the
design space is starved in that direction. Three of those are live.

## 19. The micro is sitting on the span floor - P3

`BOUNDS` has `span_m` at [0.35, 1.10] and the micro chose **0.3519**, half
a percent off the floor. The real minimum span is therefore smaller and
unknown, and finding it *is* the micro mission. Lower the bound and see
what actually binds. Strong prior: the 8 mm spar bore against root
thickness, which is the gate that has historically rejected small designs.

## 20. The two larger aircraft want all the fin they can have - P3

`fin_area_frac` is bounded at 0.06 of wing area. The trainer took 0.0505
(84% of the bound) and demon1 0.0566 (94%), both to reach Dutch-roll
zeta >= 0.08. The micro did not - it took 0.0226 and leaned on 49.5
degrees of winglet cant instead, which is the interesting half of the
result.

So the question is not "raise the bound". It is: what makes the cheap
lateral lever available to the micro unavailable to the other two? Sweep
distribution, dihedral staging, and the aft spar's mass as a yaw-inertia
term are all reachable in principle. Understand that before buying more
fin area, which is mass and drag at the longest arm on the aeroplane.

## 21. `span_m` is an inert dimension on two of three missions - P4

`build()` reads `p["span_m"]` only when `mission.span_free`, which is the
micro alone. On trainer and demon1 the optimizer is exploring a
coordinate that cannot change the aircraft - the gen5 trainer carries
0.8122 in a design that is 900 mm by mission. It costs population
diversity in a 35-dimensional search for nothing. Drop the dimension when
the mission fixes the span.

---

# Hard constraints

Carried forward. These are not preferences.

- **Local compute only.** Long searches run in the background locally and
  may keep running past the end of a session.
- **Never touch the WSL session running `deepTrader12.01.00.py` or
  `cache/watchdog_loop.sh`.** It is a live trading bot, and it is the
  child of a `cmd.exe` whose working directory is the repo folder, so an
  innocuous-looking terminal cleanup kills it. Verify before killing
  anything:
  `wsl -e bash -lc "ps -eo pid,ppid,args | grep -i deeptrader"`.
- **`loft` survives in the code as a verb** - `lofted()`, `_lofter`, "the
  faired loft". That is the real term for fairing a surface through
  stations. Never rename it.
- **Do not print anything from `out/gen1`-`out/gen4` search exports.**
  Those shells have no ribs. The only ribbed micro is
  `out/gen4/micro_v92_sized`.
- **Nothing is ever made up.** Every number leaves the program with a
  simulation, a measurement or a published reference behind it. Where
  that is not yet true - items 5 and 10 - it is written down as a debt,
  not rounded off.
- **Keep the tests.** A failing test is information; never delete one.
- **Look at `design.png`.** Geometry that looks incoherent is rejected
  even when every gate passes.

# What not to do

- **Do not model wing fences.** Neither the lattice nor the strip-theory
  drag model can see spanwise boundary-layer drift, so any benefit would
  be invented and the optimizer would pay real weight and drag for it.
- **Do not credit the AR630's AS3X gyro.** It genuinely helps in the air
  and it is genuinely unmodelled. Keep the airframe honest.
- **Do not add a weighted-sum score.** Fair beats unfair, feasible beats
  infeasible, then merit - strictly ordered, specifically so the
  optimizer cannot buy its way past a rule. It was tried and it failed:
  the optimizer parked 1 m/s below the cruise band because the penalty
  was cheaper than the L/D it bought.
- **Do not put tier 1 or tier 2 inside the optimizer loop.** That is what
  the surrogate in item 5 is for.

---

# Suggested order

1. Items 1, 3 and 4 together - one afternoon, and every number after them
   is trustworthy in a way nothing before them is.
2. Item 2, after asking about the aft tube.
3. Item 5, the surrogate: the DOE overnight, the fit and its held-out
   error the next day. Item 6 then falls out of the same data.
4. Item 7, `refine`, which is what tells you whether any of it moved the
   rankings.
5. Re-run the fleet as gen6, seeded from the gen5 winners, and expect the
   trainer to come back heavier and slower and *true*.
6. Items 12 and 14 - the ones that stop a human cutting hatches and
   guessing the CG.
7. Item 9 before anyone flies demon1 fast.
