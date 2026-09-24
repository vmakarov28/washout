# Validation: can anything here predict a section's cl_max?

Written 2026-09-23, BEFORE any of the runs below, so the criterion
cannot move after the results come in.

## Why

gen9 moved micro_fpv's objective to the slowest trimmed speed, and made
where the stall starts a gate. Both rest on `cl_max_section`, which is a
declared 0.85, the same for every section of every design. Using a
measured value per section needs a tool that predicts cl_max. The 2D LBM
tunnel says in its own notes that its stall is untrusted (windtunnel
`notes/NOTES.md`, Phase 6: "2D LES with no transition model"). So the
tool is validated first, against a measurement.

## The reference

Eppler 387 at a chord Reynolds number of 100 000, the standard low-Re
benchmark. Source: McGhee, Walker and Millard, *Experimental Results
for the Eppler 387 Airfoil at Low Reynolds Numbers in the Langley
Low-Turbulence Pressure Tunnel*, NASA TM 4062 (1988).

| source in TM 4062 | cl_max | at alpha |
|---|---|---|
| LTPT, Table B1 (run at 10 psi, M 0.04) and Fig. 17(b) | **1.20** (1.201 at 11.00 deg, 1.189 at 12.01) | about 11 deg |
| Stuttgart model tunnel, Table B3, same report | 1.12 | 13 deg |

The stall is abrupt: LTPT gives cl 0.842 at 14.00 deg. The two tunnels
differ by 7% in cl_max, which is roughly how well cl_max is known at
this Re.

Coordinates are in `e387_tm4062.dat`: TM 4062 Table I, design
coordinates with the model's thickened trailing edge (30 upper, 27
lower). They check to t/c 0.0907 at 0.31c and camber 0.038 at 0.40c.

## The criterion, fixed in advance

A tool passes if, at Re 100 000:

- **cl_max is within 10% of the LTPT value**, so between 1.08 and 1.32;
- **the angle of cl_max is within 2 deg of 11**, so between 9 and 13;
- **it actually stalls**: cl falls after the peak within the swept range
  (to 16 deg).

A tool that fails is not used for cl_max. A number from a tool that
fails would be worse than the declared constant, because it would look
measured.

## Candidates

1. The 2D LBM tunnel, fused solver, as `scripts/tunnel_sweep.py` runs
   it: 340 cells a chord, SGS on, 26 convective times an alpha.
2. NeuralFoil 0.3.2: a network trained on XFOIL, with e^N transition.
   It is installed on this machine and not yet used anywhere here.

## Results (2026-09-23 and 24)

**Neither tool passes. `cl_max_section` stays the declared 0.85.**

### NeuralFoil: fails on cl_max (+13%)

Data: `e387_neuralfoil.csv`. cl_max comes from a 0.25 deg sweep.

| Re | NeuralFoil cl_max | at | LTPT measured | error |
|---|---|---|---|---|
| 60 000 | 1.332 | 11.25 deg | 1.211 (11.0 deg) | +10.0% |
| **100 000** | **1.364** | **11.25 deg** | **1.207** (plateau 8.5-12 deg) | **+13.0%** |
| 460 000 | 1.409 | 13.0 deg | 1.284 (11.0 deg) | +9.8% |

- Angle of cl_max: passes.
- Stall: yes, cl falls to about 1.0 by 15 deg. Passes.
- cl_max at Re 100 000 is 13% high, outside the 10% band. **Fail.**
- Pre-stall it is good: cl 0.42 at 0 deg against 0.39 measured.

The bias is consistent, +10 to +13% at three Reynolds numbers. That is
the known XFOIL tendency at low Re. It is **not** corrected: scaling a
tool until it agrees with the reference is tuning a constant to match
a reference. A bias shown on ONE aerofoil says nothing about whether it
holds across shapes, which is the thing the search varies.

### 2D LBM: fails well before the stall

Data: `e387_lbm_partial.csv`. The sweep lost alphas 10-16 when the WSL
service dropped mid-run, so there is no cl_max.

| alpha | LBM cl | LTPT cl | error |
|---|---|---|---|
| 0 | 0.237 | 0.390 | -39% |
| 4 | 0.544 | 0.778 | -30% |
| 8 | 0.867 | 1.172 | -26% |
| 9 | 0.911 | 1.207 | -25% |

- The lift slope is about 4.3 per radian, against about 5.7 measured.
- To pass it would need cl_max of at least 1.08 by 13 deg. At 9 deg it
  is 0.91 and flattening.
- The missing points cannot rescue it. That is why the sweep was not
  rerun: rerunning would have risked the WSL VM a third time for no
  change in the verdict.
- This is what the tunnel's own notes predict. At 340 cells a chord and
  Re 100 000, the boundary layer is a few cells thick, and the solver
  has no transition model.

### What this means for the design

The slowest trimmed speed and the stall-onset gate still use a uniform
declared 0.85. They rank designs on how evenly the lift is spread and
what trim costs, and they say nothing measured about where the stall
starts on a real wing.

A root stall strip is not modelled. Nothing on this machine predicts
what a sharp leading edge does to cl_max at Re 100 000. The honest test
is physical: print two root sections, with and without the strip, and
tuft-test them.
