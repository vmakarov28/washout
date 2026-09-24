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
