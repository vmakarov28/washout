# Measured section polars

These are the only *measured* aerodynamic data in the repository. Every
other number in a washout design is computed: the lattice is inviscid,
and the tier-0 drag model is a flat-plate estimate. These CSVs are what
`run.py search --polar ...` uses to charge a design for flying at a high
angle of attack.

Each file is an alpha sweep produced by the companion `windtunnel`
repository (2D D2Q9 LBM, SGS on, force measured on the obstacle), driven
by `scripts/tunnel_sweep.py`. The section each one was measured on is in
`sections/`, as the exact Selig `.dat` that was handed to the solver, so
a polar is never separated from the shape it describes.

| file | section | t/c | camber | Re | alphas |
|---|---|---|---|---|---|
| `trainer_mid_re60k.csv` | `sections/trainer_mid_re60k.dat` | 0.1266 | 0.0171 | 60 000 | 0 … 16° |
| `thin_reflex_re100k.csv` | `sections/thin_reflex_re100k.dat` | 0.0985 | 0.0188 | 100 000 | −2 … 14° |

## Read this before using both files together

**The two polars are of two different aerofoils.** They differ by 28% in
thickness. `performance.MultiRePolar` interpolates between them in
log(Re) and calls the result a Reynolds-number trend — so part of what it
attributes to Reynolds number is really the thickness difference. The
gen5 fleet was searched with exactly this pair. It is a ranking tool, not
a measurement, and the level it produces should not be quoted.

The fix is a proper section-polar surrogate over (t/c, camber, reflex,
Re); see `docs/ROADMAP.md`, item 3.

Two smaller things the files themselves record:

- `trainer_mid_re60k.csv` starts at 0°, but its run log reports a −2°
  point (cl −0.1477, cd 0.0339) that never reached the file. The sweep
  ran three alphas concurrently. Treat the sweep writer as unverified
  until it is pinned by a test.
- A third sweep at Re 150 000 was attempted and every alpha failed: the
  WSL VM was in a broken state (`getpwnam` / `CreateProcessCommon`
  errors). `tunnel_sweep.py` wrote a header-only `polar.csv` anyway and
  printed a success line. That empty file is deliberately **not** kept
  here. The writer needs a guard.

## Reproducing one

```bash
python scripts/tunnel_sweep.py --design results/fleet/gen5_trainer_v3_v101/design.json \
    --alphas -2,0,2,4,6,8,10,12,14,16 --re 60000 --chord-cells 340 \
    --tag tr --out out/tunnel
```

It needs the `windtunnel` repo on this machine and a CUDA GPU under WSL.
Without one, stay on tier 0 and do not quote absolute L/D.
