# loft

Automated blended-wing-body design, from a design vector to a
vase-mode-printable STL, judged the whole way by physics rather than by
taste.

Two halves that only make sense together:

- a **search** that shapes a tailless blended wing body against a vortex
  lattice on its own mean camber surface, with a fidelity ladder up to a
  GPU lattice-Boltzmann tunnel, and
- an **exporter** that turns the winner into parts a slicer will print in
  spiral-vase mode — one bead per layer, no infill, no supports —
  with every reason that could fail checked as a gate, not hoped for.

The reference point is
[Beachless/Vase-Wing](https://github.com/Beachless/Vase-Wing), an OpenSCAD
vase-mode wing generator. That takes an aerofoil you chose and gives you
a printable wing. This chooses the aerofoil, the planform, the twist, the
reflex and the CG for you, because it can measure whether they fly.

## The idea in one line

Stand the panel on its **root** so span becomes print Z, and every layer
of a lofted wing is exactly one closed aerofoil contour — which is
precisely what spiralize mode needs.

Three consequences follow, and they are enforced as gates:

1. **Span becomes print height.** Panels taller than the Z envelope are
   split spanwise automatically and rejoined on a spar.
2. **Bed footprint is chord × thickness**, so root chord is capped by the
   bed *diagonal* — the exporter finds the best rotation.
3. **Sweep and taper become overhang**, measured perpendicular to the
   previous loop (not by point index — points slide tangentially as the
   chord shrinks, and that is not an overhang).

And one gift: the spar runs spanwise, which is now the print Z axis, so
**the spar channel needs no hole through the single wall**. The shell's
own cavity *is* the channel; a carbon tube slides in after the print.
`spar_fit` checks the tube clears every layer.

## Run it

```bash
python run.py seed                                  # evaluate the seed design
python run.py search --iters 90 --out out/run1      # overnight -> design.json + STLs
python run.py export --design out/run1/design.json  # rebuild STLs from a design
python -m pytest -q                                 # 16 validation gates
```

Printer envelope, nozzle, layer height, filament density and spar
diameter are all flags — see `python run.py --help`. Defaults are a
256³ bed, 0.4 mm nozzle at 0.45 mm width, 0.25 mm layers, LW-PLA at
0.60 g/cc.

## Real structure, not just a shell

Spiralize prints ONE contour per layer, so at first glance the shell can
have no internal structure: a rib joined to the skin is a T-junction and
the curve stops being simple. The way round it — the one
[Vase Mode Wing](https://vmw.deneb-systems.com/#studio) uses — is that a
rib is not a separate loop, it is a **detour of the skin loop**:

```
upper  ------.        .------------
             |        |
             |  rib   |     <- two legs, one bead apart
             |        |
lower  ------'--------'------------  (gap = one bead: welds, never touches)
```

Topologically still one simple closed curve; physically a welded web.
Ribs sweep chordwise as Z rises, so the stack becomes a diamond truss.

Two invariants hold the rest of the pipeline together. Point count per
layer is **constant** — the skinner joins layer *k* index *i* to layer
*k+1* index *i* — so the skin is reparametrised around the ribs rather
than having vertices deleted. And the caps are **ear-clipped**, not
laddered between upper and lower surfaces, because a rib detour doubles
back in x.

The rib sweep is a **budget, not an allowance**. A tapered swept panel
already spends 37.9° of a 50° overhang limit moving its own section
sideways before any rib exists; the truss gets the remainder. Sizing the
rib against the full limit gave 58° of real overhang — and because the
excess came from the wing, it was independent of rib pitch, which is
what eventually identified it.

`structure.py` then sizes the spar: the lightest stock carbon tube that
passes ultimate load (1.5× limit) and a tip-deflection limit, from the
VLM's own span loading rather than an assumed elliptical one. Rib pitch
comes from plate-buckling of the compression skin. Both are calculations,
not defaults.

## Designing for a beginner

`Mission.beginner_trainer()` is judged against one question: what happens
when a first-time pilot lets go of the sticks? That needs a big static
margin, real dihedral, low wing loading, and — the important one — a
**stall-progression gate**: the tip must work at least 12% below the peak
section cl, so the root lets go first and the nose drops instead of a
wing. A flying wing that drops a tip on launch is how beginners lose them.

Two things that gate turned up. Static margin is **pinned by sweep**, not
by reflex or camber: it moves only with CG position, and the battery
cannot go further forward without the pack hanging off the nose. Sweeping
the leading edge moves the neutral point aft instead — 38° → SM 0.081,
46° → 0.240, 54° → 0.390. And the optimizer will happily trim at +14.3°
if you let it, because the lattice is inviscid and tier-0 drag has no
alpha dependence, so `max_trim_alpha_deg` caps it.

## The fidelity ladder

A D3Q19 run of a whole aircraft is hours; differential evolution wants
thousands of evaluations. Putting the tunnel in the loop would buy one
design a week. So the tunnel is spent where it changes a decision:

| tier | model | cost | used for |
|---|---|---|---|
| 0 | VLM + flat-plate Cf | ~0.4 s | the search — thousands of designs |
| 1 | 2D LBM section polar | ~1 min | the finalists' sections |
| 2 | 3D LBM whole aircraft | hours | the one you are going to print |

Tier 1 replaces the weakest number in tier 0: profile drag, guessed by
Prandtl-Schlichting with a form factor and a printed-surface roughness
multiplier, and wrong by 20-40% on a 10%-thick section at Re 100k-400k
where the boundary layer is transitional. That is exactly the regime
`windtunnel-sim` was validated in (Phase 6, MH45 at Re 20k).

`aero/tunnel.py` **shells out** to that repo rather than importing it.
The tunnel needs CUDA, and it has its own units discipline and its own
validation gates; importing the solver here would fork those rules.
Writing a scene file and calling its entry point keeps one source of
truth for the physics — and lets polars be produced on the WSL/GPU side
while the search runs anywhere. Polars are cached by a content hash of
the CST coefficients and Reynolds number, so a repeat section is free and
a miss is a deliberate spend.

## Why the design vector looks like this

**CST (Kulfan) sections.** The optimizer needs a small, always-valid
parametrization, and blending two point clouds is ill-defined while
blending two coefficient vectors is a lerp. Fitting is linear least
squares, so any Selig `.dat` becomes a design vector exactly.

**Reflex is a trailing-edge deflection, not a coefficient nudge.** This
one cost an afternoon. The class function `C(x) = x^0.5 (1-x)` is *zero
at x = 1*, so `C(x)S(x)` pins the camber line to the chord line at the
trailing edge: a reflexed section is literally unrepresentable without an
explicit `x·z_te` camber term. Fitting one anyway produces a violent
downward hook over the last 1% of chord, which behaves like a large
*down*-flap — and Cm0 comes out with the wrong sign. Both terms now
exist: `te_gap` (opposite signs on the two surfaces) is thickness,
`te_camber` (same sign) is camber.

**Battery position is a design variable.** On a tailless aircraft the CG
is the strongest single lever on trim. Fixing it by hand and optimising
the wing around it is solving the problem backwards; the pack is the one
part that is trivial to slide.

**Cruise speed is an output, not an input.** A wing with fixed elevons
trims at *one* lift coefficient, set by camber, twist and CG. So trim
fixes CL, and speed follows from wing loading:

```
Cm(alpha) = 0  ->  CL_trim  ->  v = sqrt(2W / (rho S CL_trim))
```

A design that trims at CL 0.15 is not "efficient", it is a 30 m/s
aeroplane. The mission band is what says whether that is what you asked
for.

## What the gates actually caught

Each of these was a real bug, and each is now a test in
`tests/test_validation.py`:

- **Cap winding inverted** — both STL end caps wound against the side
  ring, giving exactly 2N inconsistent windings and a mesh a slicer would
  quietly patch for you.
- **Wall-separation gate measured the wrong thing** — the y-difference of
  rotated points instead of the rigid-body distance, and it included the
  shared leading-edge point, whose thickness is trivially zero. Every
  design failed.
- **Overhang measured against nominal layer height** while the search
  sampled coarsely — reported 82° of overhang on a wing that has 42°.
- **Reflex sign inverted** (above), which made every tailless design
  untrimmable.
- **Best-score tracking returned infeasible designs** — the penalised
  score exists to give the optimizer a slope back into the feasible set,
  not to define the answer.

Independent cross-checks that hold: the skinned mesh's
divergence-theorem volume agrees with the planform's section-area
integral to 0.3%; the VLM puts the aerodynamic centre at 0.2473c, reaches
2π in the 2D limit, gives Oswald e = 0.99 for an elliptic wing and less
for a rectangular one, and reproduces published NACA 4412 Cm.

## Known limits, stated because the objective weighs them

- The VLM is **inviscid**: there is no stall and no separation. CL_max
  comes from section data, never from the lattice. Post-stall behaviour
  is outside its reach entirely.
- Lift slope runs a few percent below lifting-line at moderate AR. That
  is the normal lifting-line-vs-lifting-surface gap, not an error, but it
  is a systematic bias in absolute L/D.
- Tier 0 profile drag is an estimate. Absolute L/D should not be quoted
  from a tier-0 number; rankings are far more trustworthy than levels.
- MH45's own reflex sits close to the CST fit's resolution. The seed does
  not perfectly reproduce its published Cm0, which is why reflex is an
  explicit design variable rather than something inherited from the seed.
- Structural strength is **not modelled at all**. The spar bore is
  checked for fit, not for bending. Nothing here says the wing survives a
  launch.

## Layout

```
loft/
  geom/      cst.py        CST sections, fitting, blending, reflex deflection
             planform.py   BWB stations, lofting, planform integrals
  aero/      vlm.py        vortex lattice on the mean camber surface
             performance.py mass budget, trim, static margin, drag build-up
             tunnel.py     bridge to windtunnel-sim; cached section polars
  printing/  vase.py       layer stacks, TE thickening, printability gates
             stl.py        watertight skinning, binary STL
  search/    design.py     design vector, mission, scoring
             optimize.py   differential evolution
run.py       CLI
tests/       the gates above
```
