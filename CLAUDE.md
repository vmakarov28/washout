# Project conventions (read before touching anything)

Automated blended-wing-body design: a search over planform + section
that produces vase-mode-printable STLs. Companion to the wind tunnel in
`~/Desktop/windtunnel` — that repo owns the physics of the tunnel, this
one owns the aircraft. Target machine: RTX 5080, WSL2 Ubuntu on Win 11;
the search itself is numpy/scipy only and runs on the Windows side too.

`docs/ROADMAP.md` is the current plan and the list of known debts, in
value order. Read it before starting anything; the top items mean some
reported numbers are not yet the numbers that would be built.
`docs/ROADMAP-BUILD.md` is the other axis: what has to exist before an
export prints and flies without a CAD step. Its section 0 derives what
the one-contour-per-layer constraint does and does not allow, and every
feature request should be classified against it before being designed.
`docs/ROADMAP-CAD.md` is the third: the aircraft in 3D. Every gate looks
at one section at a time, and a tube being straight or a panel being
straight lives between sections -- both were wrong until 2026-09-22. The
STEP export (OpenCASCADE, an optional install, never imported by the
search) is built from the loft and checks the print against it.

## Non-negotiable rules

- **Feasibility dominates score.** Any design meeting the mission beats
  any design that does not. A single weighted sum was tried and failed:
  the optimizer parked 1 m/s below the cruise band because the penalty
  was cheaper than the L/D it bought. Penalties rank the infeasible; they
  never outrank feasibility.
- **Cruise speed is an OUTPUT.** Trim fixes CL, CL and wing loading fix
  speed. Never assume a speed and solve the wing around it.
- **Every printability claim is a gate with a number**, in
  `printing/vase.py`, reported pass/fail with the measured value and the
  limit. No warnings, no "should be fine".
- **The tunnel is shelled out to, never imported.** `windtunnel-sim` has
  its own units discipline and validation gates; importing its solver
  would fork them. Write a scene, call `run.py`, cache the polar by a
  content hash of the CST coefficients + Re.
- **Absolute L/D from tier 0 is not quotable.** Tier-0 profile drag is a
  flat-plate estimate. Rankings are trustworthy, levels are not. Say so
  wherever a number leaves the program.
- **A bug found becomes a test**, in `tests/test_validation.py`, with the
  physical reason in the docstring. The suite is the specification.
- Geometry is validated against closed-form or published references
  only: 2π in the 2D limit, AC at 0.25c, Oswald e for elliptic vs
  rectangular, published NACA 4412 Cm, mesh volume vs planform integral.
  Never tune a constant to match a reference.

## Architecture: the fidelity ladder is the design

| tier | model | cost | used for |
|---|---|---|---|
| 0 | VLM + flat-plate Cf | ~0.4 s | the search |
| 1 | 2D LBM section polar | ~1 min | finalists' sections |
| 2 | 3D LBM whole aircraft | hours | the one being printed |

Never put tier 1 or 2 inside the optimizer loop. The reason the ladder
exists is that a whole-aircraft D3Q19 run is hours and DE wants thousands
of evaluations.

## Two traps that already cost a session

1. **CST cannot represent reflex without a TE-camber term.** The class
   function is zero at x=1, so the camber line is pinned to the chord
   line at the trailing edge. `te_gap` (opposite signs) is thickness;
   `te_camber` (same sign) is camber. Both are required. Nudging raw CST
   coefficients is NOT a reflex knob — it builds a bump whose tail slopes
   down, which is a positive flap deflection, and Cm0 goes the wrong way.
   Use `deflect_te()`.
2. **Overhang and mass must both respect the sampling pitch.** The search
   samples layers coarsely. Mass is weighted by layers-per-sample;
   overhang uses the ACTUAL z rise between sampled contours. Using the
   nominal layer height reports 82° on a wing that has 42°.

## Print orientation is fixed and load-bearing

Panels print root-down, span along Z, so each layer is one aerofoil
contour. Do not add a "print flat" mode — a horizontal slice of a flat
wing is multiple contours and spiralize mode cannot express it. The spar
runs along print Z, so it needs no hole through the wall; the cavity is
the channel.
