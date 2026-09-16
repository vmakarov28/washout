#!/usr/bin/env bash
# Generation 4: the same fleet, judged on whether its Dutch roll dies out.
#
# What changed under this fleet:
#   * lateral dynamics (aero/dynamics.py): Dutch-roll damping and spiral
#     mode from the linearised lateral equations, with wing derivatives
#     from the same vortex lattice that trims the aircraft
#   * every mission now gates on Dutch-roll damping ratio >= 0.08
#     (MIL-F-8785C Level 1); the trainer also gates on spiral divergence
#   * the trainer's roll/yaw ratio ceiling tightened from 11 to 8.5
#   * vertical tip fins (aero/fins.py) are three new design variables:
#     flat plates printed on the bed, charged for mass and drag, and
#     exported as their own STL when a design uses them
cd "$(dirname "$0")/../.." || exit 1
# The same two files as before, now tracked in the repo so a fresh clone
# can run a measured-drag search. See data/polars/README.md: these are two
# DIFFERENT sections, and the Re trend between them is partly thickness.
POLAR="data/polars/trainer_mid_re60k.csv@60000,data/polars/thin_reflex_re100k.csv@100000"
mkdir -p out/gen4
pids=()
for m in trainer_v3 demon1 micro; do
  for s in 0 1 2; do
    tag="${m}_v9${s}"
    python run.py search --mission "$m" --iters 45 --popsize 6 --seed "$s" \
        --polar "$POLAR" --out "out/gen4/${tag}" > "out/gen4/${tag}.log" 2>&1 &
    pids+=($!)
    echo "launched gen4/${tag} (pid $!)"
  done
done
echo "waiting on ${#pids[@]} searches..."
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "ALL DONE, $fail nonzero exits"
