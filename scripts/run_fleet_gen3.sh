#!/usr/bin/env bash
# Generation 3: the same fleet on a generator that has to make ONE shape.
#
# gen2 passed every flight and print gate and still looked assembled from
# parts. What changed under this fleet:
#   * the loft runs through the SLOPES (sweep, dihedral, taper rate),
#     linear between segment midpoints and pinned to zero on the
#     centreline -- no overshoot, no V nose, no trailing-edge notch
#   * chord is a running product of tapers <= 1, sweep may only relax
#     outboard, twist is one smooth washout curve, dihedral turns up in
#     stages -- the waves are unrepresentable, not just penalised
#   * root t/c, winglet height and body thickness slope are clamped to
#     the mission's fairness limits inside build()
#   * a fairness gate measured on the lofted surface runs before any
#     lattice: unfair shapes are rejected in ~30 ms and rank below every
#     fair one
cd "$(dirname "$0")/.." || exit 1
POLAR="out/tunnel/polar.csv@60000,out/tunnel_re100/polar.csv@100000"
mkdir -p out/gen3
pids=()
for m in trainer_v3 demon1 micro; do
  for s in 0 1 2; do
    tag="${m}_v8${s}"
    python run.py search --mission "$m" --iters 45 --popsize 6 --seed "$s" \
        --polar "$POLAR" --out "out/gen3/${tag}" > "out/gen3/${tag}.log" 2>&1 &
    pids+=($!)
    echo "launched gen3/${tag} (pid $!)"
  done
done
echo "waiting on ${#pids[@]} searches..."
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "ALL DONE, $fail nonzero exits"
