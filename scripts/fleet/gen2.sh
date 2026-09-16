#!/usr/bin/env bash
# Generation 2 of the geometry generator, run into its own directory.
#
# What changed under this fleet, all of it invisible to the old runs:
#   * five stations, not four, with the outer break placed by a gap
#     fraction so the ordering constraint stays unrepresentable
#   * polyhedral: three dihedrals; the last one is the winglet
#   * a real spanwise section family -- root and tip are different
#     airfoils, lerped in CST coefficient space
#   * induced drag from a TWO-dimensional Trefftz plane, so a winglet
#     is finally worth something instead of being priced at zero
#   * directional and roll stability gated for the first time
cd "$(dirname "$0")/../.." || exit 1
# The same two files as before, now tracked in the repo so a fresh clone
# can run a measured-drag search. See data/polars/README.md: these are two
# DIFFERENT sections, and the Re trend between them is partly thickness.
POLAR="data/polars/trainer_mid_re60k.csv@60000,data/polars/thin_reflex_re100k.csv@100000"
pids=()
for m in trainer_v3 demon1 micro; do
  for s in 0 1 2; do
    tag="${m}_v7${s}"
    python run.py search --mission "$m" --iters 45 --popsize 6 --seed "$s" \
        --polar "$POLAR" --out "out/gen2/${tag}" > "out/gen2/${tag}.log" 2>&1 &
    pids+=($!)
    echo "launched gen2/${tag} (pid $!)"
  done
done
echo "waiting on ${#pids[@]} searches..."
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "ALL DONE, $fail nonzero exits"
