#!/usr/bin/env bash
# Rerun the whole fleet after the z-sampling and trim-CL fixes.
# Three seeds per mission: differential evolution from a random
# population is seed-sensitive, and 24 cores were sitting idle at two.
cd "$(dirname "$0")/../.." || exit 1
# The same two files as before, now tracked in the repo so a fresh clone
# can run a measured-drag search. See data/polars/README.md: these are two
# DIFFERENT sections, and the Re trend between them is partly thickness.
POLAR="data/polars/trainer_mid_re60k.csv@60000,data/polars/thin_reflex_re100k.csv@100000"
pids=()
for m in trainer_v3 demon1 micro; do
  for s in 0 1 2; do
    tag="${m}_v6${s}"
    python run.py search --mission "$m" --iters 45 --popsize 6 --seed "$s" \
        --polar "$POLAR" --out "out/${tag}" > "out/${tag}.log" 2>&1 &
    pids+=($!)
    echo "launched ${tag} (pid $!)"
  done
done
echo "waiting on ${#pids[@]} searches..."
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "ALL DONE, $fail nonzero exits"
