#!/usr/bin/env bash
# Generation 5: trainer and demon1, scored as BUILT.
#
# What changed under this fleet:
#   * evaluate() sizes the spar and the buckling-driven ribs and judges the
#     ribbed aircraft. Every earlier search scored a bare vase shell; the
#     ribs are 8-22 g aft of the CG, and they turned gen4's trainer and
#     demon1 winners infeasible once added (Dutch-roll zeta +0.083 -> +0.058
#     and +0.063).
#   * the search's own export uses the sized settings, so its STLs carry
#     the ribs. No earlier search's STLs did.
#   * each seed starts from a previous winner (--seed-design), so a
#     generation cannot finish behind a design that is still feasible.
#
# micro is not rerun: gen4 micro_v92 already passes with ribs, and its
# ribbed parts are in out/gen4/micro_v92_sized.
cd "$(dirname "$0")/../.." || exit 1
# The same two files as before, now tracked in the repo so a fresh clone
# can run a measured-drag search. See data/polars/README.md: these are two
# DIFFERENT sections, and the Re trend between them is partly thickness.
POLAR="data/polars/trainer_mid_re60k.csv@60000,data/polars/thin_reflex_re100k.csv@100000"
mkdir -p out/gen5

declare -A SEED_DESIGN=(
  [trainer_v3_0]=out/gen4/trainer_v3_v92/design.json
  [trainer_v3_1]=out/gen4/trainer_v3_v90/design.json
  [trainer_v3_2]=out/gen3/trainer_v3_v80/design.json
  [demon1_0]=out/gen4/demon1_v91/design.json
  [demon1_1]=out/gen3/demon1_v80/design.json
  [demon1_2]=out/gen3/demon1_v82/design.json
)

pids=()
for m in trainer_v3 demon1; do
  for s in 0 1 2; do
    tag="${m}_v10${s}"
    seed="${SEED_DESIGN[${m}_${s}]}"
    python run.py search --mission "$m" --iters 45 --popsize 6 --seed "$s" \
        --polar "$POLAR" --seed-design "$seed" \
        --out "out/gen5/${tag}" > "out/gen5/${tag}.log" 2>&1 &
    pids+=($!)
    echo "launched gen5/${tag} seeded from ${seed} (pid $!)"
  done
done
echo "waiting on ${#pids[@]} searches..."
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "ALL DONE, $fail nonzero exits"
