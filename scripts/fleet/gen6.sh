#!/usr/bin/env bash
# Generation 6: the first fleet searched against the BUILD.
#
# Every earlier generation scored an aeroplane and left the question of
# whether anyone could assemble one to a person with a knife. This one is
# searched against the parts as well, and against a dozen gates that did
# not exist when gen5 ran:
#
#   * the openings are CUT, with ramps inside the overhang budget, and
#     each is a box at an absolute station rather than a chord fraction;
#   * the bonded joints between panels carry the shear and torque the
#     spar does not, at a declared allowable;
#   * the propeller disc must clear the trailing edge where the BLADE
#     TIPS reach, which on a swept pusher is not the centreline -- micro
#     was 17.6 mm inside its own trailing edge and nothing had asked;
#   * the servo's output shaft must lie under the surface it drives, and
#     its lead must have somewhere to run;
#   * the control horn's socket must hold the servo's stall torque;
#   * the motor mount's bond must carry static thrust and motor torque;
#   * a panel must be able to carry the rib truss the buckling gate
#     sized for it.
#
# So gen5's winners are NOT seeds here. They were searched against a
# different question and they miss between eight and fourteen of these
# gates; seeding from them would start the population in a corner that
# is now known to be wrong, which is the opposite of what seeding is
# for. This generation starts from scratch.
cd "$(dirname "$0")/../.." || exit 1

# The same two measured polars gen5 used, tracked in the repo. See
# data/polars/README.md: these are two DIFFERENT sections and the Re
# trend between them is partly thickness.
POLAR="data/polars/trainer_mid_re60k.csv@60000,data/polars/thin_reflex_re100k.csv@100000"
mkdir -p out/gen6

pids=()
for m in trainer_v3 demon1 micro; do
  for s in 0 1 2; do
    tag="${m}_v11${s}"
    python run.py search --mission "$m" --iters 45 --popsize 8 --seed "$s" \
        --polar "$POLAR" \
        --out "out/gen6/${tag}" > "out/gen6/${tag}.log" 2>&1 &
    pids+=($!)
    echo "launched gen6/${tag} (pid $!)"
  done
done
echo "waiting on ${#pids[@]} searches..."
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "ALL DONE, $fail nonzero exits"
for m in trainer_v3 demon1 micro; do
  for s in 0 1 2; do
    f="out/gen6/${m}_v11${s}/design.json"
    [ -f "$f" ] && python - "$f" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"  {sys.argv[1]:<34} feasible={d['feasible']}  score={d['score']:.2f}  "
      f"{d['mass_kg']*1000:.0f} g  {len(d['reasons'])} misses")
PY
  done
done
