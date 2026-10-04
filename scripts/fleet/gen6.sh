#!/usr/bin/env bash
# Generation 6: the first fleet searched against the BUILD.
#
# Every earlier generation scored an aeroplane and left whether anyone
# could assemble one to a person with a knife. This one is searched
# against the parts as well, and against a dozen gates that did not
# exist when gen5 ran:
#
#   * the openings are CUT, with ramps inside the overhang budget, and
#     each is a rigid box at an absolute station rather than a chord
#     fraction that slides as the wing tapers;
#   * the bonded joints between panels carry the shear and the torque
#     the spar does not, at a declared allowable;
#   * the propeller disc must clear the trailing edge where the BLADE
#     TIPS reach, which on a swept pusher is not the centreline -- micro
#     sat 17.6 mm inside its own trailing edge and nothing had asked;
#   * the servo's output shaft must lie under the surface it drives, and
#     its lead must have somewhere to run;
#   * the control horn's socket must hold the servo's stall torque;
#   * the motor mount's bond must carry static thrust and motor torque;
#   * a panel must be able to carry the rib truss the buckling gate
#     sized for it;
#   * no two vertices of a contour may coincide, which is the difference
#     between a watertight mesh and one a slicer quietly patches.
#
# gen5's winners ARE seeded, one per mission, even though each misses
# between seven and fourteen of these gates. They remain the best
# designs anyone has on the flight axes -- trim, damping, stall
# progression -- and those did not change; what they fail is the build,
# which is what this search has to fix. A feasibility probe of 595
# random draws found none feasible and only a handful passing the
# rarest gates, so starting the whole population from scratch would
# spend most of it rediscovering what the seed already knows.
#
# Sized for the machine rather than for ambition. An evaluation now
# costs about 9.5 s against about 2 s when gen5 ran, because every one
# of them cuts the bays, builds the elevons and their sockets and skins
# the lot: gen5's popsize 6 over 45 iterations would be 25 hours per
# search. 4 over 22 is about 7, and nine of them fit on 24 cores at one
# BLAS thread each.
cd "$(dirname "$0")/../.." || exit 1

# The same two measured polars gen5 used, tracked in the repo. See
# data/polars/README.md: these are two DIFFERENT sections and the Re
# trend between them is partly thickness.
POLAR="data/polars/trainer_mid_re60k.csv@60000,data/polars/thin_reflex_re100k.csv@100000"
mkdir -p out/gen6

declare -A SEED_DESIGN=(
  [trainer_v3]=results/fleet/gen5_trainer_v3_v101/design.json
  [demon1]=results/fleet/gen5_demon1_v102/design.json
  [micro]=results/fleet/gen4_micro_v92_sized/design.json
)

pids=()
for m in trainer_v3 demon1 micro; do
  for s in 0 1 2; do
    tag="${m}_v11${s}"
    python run.py search --mission "$m" --iters 22 --popsize 4 --seed "$s" \
        --polar "$POLAR" --seed-design "${SEED_DESIGN[$m]}" \
        --out "out/gen6/${tag}" > "out/gen6/${tag}.log" 2>&1 &
    pids+=($!)
    echo "launched gen6/${tag} seeded from ${SEED_DESIGN[$m]} (pid $!)"
  done
done
echo "waiting on ${#pids[@]} searches..."
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "ALL DONE, $fail nonzero exits"
for m in trainer_v3 demon1 micro; do
  for s in 0 1 2; do
    f="out/gen6/${m}_v11${s}/design.json"
    if [ -f "$f" ]; then
      python - "$f" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"  {sys.argv[1]:<34} feasible={d['feasible']}  score={d['score']:.2f}  "
      f"{d['mass_kg']*1000:.0f} g  {len(d['reasons'])} misses")
PY
    fi
  done
done
