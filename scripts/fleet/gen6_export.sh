#!/usr/bin/env bash
# Export every gen6 design as its search lands, feasible or not.
#
# The search refuses to write an STL for a design that misses a gate,
# which is right for an automated fleet: nobody should print something
# nothing checked. But gen6's leaders all miss the same two WIRING
# gates, which are a limitation of the one-trapezoid-per-detour cut
# model rather than a fault in any aeroplane -- and the panels
# themselves pass every printability gate. So the STLs are worth having
# now, with the misses stamped on the build sheet, and `run.py export`
# already writes them with the issues printed.
#
# Polls rather than waits on a pid, so it survives being started after
# some searches have already finished.
cd "$(dirname "$0")/../.." || exit 1
POLAR="data/polars/trainer_mid_re60k.csv@60000,data/polars/thin_reflex_re100k.csv@100000"

pending() {
  local n=0
  for m in trainer_v3 demon1 micro; do
    for s in 0 1 2; do
      d="out/gen6/${m}_v11${s}"
      [ -f "$d/design.json" ] || { n=$((n+1)); continue; }
      compgen -G "$d/*.stl" > /dev/null || n=$((n+1))
    done
  done
  echo "$n"
}

while :; do
  for m in trainer_v3 demon1 micro; do
    for s in 0 1 2; do
      tag="${m}_v11${s}"; d="out/gen6/${tag}"
      [ -f "$d/design.json" ] || continue
      compgen -G "$d/*.stl" > /dev/null && continue
      echo "[$(date +%H:%M:%S)] exporting ${tag}"
      python run.py export --mission "$m" --design "$d/design.json" \
          --out "$d" --polar "$POLAR" > "$d/export.log" 2>&1 \
        && echo "[$(date +%H:%M:%S)]   ${tag}: $(ls "$d"/*.stl 2>/dev/null | wc -l) STLs" \
        || echo "[$(date +%H:%M:%S)]   ${tag}: EXPORT FAILED, see $d/export.log"
    done
  done
  [ "$(pending)" -eq 0 ] && { echo "[$(date +%H:%M:%S)] all nine exported"; break; }
  sleep 300
done
