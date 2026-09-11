#!/usr/bin/env bash
# Launch all 5 weekend-validation jobs in parallel on ONE machine.
# Each job is a shard (0..4); logs go to results/raw/weekend_logs/.
#
#   bash scripts/launch_all_weekend.sh
#
# To run on 5 SEPARATE nodes instead, run on each node:
#   bash scripts/run_weekend_validation.sh <node_index 0..4> 5
#
# Tune via the same environment variables documented in run_weekend_validation.sh,
# e.g.:  TIMEOUT=300 SEEDS="0 1 2 3 4" CONCURRENCY=12 bash scripts/launch_all_weekend.sh
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/.local/jdk-17.0.10+7/bin:$PWD/.local/apache-maven-3.9.9/bin:$PATH"

NUM_SHARDS="${NUM_SHARDS:-5}"
LOGDIR="${OUTBASE:-results/raw}/weekend_logs"
mkdir -p "${LOGDIR}"

echo "Launching ${NUM_SHARDS} parallel jobs; logs in ${LOGDIR}/"
PIDS=()
for s in $(seq 0 $((NUM_SHARDS - 1))); do
  nohup bash scripts/run_weekend_validation.sh "${s}" "${NUM_SHARDS}" \
        > "${LOGDIR}/shard${s}.log" 2>&1 &
  PIDS+=($!)
  echo "  shard ${s} -> PID $! -> ${LOGDIR}/shard${s}.log"
done

echo "PIDs: ${PIDS[*]}"
echo "Monitor with:  tail -f ${LOGDIR}/shard*.log"
echo "Waiting for all shards to finish..."
wait "${PIDS[@]}"
echo "ALL SHARDS DONE. Now aggregate with:  bash scripts/aggregate_weekend.sh ${TIMEOUT:-300}"
