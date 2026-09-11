#!/usr/bin/env bash
# Per-JOB entrypoint for the weekend validation. Launch 5 of these (shard 0..4),
# one per parallel job, on your dedicated machine.
#
#   bash scripts/run_weekend_validation.sh <shard_id> [num_shards]
#
# Each job runs, for its shard: the main sweep, then the full ablation, then the
# memory profiling. Shard 0 also runs the (light) theory/convergence phase at the
# end. Everything is RESUMABLE: re-running the same command continues where it
# stopped.
#
# Override any of these via environment variables:
#   TIMEOUT(=300) SEEDS("0 1 2 3 4") CONCURRENCY(=8) SOLVER_WORKERS(=4)
#   MANIFEST(=data/frozen/instances_clean.csv) OUTBASE(=results/raw)
#   SOLVERS("mirage mirage_regions hybrid ortools choco")
#   SWEEP_MAXMB(=0  no cap) ABL_MAXMB(=5) MEM_MAXMB(=20)
#   JAVA(=java) CHOCO_CP(=baselines/choco/target/classes:baselines/choco/target/dependency/*)
set -uo pipefail

SHARD="${1:?usage: run_weekend_validation.sh <shard_id> [num_shards]}"
NUM_SHARDS="${2:-5}"

cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"

TIMEOUT="${TIMEOUT:-300}"
SEEDS="${SEEDS:-0 1 2 3 4}"
CONCURRENCY="${CONCURRENCY:-8}"
SOLVER_WORKERS="${SOLVER_WORKERS:-4}"
MANIFEST="${MANIFEST:-data/frozen/instances_clean.csv}"
OUTBASE="${OUTBASE:-results/raw}"
SOLVERS="${SOLVERS:-mirage mirage_regions hybrid ortools choco gecode runcsp}"
SWEEP_MAXMB="${SWEEP_MAXMB:-0}"
ABL_MAXMB="${ABL_MAXMB:-5}"
MEM_MAXMB="${MEM_MAXMB:-20}"
JAVA="${JAVA:-java}"
CHOCO_CP="${CHOCO_CP:-baselines/choco/target/choco-baseline-0.1.0.jar}"

echo "=== weekend validation: shard ${SHARD}/${NUM_SHARDS} | timeout=${TIMEOUT}s | seeds=[${SEEDS}] ==="
date

echo ">>> [1/4] main sweep"
python -m experiments.validation.run_sweep \
  --manifest "${MANIFEST}" --solvers ${SOLVERS} --seeds ${SEEDS} \
  --timeout "${TIMEOUT}" --num-shards "${NUM_SHARDS}" --shard-id "${SHARD}" \
  --concurrency "${CONCURRENCY}" --solver-workers "${SOLVER_WORKERS}" \
  --max-mb "${SWEEP_MAXMB}" --out-dir "${OUTBASE}/weekend" \
  --java "${JAVA}" --choco-cp "${CHOCO_CP}"

echo ">>> [2/4] full ablation"
python -m experiments.validation.run_ablation_full \
  --manifest "${MANIFEST}" --seeds ${SEEDS} \
  --timeout "${TIMEOUT}" --num-shards "${NUM_SHARDS}" --shard-id "${SHARD}" \
  --concurrency "${CONCURRENCY}" --max-mb "${ABL_MAXMB}" \
  --out-dir "${OUTBASE}/weekend_ablation"

echo ">>> [3/4] memory profiling"
python -m experiments.validation.run_memory_full \
  --manifest "${MANIFEST}" \
  --timeout "${TIMEOUT}" --num-shards "${NUM_SHARDS}" --shard-id "${SHARD}" \
  --concurrency "$(( CONCURRENCY / 2 > 0 ? CONCURRENCY / 2 : 1 ))" --max-mb "${MEM_MAXMB}" \
  --out-dir "${OUTBASE}/weekend_memory"

if [ "${SHARD}" = "0" ]; then
  echo ">>> [4/4] theory / convergence trajectories (shard 0 only)"
  python -m experiments.validation.run_theory \
    --manifest "${MANIFEST}" --sample 40 --timeout "${TIMEOUT}" \
    --out-dir "${OUTBASE}/weekend_theory"
fi

echo "=== shard ${SHARD} COMPLETE ==="
date
