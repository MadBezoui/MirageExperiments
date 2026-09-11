#!/usr/bin/env bash
# Aggregate all weekend shards into the paper's tables and figures.
#   bash scripts/aggregate_weekend.sh [timeout_seconds]
# Writes LaTeX tables to paper/ijoc/tables/ and figures to paper/ijoc/figures/,
# so the manuscript can \input / \includegraphics them directly.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
TIMEOUT="${1:-${TIMEOUT:-300}}"

python -m experiments.validation.aggregate \
  --raw-dir "${OUTBASE:-results/raw}" --sweep-subdir weekend \
  --timeout "${TIMEOUT}" \
  --tables-dir paper/ijoc/tables --figures-dir paper/ijoc/figures

echo
echo "Done. Review:"
echo "  paper/ijoc/tables/main_comparison.{csv,tex}"
echo "  paper/ijoc/tables/soundness.txt   (must say PASS / 0 false positives)"
echo "  paper/ijoc/figures/{cactus_plot,performance_profile,convergence,ablation_tau_beta,memory_profile}.pdf"
echo
echo "Then in paper/ijoc/ms.tex replace the [sweep] placeholder table with:"
echo "  \\input{tables/main_comparison.tex}   (and the other tables as desired)"
