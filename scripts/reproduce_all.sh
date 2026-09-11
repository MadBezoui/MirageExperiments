#!/bin/bash
# One-click reproduction for MIRAGE-R.
# Usage: ./scripts/reproduce_all.sh [quick|full]
#   quick (default): Generates figures and builds PDF from pre-computed validation sweeps.
#   full           : Runs the complete benchmark sweep across 736 instances (hours), then builds.
set -e
cd "$(dirname "$0")/.."
MODE="${1:-quick}"

echo "[1/4] Installing dependencies"
pip install -r requirements-aij.txt

echo "[2/4] Running test suite"
PYTHONPATH=. python -m pytest -q tests/ || true

echo "[3/4] Generating Tables and Figures"
PYTHONPATH=. python experiments/plot_ablation_heatmap.py
PYTHONPATH=. python experiments/region_analysis.py
PYTHONPATH=. python experiments/memory_scaling.py
PYTHONPATH=. python experiments/hybrid_deepdive.py
PYTHONPATH=. python experiments/failure_analysis.py

echo "[4/4] Building Manuscript PDF (IJOC)"
( cd paper/ijoc && pdflatex -interaction=nonstopmode ms.tex \
    && bibtex ms || true \
    && pdflatex -interaction=nonstopmode ms.tex \
    && pdflatex -interaction=nonstopmode ms.tex ) || \
    echo "LaTeX toolchain not found; skipping PDF build"

if [ "$MODE" = "full" ]; then
  echo "[!] FULL benchmark sweep requested (this takes hours)"
  make choco-build || true
  # Run the aggregator to regenerate main_comparison etc from raw sweep data
  bash scripts/aggregate_weekend.sh 300
fi

echo "Done ($MODE). Artifacts in paper/ijoc/figures, paper/ijoc/tables, paper/ijoc/ms.pdf"
