#!/usr/bin/env bash
# Reproduce everything in the paper that can be reproduced, and verify it.
#
#   ./scripts/reproduce.sh            # tests + code-only results + audit + PDFs
#   ./scripts/reproduce.sh tests      # unit and regression tests only
#   ./scripts/reproduce.sh theory     # Theorem 1 verification (Table 1, Figure 2)
#   ./scripts/reproduce.sh diagnostics# fixed-point diagnostics (Figure 4)
#   ./scripts/reproduce.sh audit      # archive audit from the retained raw logs
#   ./scripts/reproduce.sh paper      # build manuscript.pdf, ec.pdf, paper.pdf
#   ./scripts/reproduce.sh verify     # rerun everything and diff against the
#                                     # committed tables: nothing may change
#
# What CANNOT be reproduced, by design and stated in the paper: the archived
# 736-entry sweep. The normalized representations, their provenance and the
# derived feature records were not preserved, so only the retained raw logs
# survive and only the audit over them can be re-run.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-.}"

MODE="${1:-all}"

if [ -f paper/ijoc/paper.tex ]; then PAPERDIR=paper/ijoc
elif [ -f ../paper/ijoc/paper.tex ]; then PAPERDIR=../paper/ijoc
else PAPERDIR=""
fi

if [ -n "$PAPERDIR" ]; then
  export PAPERDIR
  TABLES=$PAPERDIR/tables
  FIGURES=$PAPERDIR/figures
else
  # If manuscript is absent, generate into a local directory
  TABLES=results/generated/tables
  FIGURES=results/generated/figures
  if [[ "$MODE" == "verify" || "$MODE" == "paper" || "$MODE" == "all" ]]; then
    echo "paper/ijoc/paper.tex not found, here or beside this repository."
    echo "Table regeneration and the PDF build need it; see README.md."
    exit 2
  fi
  mkdir -p "$TABLES" "$FIGURES"
fi

PY=${PYTHON:-python3}

# The Choco baseline jar is built for Java 17; a Java 11 runtime fails with
# UnsupportedClassVersionError. Point JAVA_HOME at a 17+ JDK if one is around.
if [ -z "${JAVA_HOME:-}" ] && [ -x /usr/local/opt/openjdk@17/bin/java ]; then
  export JAVA_HOME=/usr/local/opt/openjdk@17
  export PATH="$JAVA_HOME/bin:$PATH"
fi

hr() { printf '\n\033[1m== %s\033[0m\n' "$1"; }

run_tests() {
  hr "unit and regression tests"
  "$PY" -m pytest tests/ -q
}

run_theory() {
  hr "Theorem 1 against the implementation (Table 1, Figure 2, theory macros)"
  "$PY" -m experiments.theory_multifactor --out "$1" --figdir "$FIGURES"
  hr "exact trajectories of the implemented map (Figure 2)"
  "$PY" -m experiments.theory_portrait --out "$1" --figdir "$FIGURES"
}

run_diagnostics() {
  hr "fixed-point diagnostics, 7 configurations x 8 seeds (Figure 4)"
  "$PY" -m experiments.fixedpoint_diagnostics \
      --n 30 --seeds 8 \
      --out results/raw/diagnostics --tables "$1" --figdir "$FIGURES"
}

run_audit() {
  hr "archive audit over the retained raw logs"
  "$PY" -m experiments.validation.audit_archive \
      --raw results/raw --tables "$1"
  hr "recomputing every coverage number the paper states, and asserting it"
  "$PY" -m experiments.validation.paper_numbers \
      --raw results/raw --tables "$1"
  hr "temperature-underflow arithmetic over the archived epoch counts"
  "$PY" -m experiments.validation.underflow_analysis \
      --raw results/raw --tables "$1"
  hr "decoded-progress trajectories (Figure 5)"
  "$PY" -m experiments.fig_trajectories \
      --raw results/raw/revision_theory/trajectories.jsonl \
      --out "$1" --figdir "$FIGURES"
}

run_paper() {
  hr "the manuscript"
  "$PY" scripts/build_submission.py
}

case "$MODE" in
  tests)        run_tests ;;
  theory)       run_theory "$TABLES" ;;
  diagnostics)  run_diagnostics "$TABLES" ;;
  audit)        run_audit "$TABLES" ;;
  paper)        run_paper ;;
  all)
    run_tests
    run_theory "$TABLES"
    run_diagnostics "$TABLES"
    run_audit "$TABLES"
    run_paper
    ;;
  verify)
    # Regenerate into a scratch directory and require byte-identical output.
    # This is what makes "no number in the paper is typed by hand" checkable.
    TMP=$(mktemp -d)
    trap 'rm -rf "$TMP"' EXIT
    run_tests
    run_theory "$TMP"
    run_diagnostics "$TMP"
    run_audit "$TMP"
    hr "diffing regenerated tables against the committed ones"
    # Only the tables the manuscript actually \input matter for the paper's
    # numbers. The others are regenerated too, but they are deposited in the
    # artifact rather than typeset, and their committed copies carry editorial
    # trims that the generators do not reproduce.
    used=$("$PY" - <<'INNEREOF'
import os, re, pathlib
src = (pathlib.Path(os.environ["PAPERDIR"]) / "paper.tex").read_text()
print(" ".join(sorted({m + ".tex" for m in
      re.findall(r"\\input\{tables/([a-z_]+)", src)})))
INNEREOF
)
    fail=0
    for b in $used; do
      f="$TMP/$b"
      [ -f "$f" ] || { echo "  (not generated, hand-maintained): $b"; continue; }
      if ! diff -q "$f" "$TABLES/$b" >/dev/null 2>&1; then
        echo "  CHANGED: $b"; diff "$f" "$TABLES/$b" | head -20; fail=1
      else
        echo "  ok: $b"
      fi
    done
    [ "$fail" -eq 0 ] || {
      echo
      echo "Regenerated tables differ from the committed ones."
      echo "Round-off-level columns (f.p. err.) can move in the last digit"
      echo "with a different BLAS; anything else is a real discrepancy."
      exit 1
    }
    run_paper
    echo
    echo "VERIFIED: every regenerated table matches the committed one."
    ;;
  *)
    echo "unknown mode: $MODE" >&2; exit 2 ;;
esac

echo
echo "done ($MODE)."
