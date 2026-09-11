# Weekend Validation Suite

A robust, **shardable, resumable** harness that fully validates the MIRAGE-R paper
on a dedicated machine over a weekend, then aggregates everything into the
paper's tables and figures.

## What it produces

- **Main comparison** of MIRAGE-R, MIRAGE-R+regions, the MIRAGE-R→CP-SAT hybrid,
  OR-Tools CP-SAT, and Choco: coverage, #SAT/#UNSAT, median time, PAR2.
- **Soundness cross-check**: MIRAGE-R must never report SAT on an instance a
  complete solver proved UNSAT (zero false positives) — written to
  `tables/soundness.txt` (`PASS`/`FAIL`).
- **Ablation** over temperature `τ`, polarization growth `β`, and region topology.
- **Memory profiling** vs CP-SAT.
- **Convergence trajectories** validating the `O(√(log m / T))` rate (Theorem 1).
- **Performance profile** (Dolan–Moré) and **cactus plot**.
- **Wilcoxon signed-rank** tests (Holm-corrected) between solver pairs.

## How to run (5 parallel jobs)

One machine, 5 background jobs:

```bash
# optional: build the Java baseline first
make choco-build
# launch all 5 shards (logs in results/raw/weekend_logs/)
TIMEOUT=300 SEEDS="0 1 2 3 4" CONCURRENCY=12 SOLVER_WORKERS=4 \
  bash scripts/launch_all_weekend.sh
```

Five separate nodes (run one per node, index 0..4):

```bash
bash scripts/run_weekend_validation.sh 0 5   # node 0
bash scripts/run_weekend_validation.sh 1 5   # node 1
... etc ...
```

Each job runs, **for its shard**: main sweep → full ablation → memory profiling;
shard 0 also runs the light theory/convergence phase. Tasks are split
deterministically (`task_index % num_shards == shard_id`), so the five jobs do
disjoint work.

## Resume after interruption

Just re-run the same command(s). Every phase records results incrementally to a
per-shard JSONL and skips `(instance, solver, seed)` (or config) tuples already
present. Safe to Ctrl-C and restart.

## Robustness

- Every task runs in its **own subprocess** with a hard wall-clock timeout
  (`timeout + grace`), so a hang/OOM/segfault in one instance never kills a shard.
- Empty/oversized instances are skipped and recorded (`SKIPPED_SIZE`), not crashed.
- Choco runs cross-platform; if `java` or the jar is missing, the task is recorded
  as `ERROR` and the rest proceeds.

## Aggregate into the paper

```bash
bash scripts/aggregate_weekend.sh 300
```

Writes LaTeX tables to `paper/ijoc/tables/` and figures to `paper/ijoc/figures/`.
Then in `paper/ijoc/ms.tex`, replace the `[sweep]` placeholder table with:

```latex
\input{tables/main_comparison.tex}
\input{tables/per_family_coverage.tex}
\input{tables/wilcoxon.tex}
\input{tables/memory_summary.tex}
```

and rebuild (`make paper-ijoc` or the pdflatex+bibtex sequence). Check
`tables/soundness.txt` says **PASS**.

## Key parameters (environment variables)

| var | default | meaning |
|---|---|---|
| `TIMEOUT` | 300 | per-task timeout (s) |
| `SEEDS` | `0 1 2 3 4` | random seeds |
| `CONCURRENCY` | 8 | tasks in flight per shard |
| `SOLVER_WORKERS` | 4 | threads CP-SAT/hybrid may use per task |
| `MANIFEST` | `data/frozen/instances_clean.csv` | instance list |
| `SOLVERS` | all five | solvers to run |
| `SWEEP_MAXMB` | 0 (no cap) | skip instances larger than this in the sweep |
| `ABL_MAXMB` | 5 | size cap for ablation instances |
| `MEM_MAXMB` | 20 | size cap for memory profiling |
| `JAVA` / `CHOCO_CP` | `java` / target classpath | Choco invocation |
| `OUTBASE` | `results/raw` | where raw shard JSONL go |

## Capacity planning

With `CONCURRENCY × num_shards` tasks in flight, total wall-clock ≈
`(#instances × #solvers × #seeds × avg_task_time) / (CONCURRENCY × num_shards)`
plus the ablation grid (`|τ|·|β|·2 × #instances(≤ABL_MAXMB) × #seeds`). Reduce
`SEEDS`, raise `CONCURRENCY`, or set `SWEEP_MAXMB` to fit your time budget.
Everything is resumable, so you can stop Sunday night and aggregate whatever
finished.
