# Fractional Fixed Points and Decoded-Progress Plateaus in a Product-of-Experts Heuristic for Constraint Satisfaction

This archive accompanies the manuscript of the same name, submitted to the
[INFORMS Journal on Computing](https://pubsonline.informs.org/journal/ijoc),
and is distributed under the [MIT License](LICENSE).

Authors: Lillia Ouali, Madani Bezoui, Kamal Amroun, Ahcene Bounceur (see
[AUTHORS](AUTHORS)).

> **Placeholders.** The manuscript number, DOIs and the INFORMSJoC repository
> URL are assigned at acceptance. Replace `JOC-0000-0000.00` in
> `paper/ijoc/paper.tex` and the citation block below before the camera-ready
> upload, following the
> [INFORMSJoC author instructions](https://informsjoc.github.io/InstructionsForAuthors.html).

## What this repository contains

MIRAGE-R is a training-free, incomplete witness-search heuristic for
finite-domain constraint satisfaction. It keeps one categorical marginal per
variable, performs relation-conditioned marginalization over represented table
constraints, combines factor summaries by geometric consensus, and applies a
polarization schedule. The paper studies it as a diagnostic object rather than
a competitive solver: it characterizes its frozen dynamics exactly on Boolean
disequality graphs, measures how little a decoded-progress plateau says about
the continuous state, and audits an archived benchmark evaluation.

## Reproducibility: what can and cannot be re-run

| Component | Status |
|---|---|
| Unit and regression tests, including the four numerical corrections and the scope of Theorem 1 | reproducible |
| Theorem 1 verified against the implementation (Table 1, Figure 2) | reproducible |
| Exact trajectories of the implemented map (Figure 3) | reproducible |
| Fixed-point diagnostics, 7 configurations × 8 seeds (Figure 4) | reproducible |
| Archive audit and every coverage figure the paper states (Tables 3–6, Figure 5) | reproducible **from the retained raw logs** |
| The 736-entry benchmark sweep itself | **not reproducible** |

The sweep cannot be re-run and the paper says so throughout. The normalized
representations, their provenance, file checksums and the derived feature
records were not preserved; only the run-level logs under `results/raw/`
survive. Everything derived from those logs is recomputed here, and every
number is asserted rather than transcribed.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Public repository only
./scripts/reproduce.sh tests
./scripts/reproduce.sh theory
./scripts/reproduce.sh diagnostics
./scripts/reproduce.sh audit

# Complete submission bundle containing paper/ijoc/
./scripts/reproduce.sh verify
```

> **The manuscript is not part of this repository.** `reproduce.sh` and
> `build_submission.py` look for `paper/ijoc/paper.tex` in this tree and then
> beside it, and stop with a message when neither exists. Without it the tests,
> the theorem verification and the diagnostics all run; the byte-identical
> table comparison and the PDF build do not, because there is nothing to
> compare against.

`verify` runs the tests, regenerates every table and figure, requires each
regenerated table to be **byte-identical** to the one committed under
`../paper/ijoc/tables/`, and then rebuilds the three PDFs. It exits non-zero on
any discrepancy. Expect roughly five minutes; the fixed-point diagnostics
dominate.

Other modes: `tests`, `theory`, `diagnostics`, `audit`, `paper`, `all`.

### Requirements

- Python 3.11 (pinned dependency versions in `requirements.txt`).
- A TeX Live installation with `latexmk` for the PDF build.
- **Java 17 or newer** for the Choco baseline tests. The prebuilt jar is
  compiled for class file version 61; a Java 11 runtime fails with
  `UnsupportedClassVersionError`. `scripts/reproduce.sh` picks up
  `/usr/local/opt/openjdk@17` automatically if `JAVA_HOME` is unset; otherwise
  set `JAVA_HOME` yourself, or rebuild the jar with `make choco-build`.

The pinned versions are the ones the reproduction was last verified against.
They are **not** the versions of the archived sweep, which we believe to have
used `ortools 9.8.3296` and `choco-solver 4.10.13` but cannot establish: no run
record carries a version field, and the archived `protocol.tex` names a
different OR-Tools release. The paper says so too.

## Repository layout

```
src/mirage/          the solver: projectors, geometric consensus, annealing,
                     adaptive regions, CP-SAT hybridization, verifier
baselines/ortools/   OR-Tools CP-SAT baseline adapter
baselines/choco/     Choco baseline (Java/Maven) and its prebuilt jar
baselines/runcsp/    RUN-CSP baseline adapter
experiments/         theory verification, fixed-point diagnostics, figures
experiments/validation/  archive audit, aggregation, asserted paper numbers
scripts/             reproduce.sh, build_submission.py, normalization, verifier
results/raw/         the retained run-level logs of the archived sweep
tests/               unit and regression tests
../paper/ijoc/       manuscript source, generated tables and figures
```

## How the paper's numbers are produced

Every empirical number in the manuscript is a LaTeX macro or a generated table
under `../paper/ijoc/tables/`. Twelve of the thirteen files the manuscript
typesets are regenerated byte-identically by these scripts:

| Script | Produces |
|---|---|
| `experiments/theory_multifactor.py` | `multifactor.tex`, `theory_numbers.tex`, Figure 2 |
| `experiments/theory_portrait.py` | `portrait_numbers.tex`, Figure 3 |
| `experiments/fixedpoint_diagnostics.py` | `fp_numbers.tex`, `fp_protocol.tex`, Figure 4 |
| `experiments/fig_trajectories.py` | `traj_numbers.tex`, Figure 5 |
| `experiments/validation/audit_archive.py` | `audit_numbers.tex`, `data_integrity.tex`, `error_taxonomy.tex`, `region_paired.tex` |
| `experiments/validation/paper_numbers.py` | `main_comparison.tex`, `curated_numbers.tex` |
| `experiments/validation/underflow_analysis.py` | `underflow_numbers.tex` |

The thirteenth, `numbers.tex`, was produced by
`experiments/validation/aggregate.py` during the original sweep and is retained
as archived output; it needs the per-instance feature file, which was not
preserved. Every macro the manuscript still takes from it that concerns
coverage is recomputed and asserted independently by `paper_numbers.py`; what
remains is the declared warm-up time limit, which
Section 6 of the manuscript states.

`tests/test_theorem_scope.py` is the semantic counterpart to those byte
comparisons. It pins the $K_2\cup C_3$ counterexample that forces the
connectedness hypothesis in Theorem 1, checks that every graph in Table 1 is
connected, re-derives the 52,992 schedule-ablation key count, and fails if the
manuscript ever states two different OR-Tools releases. A regenerated table can
be byte-identical and still say something false; these tests are what catch
that.

> **Open item.** The evaluated OR-Tools release is stated as `9.8.3296` here
> and in the paper, while the archived `paper/ijoc/tables/protocol.tex` says
> `9.14.6206`. That table records the environment of the machine that ran the
> aggregation, not the sweep, and the run-level logs carry no version field, so
> the sweep's release cannot be recovered from the artifact. See
> `SUBMISSION.md`.

`paper_numbers.py` is worth singling out. It recomputes the main comparison
cell by cell and the twenty-four coverage macros from `results/raw/`, and
**asserts** each against the value the manuscript was written with. Two
conventions in it matter, and both are traps:

- a verified witness is a run whose terminal status is `SAT_VERIFIED`; only the
  Choco adapter also fills the separate `witness_verified` field, so keying on
  that field alone silently reports zero witnesses for MIRAGE-R;
- instance keys must be canonicalized, because some adapters logged the `.json`
  suffix and others did not.

## Building the paper

```bash
python3 scripts/build_submission.py
```

`paper/ijoc/paper.tex` is the single source and the submission is that one
document: no appendix, no e-companion, no supplemental material. The script
builds `paper.pdf`, writes the flattened `paper_flat.tex` with every generated
table and the bibliography inlined, and **fails** if the counted pages (body
plus references) exceed 25 or an appendix heading reappears.

## What is deliberately not distributed

`results/raw/weekend/`, `weekend_ablation/`, `weekend_memory/` and
`weekend_theory/` hold the **pre-revision** runs. They carried a fabricated
`regions_added` field and broken baseline adapters, they are never mixed into
the paper's numbers (see the comment in `experiments/validation/aggregate.py`),
and **nothing in the manuscript depends on any of them**. They are present in
this repository for completeness rather than excluded from it; the three
scripts that read them (`failure_analysis.py`, `hybrid_deepdive.py` and
`plot_ablation_heatmap.py`) refuse to run and say why.

The archives the paper does use are `results/raw/revision_sweep/`,
`revision_ablation/`, `revision_hybrid_grid/` and `revision_theory/`.

## Known limitations of this archive

- The archived sweep is not reproducible (above), and no diagnostic experiment
  can ever be run on it.
- Round-off-level columns can move in the last digit under a different BLAS.
  `reproduce.sh verify` reports any such change rather than tolerating it.
- The two exploratory microbenchmarks (peak memory, epoch throughput) were each
  measured once, under uncontrolled load, with the hardware only partly
  recorded. No claim in the paper depends on them; they are deposited here for
  completeness.

## Cite

Cite the paper and this archive by their respective DOIs, to be assigned at
acceptance.
