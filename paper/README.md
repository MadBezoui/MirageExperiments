# `paper/` — what the manuscript is built from

This directory holds everything the manuscript takes from the artifact, so that
`scripts/reproduce.sh verify` has something to compare its regenerated output
against.

```
paper/ijoc/
├── tables/          the 13 generated table files the manuscript \input
├── figures/         the 4 figures it includes
└── references.bib   its bibliography
```

The manuscript source itself (`paper.tex`) is not distributed here. Everything
else is, which is what the verification needs:

- `./scripts/reproduce.sh verify` regenerates every table into a temporary
  directory and requires it to come back byte-identical to the copy above. It
  compares the whole of `tables/` when `paper.tex` is absent, and the subset the
  manuscript `\input`s when it is present. Both are the same 13 files here.
- `./scripts/reproduce.sh paper` additionally builds the PDF, and is the one
  mode that needs `paper.tex`. It says so and stops if it is missing.

Twelve of the thirteen table files regenerate byte-identically from the retained
raw logs. The thirteenth, `numbers.tex`, is archived aggregation output whose
per-instance feature file was not preserved; the coverage values the manuscript
takes from it are recomputed and asserted independently by
`experiments/validation/paper_numbers.py`, while its declared time limits are
not recoverable. `Code/README.md` in the manuscript tree describes the split.
