"""Semantic guards on the scope of Theorem 1.

The theorem is stated for a *connected* disequality graph. The hypothesis is
not decoration: on a disconnected graph the spectral condition and the
satisfiability condition come apart, and an earlier draft of this paper stated
the equivalence without connectedness. These tests pin the counterexample so
the hypothesis cannot be dropped again by accident, and check the arithmetic
of the claims the manuscript makes about it.
"""
from __future__ import annotations

import math
import re
import pathlib

import numpy as np
import pytest

from experiments.theory_multifactor import (GRAPHS, complete_graph,
                                            cycle_graph, is_bipartite,
                                            laplacian)

def _paper() -> pathlib.Path:
    """The manuscript, whether it sits in this tree or beside it."""
    here = pathlib.Path(__file__).resolve()
    for base in (here.parents[1], here.parents[2]):
        cand = base / "paper" / "ijoc" / "paper.tex"
        if cand.exists():
            return cand
    return here  # absent: the manuscript tests skip


PAPER = _paper()
HAVE_PAPER = PAPER.name == "paper.tex"


def _disjoint_union(a, b):
    """K_2 u C_3 and friends: relabel the second graph past the first."""
    n_a, e_a = a
    n_b, e_b = b
    return n_a + n_b, sorted(e_a + [(i + n_a, j + n_a) for i, j in e_b])


def _mu_max(n, edges):
    L, _, _ = laplacian(n, edges)
    return float(np.max(np.linalg.eigvals(L).real))


def test_every_verified_graph_is_connected():
    """Table 1 must not contain a graph outside the theorem's hypothesis."""
    for name, (n, edges) in GRAPHS.items():
        adj = {i: set() for i in range(n)}
        for i, j in edges:
            adj[i].add(j)
            adj[j].add(i)
        seen, stack = {0}, [0]
        while stack:
            for w in adj[stack.pop()]:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        assert len(seen) == n, f"{name} is disconnected"


def test_disconnected_counterexample_separates_the_two_conditions():
    """K_2 u C_3: mu_max = 2 (so gamma* = 1/2) while the CSP is unsatisfiable.

    This is exactly the case the connectedness hypothesis rules out. If the
    theorem were stated for arbitrary graphs, part (iii) would be false here.
    """
    n, edges = _disjoint_union(complete_graph(2), cycle_graph(3))
    m = _mu_max(n, edges)

    assert not is_bipartite(n, edges)        # the CSP has no solution
    assert math.isclose(m, 2.0, abs_tol=1e-12)   # yet the spectrum says 2
    assert math.isclose(1.0 / m, 0.5, abs_tol=1e-12)  # so gamma* = 1/2

    # and the component that carries it is the bipartite one
    assert math.isclose(_mu_max(*complete_graph(2)), 2.0, abs_tol=1e-12)
    assert _mu_max(*cycle_graph(3)) < 2.0


@pytest.mark.skipif(not HAVE_PAPER, reason="manuscript not in this repository")
def test_manuscript_states_the_connectedness_hypothesis():
    src = PAPER.read_text()
    assert "finite, connected, undirected graph" in src
    assert "with equality iff $G$ is bipartite" in src
    # and does not revert to the componentwise phrasing inside the theorem
    assert "equality iff $G$ has a bipartite component" not in src


@pytest.mark.skipif(not HAVE_PAPER, reason="manuscript not in this repository")
def test_schedule_ablation_key_arithmetic():
    """736 entries x 12 cells x 3 seeds x 2 configurations = 52,992 keys."""
    assert 736 * 12 * 3 * 2 == 52_992
    assert "736\\times12\\times3\\times2=52{,}992" in PAPER.read_text()


@pytest.mark.skipif(not HAVE_PAPER, reason="manuscript not in this repository")
def test_ortools_release_is_stated_once_and_consistently():
    """The evaluated release must not appear with two different values."""
    src = PAPER.read_text()
    versions = set(re.findall(r"ortools\}?\s*(\d+\.\d+\.\d+)", src))
    assert len(versions) <= 1, f"conflicting OR-Tools versions in the paper: {versions}"


@pytest.mark.skipif(not HAVE_PAPER, reason="manuscript not in this repository")
def test_paired_interval_is_newcombe_not_wald():
    """The region comparison has two discordant pairs out of 736.

    A Wald interval built on the McNemar variance is far too narrow there,
    which is why the manuscript reports Newcombe's method 10. The expected
    bounds below were computed by hand from Newcombe (1998).
    """
    from experiments.validation.audit_archive import newcombe_paired_ci

    lo, hi = newcombe_paired_ci(n11=62, n01=2, n10=0, n00=672)
    assert abs(100 * lo - (-0.34)) < 0.01
    assert abs(100 * hi - 0.92) < 0.01

    # the Wald interval it replaces, for contrast: strictly and wrongly tighter
    n, d = 736, 2 / 736
    se = math.sqrt((2 + 0 - 2 ** 2 / n) / n ** 2)
    assert d - 1.96 * se > lo and d + 1.96 * se < hi

    src = PAPER.read_text()
    assert "newcombe1998paired" in src, "the interval must be cited"
    assert "method 10" in src, "and the variant named, not just the author"
    assert "Wilson score interval for the paired difference" not in src


def test_nonbipartite_generator_actually_produces_odd_cycles():
    """Chords inside one part create an odd cycle only under conditions the
    generator does not enforce by construction, so every seed is checked."""
    import numpy as np
    from experiments.fixedpoint_diagnostics import random_nonbipartite
    from experiments.theory_multifactor import is_bipartite

    for seed in range(8):
        n, edges = random_nonbipartite(30, 3, np.random.default_rng(seed))
        assert not is_bipartite(n, edges), f"seed {seed} stayed 2-colourable"
        # Non-bipartite is not enough: a bipartite component anywhere would
        # still put mu_max at 2 and make gamma = 1/2 critical, not subcritical.
        # Check the spectral quantity the interpretation actually rests on.
        L, _, _ = laplacian(n, edges)
        mu = float(np.max(np.linalg.eigvals(L).real))
        assert mu < 2 - 1e-9, f"seed {seed} has mu_max = {mu}, not subcritical"


@pytest.mark.skipif(not HAVE_PAPER, reason="manuscript not in this repository")
def test_bipartite_critical_state_is_not_uniform():
    """At the exactly critical ratio the bipartition direction is neutral, so
    the terminal state keeps the component the initialization gave it. The
    manuscript must not claim convergence to the uniform point there."""
    import json
    import statistics

    root = pathlib.Path(__file__).resolve().parents[1]
    recs = [json.loads(l) for l in
            open(root / "results/raw/diagnostics/diagnostics.jsonl")]
    crit = [r for r in recs if r["config"] == r"bipartite, frozen $\gamma=1/2$"]
    gap = statistics.median(r["gap_mean"] for r in crit)
    assert gap < 0.5, "a neutral direction cannot reach the uniform point"
    assert 0.5 - gap > 1e-7, "the surviving component should be measurable"

    src = PAPER.read_text()
    assert "the two disequality families instead converge to the uniform" not in src


@pytest.mark.skipif(not HAVE_PAPER, reason="manuscript not in this repository")
def test_warm_start_grid_numbers_are_deduplicated_and_decision_rates():
    """The archived aggregation rated the grid on duplicated rows and under the
    wrong metric. These are the deduplicated decision rates the paper quotes."""
    import collections
    import statistics

    from experiments.validation.audit_archive import load_jsonl, canon

    root = pathlib.Path(__file__).resolve().parents[1]
    rows = load_jsonl(str(root / "results/raw/revision_hybrid_grid/*.jsonl"))
    keyed = {}
    for r in rows:
        keyed.setdefault((r.get("warmup_epochs"), r.get("hint_mode"),
                          canon(r["instance"]), r["seed"]), r)
    assert len(rows) == 7800 and len(keyed) == 6240, "the grid carries duplicates"

    cells = collections.defaultdict(list)
    for (w, h, _i, _s), r in keyed.items():
        cells[(w, h)].append(r)
    decided = {c: sum(1 for r in v
                      if r["status"] in ("SAT_VERIFIED", "UNSAT", "UNSAT_PROVED"))
               for c, v in cells.items()}
    assert all(len(v) == 480 for v in cells.values())

    control = decided[(0, "none")]
    hinted = {c: n for c, n in decided.items() if c[1] != "none"}
    assert max(hinted.values()) <= control, (
        "no hinted cell may be reported as beating the control")
    assert abs(100 * control / 480 - 95.2) < 0.05

    # and the witness rate, which is the metric the paragraph used to name
    wit = sum(1 for r in cells[(0, "none")] if r["status"] == "SAT_VERIFIED")
    assert abs(100 * wit / 480 - 61.5) < 0.05, "witness and decision differ here"

    src = PAPER.read_text()
    assert "RevHybBestRate" not in src, "the duplicated-record rate must be gone"
    assert "RevHybGridRuns" in src
