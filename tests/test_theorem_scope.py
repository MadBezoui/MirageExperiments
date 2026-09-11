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

from experiments.theory_multifactor import (GRAPHS, complete_graph,
                                            cycle_graph, is_bipartite,
                                            laplacian)

PAPER = pathlib.Path(__file__).resolve().parents[2] / "paper" / "ijoc" / "paper.tex"


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


def test_manuscript_states_the_connectedness_hypothesis():
    src = PAPER.read_text()
    assert "finite, connected, undirected graph" in src
    assert "with equality iff $G$ is bipartite" in src
    # and does not revert to the componentwise phrasing inside the theorem
    assert "equality iff $G$ has a bipartite component" not in src


def test_schedule_ablation_key_arithmetic():
    """736 entries x 12 cells x 3 seeds x 2 configurations = 52,992 keys."""
    assert 736 * 12 * 3 * 2 == 52_992
    assert "736\\times12\\times3\\times2=52{,}992" in PAPER.read_text()


def test_ortools_release_is_stated_once_and_consistently():
    """The evaluated release must not appear with two different values."""
    src = PAPER.read_text()
    versions = set(re.findall(r"ortools\}?\s*(\d+\.\d+\.\d+)", src))
    assert len(versions) <= 1, f"conflicting OR-Tools versions in the paper: {versions}"


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
    assert "Newcombe" in src
    assert "Wilson score interval for the paired difference" not in src
