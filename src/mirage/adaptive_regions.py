"""Dynamic region generation for MIRAGE-R (TODO_AIJ.md sec 1).

When the solver stalls (the running-minimum violation count stops improving),
classical continuous relaxations are stuck at a *fractional local minimum*: no
single-variable move decreases the soft loss, yet no integral solution is found.
The cure is to enlarge the relaxation's granularity by materializing a *region*:
the relational join of two overlapping constraints. A region constraint is exact
over a larger sub-scope, so its local projector can escape minima that the
factored projectors cannot see.

`AdaptiveRegionManager` detects stagnation and proposes new joined regions,
bounded by `tuple_cap` so memory cannot blow up.
"""
from __future__ import annotations
from itertools import product
from typing import Dict, List, Optional

from src.mirage.csp_core import CSPInstance, TableConstraint


def _positive_tuples(c: TableConstraint, domains: Dict[str, list]) -> List[tuple]:
    """Return the satisfying (allowed) tuples of a table constraint."""
    if c.positive:
        return list(c.tuples_set)
    conflicts = c.tuples_set
    return [t for t in product(*(domains[v] for v in c.scope)) if t not in conflicts]


class AdaptiveRegionManager:
    def __init__(self, instance: CSPInstance, tuple_cap: int = 100_000):
        self.instance = instance
        self.tuple_cap = tuple_cap
        self.domains = {n: v.domain for n, v in instance.variables.items()}
        self.added_signatures = set()  # avoid re-adding the same region
        self.added_regions: List[TableConstraint] = []

    # ------------------------------------------------------------------ #
    def check_stagnation(self, history_violations: List[int], window: int = 10) -> bool:
        """True if the best violation count has not improved over `window` checks."""
        if len(history_violations) < window:
            return False
        recent = history_violations[-window:]
        prior = history_violations[:-window]
        best_prior = min(prior) if prior else float("inf")
        # stagnation: no *strict* improvement on the prior best during the window
        return min(recent) >= best_prior or max(recent) == min(recent)

    # ------------------------------------------------------------------ #
    def join_two(self, ca: TableConstraint, cb: TableConstraint) -> Optional[TableConstraint]:
        """Natural join of two table constraints over the union of their scopes.

        Returns a new *positive* TableConstraint, or None if the scopes don't
        overlap or the joined table would exceed `tuple_cap`.
        """
        shared = [v for v in ca.scope if v in set(cb.scope)]
        if not shared:
            return None
        merged_scope = list(ca.scope) + [v for v in cb.scope if v not in set(ca.scope)]
        sig = tuple(sorted(merged_scope))
        if sig in self.added_signatures:
            return None

        ta = _positive_tuples(ca, self.domains)
        tb = _positive_tuples(cb, self.domains)
        # index cb tuples by their values on the shared variables
        b_idx = {v: i for i, v in enumerate(cb.scope)}
        a_idx = {v: i for i, v in enumerate(ca.scope)}
        from collections import defaultdict
        bucket = defaultdict(list)
        for t in tb:
            key = tuple(t[b_idx[v]] for v in shared)
            bucket[key].append(t)

        out = []
        b_extra = [v for v in cb.scope if v not in set(ca.scope)]
        b_extra_idx = [b_idx[v] for v in b_extra]
        for t in ta:
            key = tuple(t[a_idx[v]] for v in shared)
            for tb_row in bucket.get(key, ()):
                out.append(t + tuple(tb_row[i] for i in b_extra_idx))
                if len(out) > self.tuple_cap:
                    return None  # too large; skip this region

        if not out:
            # join is empty -> the two constraints are jointly UNSAT on the shared
            # scope; still useful to surface, but represent as an empty positive table
            pass
        self.added_signatures.add(sig)
        region = TableConstraint(
            id=f"region({ca.id}+{cb.id})", scope=merged_scope,
            positive=True, tuples=out,
        )
        return region

    # ------------------------------------------------------------------ #
    def expand_regions(self, violated_constraint_ids: List[str],
                       variable_marginals: Dict[str, "np.ndarray"],
                       max_new: int = 4) -> List[TableConstraint]:
        """Build new region constraints by joining the most-violated table
        constraints with an overlapping neighbour. Returns up to `max_new`."""
        import numpy as np
        viol = set(violated_constraint_ids)
        tables = [c for c in self.instance.constraints if isinstance(c, TableConstraint)]
        by_id = {c.id: c for c in tables}
        
        # Calculate entropy of variable marginals
        var_entropy = {}
        for v, p in variable_marginals.items():
            p_safe = np.maximum(p, 1e-12)
            var_entropy[v] = -np.sum(p_safe * np.log(p_safe))

        # var -> constraints adjacency
        var2cons: Dict[str, List[TableConstraint]] = {}
        for c in tables:
            for v in c.scope:
                var2cons.setdefault(v, []).append(c)

        new_regions: List[TableConstraint] = []
        for cid in violated_constraint_ids:
            ca = by_id.get(cid)
            if ca is None:
                continue
            # candidate neighbours: share a variable
            neighbours = []
            for v in ca.scope:
                for cb in var2cons.get(v, ()):
                    if cb.id != ca.id:
                        neighbours.append(cb)
            
            # Score neighbours by the maximum entropy of shared variables (fractional saddle)
            # breaking ties by whether they are also violated
            def score_cb(cb):
                shared = [v for v in ca.scope if v in set(cb.scope)]
                ent = max([var_entropy.get(v, 0.0) for v in shared], default=0.0)
                is_viol = 1.0 if cb.id in viol else 0.0
                return (ent, is_viol)
                
            neighbours.sort(key=score_cb, reverse=True)
            for cb in neighbours:
                region = self.join_two(ca, cb)
                if region is not None:
                    new_regions.append(region)
                    break
            if len(new_regions) >= max_new:
                break
        self.added_regions.extend(new_regions)
        return new_regions
