import numpy as np
from src.mirage.csp_core import Constraint, TableConstraint


class EmptyPositiveRelation(Exception):
    """Raised when a positive table constraint has an empty allowed relation.

    An empty positive relation is a syntactic proof of infeasibility. The
    archived implementation replaced its factor summary by the uniform
    distribution, which prevents the contradiction from propagating and lets an
    incomplete run continue to the wall-clock limit on a representation that is
    provably UNSAT (reviewer item A.7). The corrected implementation detects the
    condition during projector construction and terminates the run with the
    distinct status KNOWN_INFEASIBLE_INPUT.
    """

    def __init__(self, scope):
        self.scope = tuple(scope)
        super().__init__(
            f"empty positive relation on scope {self.scope}: "
            "the representation is infeasible by construction")


class LocalProjector:
    """Base class for exact continuous projections onto relational constraints.
    
    In MIRAGE-R, a local projector computes the unnormalized partition function
    over the satisfying assignments of a constraint, effectively marginalizing 
    a tempered probability distribution into the satisfying feasible region.
    """
    def __init__(self, constraint: Constraint):
        self.constraint = constraint
        self.scope = constraint.scope

    def project(self, variable_marginals: dict[str, np.ndarray], tau: float) -> dict[str, np.ndarray]:
        """
        Computes the proposed marginals for variables in the constraint's scope.
        
        This is an exact projection representing the relational join:
        P(x) \propto \sum_{t \in R_c} \prod_{i \in scope} p_i(t_i)^{1/tau}
        
        Args:
            variable_marginals: Current marginals p_i for variables in scope.
            tau: Annealing temperature controlling entropic regularization.
            
        Returns:
            dict mapping variable names to their projected marginal distributions.
        """
        raise NotImplementedError

class TableProjector(LocalProjector):
    """Local projector for tabular constraints (both positive and negative).
    
    Implements a highly efficient Sparse Table Projection for negative constraints
    (conflicts). Instead of materializing the entire allowed Cartesian product 
    (which would scale as O(d^k)), it factorizes the partition function over 
    all possible tuples and subtracts only the conflict weights. 
    This achieves O(n*d + |conflicts|*n) runtime, preventing OOM errors on 
    high-arity relations such as decomposed allDifferent constraints.
    """
    def __init__(self, constraint: TableConstraint, variable_domains: dict[str, list[int]],
                 detect_empty_relation: bool = True):
        super().__init__(constraint)
        self.positive = constraint.positive
        self.detect_empty_relation = detect_empty_relation
        self.var_idx = {v: i for i, v in enumerate(self.scope)}
        self.val2idx = {v: {val: i for i, val in enumerate(variable_domains[v])} for v in self.scope}

        if self.positive:
            self.tuples = list(constraint.tuples_set)
            if not self.tuples and detect_empty_relation:
                # Correction A.7: an empty positive relation proves infeasibility.
                raise EmptyPositiveRelation(self.scope)
        else:
            self.conflicts = list(constraint.tuples_set)
            self.tuples = [] # Not materialized to save memory

    def project(self, variable_marginals: dict[str, np.ndarray], tau: float) -> dict[str, np.ndarray]:
        if self.positive:
            if not self.tuples:
                return {v: np.ones_like(variable_marginals[v]) / len(variable_marginals[v]) for v in self.scope}
                
            tuple_weights = np.zeros(len(self.tuples))
            for t_idx, tup in enumerate(self.tuples):
                log_w = 0.0
                for v_idx, v in enumerate(self.scope):
                    val = tup[v_idx]
                    val_idx = self.val2idx[v][val]
                    p = variable_marginals[v][val_idx]
                    log_w += np.log(max(p, 1e-12))
                tuple_weights[t_idx] = log_w / tau

            max_log_w = np.max(tuple_weights)
            weights = np.exp(tuple_weights - max_log_w)
            sum_w = np.sum(weights)
            if sum_w > 0:
                weights /= sum_w
            else:
                weights = np.ones_like(weights) / len(weights)

            proposals = {v: np.zeros_like(variable_marginals[v]) for v in self.scope}
            for t_idx, tup in enumerate(self.tuples):
                w = weights[t_idx]
                for v_idx, v in enumerate(self.scope):
                    val = tup[v_idx]
                    val_idx = self.val2idx[v][val]
                    proposals[v][val_idx] += w
                    
            for v in self.scope:
                s = np.sum(proposals[v])
                if s > 0:
                    proposals[v] /= s
                else:
                    proposals[v] = np.ones_like(proposals[v]) / len(proposals[v])
            return proposals
        else:
            # Sparse negative representation! Z_total = prod_i (sum_a p_i(a)^{1/tau})
            Z_i = {}
            tempered_p = {}
            for v in self.scope:
                tp = np.power(np.maximum(variable_marginals[v], 1e-12), 1.0 / tau)
                tempered_p[v] = tp
                Z_i[v] = np.sum(tp)

            Z_total_log = np.sum([np.log(Z_i[v]) for v in self.scope])
            
            # Compute weights for conflict tuples
            conflict_weights = []
            with np.errstate(divide='ignore'):
                for tup in self.conflicts:
                    cw_log = 0.0
                    for v_idx, v in enumerate(self.scope):
                        val_idx = self.val2idx[v][tup[v_idx]]
                        cw_log += np.log(tempered_p[v][val_idx])
                    conflict_weights.append(cw_log)

            # Max shift for numerical stability
            max_cw_log = np.max(conflict_weights) if conflict_weights else -np.inf
            shift = max(Z_total_log, max_cw_log)
            
            Z_total_shifted = np.exp(Z_total_log - shift)
            cw_shifted = [np.exp(cw - shift) for cw in conflict_weights]
            
            Z_allowed = Z_total_shifted - np.sum(cw_shifted)
            if Z_allowed <= 0:
                # All mass is in conflicts, return uniform
                return {v: np.ones_like(variable_marginals[v]) / len(variable_marginals[v]) for v in self.scope}
                
            proposals = {}
            for v_idx, v in enumerate(self.scope):
                prop_v = np.zeros_like(variable_marginals[v])
                Z_v_factor = np.exp(Z_total_log - np.log(Z_i[v]) - shift)
                for val_idx in range(len(prop_v)):
                    prop_v[val_idx] = tempered_p[v][val_idx] * Z_v_factor
                
                # Subtract conflicts where v == val
                for c_idx, tup in enumerate(self.conflicts):
                    val_idx = self.val2idx[v][tup[v_idx]]
                    prop_v[val_idx] -= cw_shifted[c_idx]
                    
                prop_v = np.maximum(prop_v, 0.0)
                s = np.sum(prop_v)
                if s > 0:
                    prop_v /= s
                else:
                    prop_v = np.ones_like(prop_v) / len(prop_v)
                proposals[v] = prop_v
                
            return proposals

class BooleanTableProjector(LocalProjector):
    """Fast-track closed-form projector for Boolean table constraints (domain size = 2).
    Avoids O(m) general overhead and allows highly optimized vectorized tensor ops.
    """
    def __init__(self, constraint: TableConstraint, variable_domains: dict[str, list[int]],
                 detect_empty_relation: bool = True):
        super().__init__(constraint)
        self.k = len(self.scope)
        self.allowed = np.zeros(1 << self.k, dtype=bool)

        if constraint.positive:
            for tup in constraint.tuples_set:
                idx = sum(val << i for i, val in enumerate(tup))
                self.allowed[idx] = True
        else:
            self.allowed.fill(True)
            for tup in constraint.tuples_set:
                idx = sum(val << i for i, val in enumerate(tup))
                self.allowed[idx] = False

        if detect_empty_relation and not self.allowed.any():
            # Correction A.7: no satisfying tuple, so the input is infeasible.
            raise EmptyPositiveRelation(self.scope)


        self.bits = np.array([[(idx >> i) & 1 for i in range(self.k)] for idx in range(1 << self.k)])

    def project(self, variable_marginals: dict[str, np.ndarray], tau: float) -> dict[str, np.ndarray]:
        p_tempered = np.zeros((self.k, 2))
        for i, v in enumerate(self.scope):
            p = variable_marginals[v]
            p_tempered[i] = np.power(np.maximum(p, 1e-12), 1.0 / tau)
            
        w = np.ones(1 << self.k)
        for i in range(self.k):
            w *= np.where(self.bits[:, i], p_tempered[i, 1], p_tempered[i, 0])
            
        w *= self.allowed
        sum_w = np.sum(w)
        if sum_w <= 0:
            return {v: np.array([0.5, 0.5]) for v in self.scope}
            
        proposals = {}
        for i, v in enumerate(self.scope):
            # Both branches are summed directly. The earlier implementation
            # used w0 = sum_w - w1, which cancels catastrophically once the
            # marginals are strongly polarized (w1 / sum_w = 1 - O(1e-16)) and
            # cost roughly half the significant digits of the smaller branch.
            mask1 = self.bits[:, i] == 1
            w1 = np.sum(w[mask1])
            w0 = np.sum(w[~mask1])
            total = w0 + w1
            if total <= 0:
                proposals[v] = np.array([0.5, 0.5])
            else:
                proposals[v] = np.array([w0, w1]) / total

        return proposals

