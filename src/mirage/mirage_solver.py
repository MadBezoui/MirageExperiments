import random
import math
import time
import numpy as np
from typing import Dict, List, Tuple, Set, Any
from src.mirage.csp_core import CSPInstance, TableConstraint
from src.mirage.annealing import Hyperparameters, AnnealingSchedule
from src.mirage.consensus import (geometric_consensus, apply_polarization,
                                  FallbackCounter)
from src.mirage.local_projectors import (TableProjector, BooleanTableProjector,
                                         EmptyPositiveRelation)
from src.mirage.verifier import count_violations, verify_assignment, violated_constraints
from src.mirage.adaptive_regions import AdaptiveRegionManager

class MirageSolver:
    """The MIRAGE-R (Mirror-Reflected Region Annealing) continuous CSP solver.
    
    MIRAGE-R operates by relaxing the discrete CSP into a continuous soft-satisfaction
    objective. Each iteration (epoch) performs exact local projection onto constraint 
    relations, a weighted geometric consensus to exchange messages between variables,
    and a mirror-reflected polarization step that actively repels fractional stationary
    points. When the solver stagnates, it dynamically materializes joined constraint
    regions (Adaptive Regions) to restore discrete structure.
    """
    def __init__(self, instance: CSPInstance, hp: Hyperparameters):
        self.instance = instance
        self.hp = hp
        self.schedule = AnnealingSchedule(hp)
        
        self.variables = instance.variables
        self.var_names = list(self.variables.keys())
        
        # Initialize marginals uniformly with tiny noise to break symmetry
        self.marginals = {}
        for name, v in self.variables.items():
            domain_size = len(v.domain)
            noise = np.random.uniform(0, 1e-4, domain_size)
            p = np.ones(domain_size) / domain_size + noise
            self.marginals[name] = p / np.sum(p)
            
        # Numerical-guard accounting (correction A.8): how often a fallback fired.
        self.fallbacks = FallbackCounter()
        # Correction A.7: an empty positive relation proves infeasibility.
        self.infeasible_scope = None
        detect_empty = getattr(hp, "detect_empty_relation", True)

        # Build projectors (Phase 4 focuses on Table constraints and regions)
        self.projectors = []
        for c in instance.constraints:
            if isinstance(c, TableConstraint):
                domain_dict = {v: self.variables[v].domain for v in c.scope}
                is_boolean = all(len(domain_dict[v]) == 2 for v in c.scope)
                cls = BooleanTableProjector if is_boolean else TableProjector
                try:
                    self.projectors.append(cls(c, domain_dict, detect_empty))
                except EmptyPositiveRelation as exc:
                    self.infeasible_scope = exc.scope
                    break
            else:
                # In full MIRAGE-R we would have projectors for Sum, Count, etc.
                # For Phase 4 we just skip non-Table if we are doing generic Table projection.
                # Crosswords only have Table constraints after conversion.
                pass

        # Adjacency for consensus
        self.var_to_projectors = {v: [] for v in self.var_names}
        for proj in self.projectors:
            for v in proj.scope:
                self.var_to_projectors[v].append(proj)

        # Dynamic region generation (AIJ extension)
        self.violation_history: list[int] = []
        self.region_mgr = (AdaptiveRegionManager(instance, hp.tuple_cap)
                           if getattr(hp, "use_adaptive_regions", False) else None)
        self.regions_added = 0

    def _maybe_expand_regions(self, assignment):
        """On stagnation, materialize joined regions and add their projectors."""
        if self.region_mgr is None:
            return
        if not self.region_mgr.check_stagnation(self.violation_history,
                                                self.hp.stagnation_window):
            return
        vids = violated_constraints(self.instance, assignment)
        new_regions = self.region_mgr.expand_regions(vids, self.marginals)
        for region in new_regions:
            domain_dict = {v: self.variables[v].domain for v in region.scope}
            proj = TableProjector(region, domain_dict)
            self.projectors.append(proj)
            for v in region.scope:
                self.var_to_projectors.setdefault(v, []).append(proj)
            self.regions_added += 1

    def decode_assignment(self) -> dict[str, int]:
        """Decode the current marginals to a discrete assignment by taking argmax."""
        assignment = {}
        for name in self.var_names:
            idx = np.argmax(self.marginals[name])
            assignment[name] = self.variables[name].idx2val[idx]
        return assignment

    def run(self) -> dict[str, Any]:
        start_time = time.time()
        best_violations = float('inf')
        best_assignment = None

        # Correction A.7: terminate immediately on a syntactically infeasible
        # input instead of iterating on a uniform summary until the time limit.
        if self.infeasible_scope is not None:
            return {
                "status": "KNOWN_INFEASIBLE_INPUT",
                "violations": -1,
                "time": time.time() - start_time,
                "epochs": 0,
                "regions_added": 0,
                "reason": "empty positive relation on scope "
                          f"{self.infeasible_scope}",
                **self.fallbacks.as_dict(),
            }

        epoch = 0
        while epoch < self.hp.max_epochs:
            if time.time() - start_time > self.hp.timeout:
                return {
                    "status": "UNKNOWN_TIMEOUT",
                    "violations": best_violations,
                    "time": time.time() - start_time,
                    "epochs": epoch,
                    "regions_added": self.regions_added,
                    **self.fallbacks.as_dict(),
                }
                
            # 1. Local Projections
            proposals = {v: [] for v in self.var_names}
            for proj in self.projectors:
                proj_marginals = {v: self.marginals[v] for v in proj.scope}
                proj_out = proj.project(proj_marginals, self.schedule.tau)
                for v in proj.scope:
                    proposals[v].append(proj_out[v])
                    
            # 2. Consensus & Polarization
            new_marginals = {}
            for v in self.var_names:
                props = proposals[v]
                if not props:
                    new_marginals[v] = self.marginals[v]
                    continue
                # Geometric consensus (stable log-sum-exp normalization)
                weights = [1.0 / len(props)] * len(props)
                p_cons = geometric_consensus(props, weights, self.hp.epsilon,
                                             self.fallbacks)

                # Polarization followed by the clipping operator of Eq. (1)
                p_pol = apply_polarization(p_cons, self.schedule.beta,
                                           self.hp.epsilon, self.fallbacks)
                if getattr(self.hp, "strict_finite", False):
                    assert np.all(np.isfinite(p_pol)), \
                        f"non-finite marginal for variable {v}"
                new_marginals[v] = p_pol
                
            self.marginals = new_marginals
            
            # Decoding and Verification
            if epoch % self.hp.decoding_frequency == 0:
                assignment = self.decode_assignment()
                violations = count_violations(self.instance, assignment)
                if violations < best_violations:
                    best_violations = violations
                    best_assignment = assignment
                self.violation_history.append(violations)
                self._maybe_expand_regions(assignment)
                
                if violations == 0:
                    # Double check with independent verifier
                    if verify_assignment(self.instance, assignment):
                        return {
                            "status": "SAT_VERIFIED",
                            "violations": 0,
                            "time": time.time() - start_time,
                            "epochs": epoch,
                            "regions_added": self.regions_added,
                            **self.fallbacks.as_dict(),
                        }
                        
            # Annealing step
            self.schedule.step()
            epoch += 1
            
        return {
            "status": "UNKNOWN_MAX_EPOCHS",
            "violations": best_violations,
            "time": time.time() - start_time,
            "epochs": epoch,
            "regions_added": self.regions_added,
            **self.fallbacks.as_dict(),
        }
