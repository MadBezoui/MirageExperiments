"""MIRAGE-R x CP-SAT hybridization.

Run MIRAGE-R for a short continuous "warm-up", then hand its marginals to a
complete solver (OR-Tools CP-SAT) as a search heuristic. Four hint modes
(the full grid of revision item P3.5 / Table 1):

  * ``value``  -- each variable is value-hinted to its argmax marginal
                  (CP-SAT ``AddHint``);
  * ``order``  -- variables are branched in decreasing order of marginal
                  confidence max_a p_i(a) via ``AddDecisionStrategy``;
  * ``both``   -- value hints + branching order;
  * ``none``   -- pure CP-SAT (warm-up marginals are computed but unused;
                  with ``warmup_epochs=0`` this is exactly the CP-SAT baseline
                  including its wall-clock accounting).

Time accounting is honest: the CP-SAT budget is the remaining wall-clock after
the warm-up, so the hybrid never gets more total time than the baselines.
"""
from __future__ import annotations
import os
import sys
import time
from typing import Any, Dict

import numpy as np
from ortools.sat.python import cp_model

from src.mirage.csp_core import CSPInstance
from src.mirage.mirage_solver import MirageSolver
from src.mirage.annealing import Hyperparameters

# reuse the validated CP-SAT model builder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "baselines", "ortools"))

HINT_MODES = ("value", "order", "both", "none")


def warmup_marginals(instance: CSPInstance, epochs: int,
                     time_budget: float) -> Dict[str, np.ndarray]:
    hp = Hyperparameters(max_epochs=max(epochs, 1),
                         timeout=max(time_budget, 0.1))
    solver = MirageSolver(instance, hp)
    solver.run()
    return solver.marginals


def solve_hybrid(instance: CSPInstance, data: Dict[str, Any],
                 warmup_epochs: int = 30, timeout: float = 60.0,
                 seed: int = 0, workers: int = 8,
                 hint_mode: str = "both") -> Dict[str, Any]:
    """Build the CP-SAT model and seed it with MIRAGE-R marginals."""
    from run_ortools import build_model  # validated builder

    if hint_mode not in HINT_MODES:
        raise ValueError(f"hint_mode must be one of {HINT_MODES}")

    t0 = time.time()
    np.random.seed(seed)

    marg: Dict[str, np.ndarray] = {}
    if warmup_epochs > 0:
        # warm-up may use at most 20% of the total budget
        marg = warmup_marginals(instance, warmup_epochs,
                                time_budget=0.2 * timeout)
    warmup_time = time.time() - t0

    model, var_map, unsupported = build_model(data)
    if unsupported:
        return {"status": "ERROR", "error": f"unsupported: {unsupported}",
                "time": time.time() - t0}

    if marg and hint_mode in ("value", "both"):
        for name, v in instance.variables.items():
            p = marg[name]
            model.AddHint(var_map[name], v.idx2val[int(np.argmax(p))])

    if marg and hint_mode in ("order", "both"):
        confidences = sorted(
            ((float(np.max(marg[name])), name) for name in instance.variables),
            reverse=True)
        ordered_vars = [var_map[n] for _, n in confidences]
        model.AddDecisionStrategy(ordered_vars,
                                  cp_model.CHOOSE_FIRST,
                                  cp_model.SELECT_MIN_VALUE)

    solver = cp_model.CpSolver()
    remaining = max(timeout - (time.time() - t0), 1.0)
    solver.parameters.max_time_in_seconds = float(remaining)
    solver.parameters.random_seed = int(seed)
    solver.parameters.num_search_workers = int(workers)
    solver.parameters.cp_model_presolve = True

    status = solver.Solve(model)
    elapsed = time.time() - t0
    out = ("SAT_VERIFIED" if status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
           else "UNSAT" if status == cp_model.INFEASIBLE
           else "UNKNOWN_TIMEOUT")
    rec: Dict[str, Any] = {
        "status": out, "time": elapsed, "warmup_epochs": warmup_epochs,
        "warmup_time": warmup_time, "hint_mode": hint_mode,
        "branches": solver.NumBranches(), "conflicts": solver.NumConflicts(),
    }
    if out == "SAT_VERIFIED":
        rec["solution"] = {n: int(solver.Value(v)) for n, v in var_map.items()}
        rec["violations"] = 0
    return rec
