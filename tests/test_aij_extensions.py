"""Tests for the AIJ-extension components: OR-Tools baseline, dynamic regions,
and the MIRAGE-R x CP-SAT hybrid."""
import json
import os
import sys

import numpy as np
import pytest

from src.mirage.csp_core import CSPInstance, Variable, TableConstraint
from src.mirage.mirage_solver import MirageSolver
from src.mirage.annealing import Hyperparameters
from src.mirage.adaptive_regions import AdaptiveRegionManager

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "baselines", "ortools"))


def _tiny_sat():
    # x,y in {0,1}; table1: x==y allowed {(0,0),(1,1)}; table2: y==1 allowed {(1,)}
    # only solution: x=1,y=1
    variables = {
        "x": Variable("x", [0, 1]),
        "y": Variable("y", [0, 1]),
    }
    c1 = TableConstraint("c1", ["x", "y"], True, [(0, 0), (1, 1)])
    c2 = TableConstraint("c2", ["y"], True, [(1,)])
    return CSPInstance("tiny", "mem", variables, [c1, c2], {})


def test_ortools_solves_tiny():
    from run_ortools import solve_instance
    inst = _tiny_sat()
    path = "/tmp/_tiny_aij.json"
    json.dump(inst.to_normalized_json(), open(path, "w"))
    rec = solve_instance(path, timeout=5, workers=1)
    assert rec["status"] in ("SAT_VERIFIED",)


def test_region_join_is_exact():
    inst = _tiny_sat()
    mgr = AdaptiveRegionManager(inst, tuple_cap=1000)
    region = mgr.join_two(inst.constraints[0], inst.constraints[1])
    assert region is not None
    assert set(region.scope) == {"x", "y"}
    # the only joint satisfying tuple is x=1,y=1
    xi = region.scope.index("x"); yi = region.scope.index("y")
    sols = {(t[xi], t[yi]) for t in region.tuples}
    assert sols == {(1, 1)}


def test_solver_runs_with_adaptive_regions():
    inst = _tiny_sat()
    np.random.seed(0)
    hp = Hyperparameters(timeout=5, max_epochs=200, use_adaptive_regions=True,
                         stagnation_window=3)
    res = MirageSolver(inst, hp).run()
    assert res["status"] == "SAT_VERIFIED"
    assert "regions_added" in res


def test_hybrid_solves_tiny():
    from src.mirage.hybrid import solve_hybrid
    inst = _tiny_sat()
    data = inst.to_normalized_json()
    res = solve_hybrid(inst, data, warmup_epochs=5, timeout=5, workers=1)
    assert res["status"] == "SAT_VERIFIED"
