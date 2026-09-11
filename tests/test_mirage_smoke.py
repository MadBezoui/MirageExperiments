import pytest
from src.mirage.csp_core import CSPInstance, Variable, TableConstraint
from src.mirage.annealing import Hyperparameters
from src.mirage.mirage_solver import MirageSolver

def test_mirage_smoke_sat():
    # Simple XOR problem: x != y, x,y in {0,1}
    v1 = Variable("v1", [0, 1])
    v2 = Variable("v2", [0, 1])
    # x != y -> (0,1), (1,0)
    c1 = TableConstraint("c1", ["v1", "v2"], True, [(0, 1), (1, 0)])
    
    inst = CSPInstance("xor", "mock", {"v1": v1, "v2": v2}, [c1], {})
    
    hp = Hyperparameters(max_epochs=50, decoding_frequency=1)
    solver = MirageSolver(inst, hp)
    
    result = solver.run()
    assert result["status"] == "SAT_VERIFIED"
    assert result["violations"] == 0

def test_mirage_smoke_unsat():
    # x == 0 and x == 1
    v1 = Variable("v1", [0, 1])
    c1 = TableConstraint("c1", ["v1"], True, [(0,)])
    c2 = TableConstraint("c2", ["v1"], True, [(1,)])
    
    inst = CSPInstance("unsat", "mock", {"v1": v1}, [c1, c2], {})
    
    hp = Hyperparameters(max_epochs=10, decoding_frequency=1)
    solver = MirageSolver(inst, hp)
    
    result = solver.run()
    # It should timeout/max epochs
    assert result["status"] == "UNKNOWN_MAX_EPOCHS"
    assert result["violations"] > 0
